package actorcritic

import (
	"math/rand/v2"
	"sync"

	"github.com/reallyoldfogie/cRL-go/pkg/mat"
)

// LSTMLayer holds one LSTM cell's four gates (forget, input, output, and
// the candidate/"new memory content" gate, usually written C-tilde or g
// in LSTM literature — named G here to keep C free for the cell-state
// variable itself elsewhere). Each gate has an input-side weight (Wx*,
// applied to the trunk's output), a recurrent-side weight (Wh*, applied
// to the previous step's hidden state), and a bias — see
// docs/plans/20-configurable-depth-step-memory-and-dynamic-architecture.md,
// Part B.
type LSTMLayer struct {
	WxF, WhF, BF *mat.Matrix
	WxI, WhI, BI *mat.Matrix
	WxO, WhO, BO *mat.Matrix
	WxG, WhG, BG *mat.Matrix
}

// NewLSTMLayer allocates an LSTMLayer consuming inputSize-wide input
// (the MLP trunk's HiddenSize) and producing hiddenSize-wide hidden/cell
// state, using the same Xavier/Glorot uniform initialization as
// NewParams for every weight matrix.
//
// The forget gate's bias (BF) is initialized to 1.0, not 0 like every
// other bias here — the well-known Jozefowicz et al. 2015 ("An Empirical
// Exploration of Recurrent Network Architectures") initialization trick.
// A freshly-initialized LSTM with BF=0 starts by forgetting roughly half
// its cell state every step (sigmoid(0)=0.5), which empirically makes
// early training much harder to get off the ground than starting biased
// toward retaining memory (sigmoid(1)≈0.73) and letting training learn
// to forget where appropriate.
func NewLSTMLayer(rng *rand.Rand, inputSize, hiddenSize int) *LSTMLayer {
	newGateWeights := func() (wx, wh, b *mat.Matrix) {
		wx = mat.New(hiddenSize, inputSize)
		wh = mat.New(hiddenSize, hiddenSize)
		b = mat.New(hiddenSize, 1)
		wx.FillRand(rng, -xavierBound(inputSize, hiddenSize), xavierBound(inputSize, hiddenSize))
		wh.FillRand(rng, -xavierBound(hiddenSize, hiddenSize), xavierBound(hiddenSize, hiddenSize))
		return wx, wh, b
	}

	l := &LSTMLayer{}
	l.WxF, l.WhF, l.BF = newGateWeights()
	l.BF.Fill(1.0)
	l.WxI, l.WhI, l.BI = newGateWeights()
	l.WxO, l.WhO, l.BO = newGateWeights()
	l.WxG, l.WhG, l.BG = newGateWeights()
	return l
}

// HiddenSize reports the LSTM's hidden/cell state width.
func (l *LSTMLayer) HiddenSize() int { return l.WhF.Rows }

// InputSize reports the width of the input this LSTM consumes (the MLP
// trunk's HiddenSize, in RecurrentParams).
func (l *LSTMLayer) InputSize() int { return l.WxF.Cols }

func (l *LSTMLayer) clone() *LSTMLayer {
	return &LSTMLayer{
		WxF: cloneMatrix(l.WxF), WhF: cloneMatrix(l.WhF), BF: cloneMatrix(l.BF),
		WxI: cloneMatrix(l.WxI), WhI: cloneMatrix(l.WhI), BI: cloneMatrix(l.BI),
		WxO: cloneMatrix(l.WxO), WhO: cloneMatrix(l.WhO), BO: cloneMatrix(l.BO),
		WxG: cloneMatrix(l.WxG), WhG: cloneMatrix(l.WhG), BG: cloneMatrix(l.BG),
	}
}

// RecurrentParams wraps an actor-critic Trunk (the existing MLP +
// policy/value heads — see Params) with an LSTM layer sitting strictly
// between the trunk's final hidden layer and its heads:
//
//	input -> Trunk.Hidden[...] -> ReLU -> LSTM(hPrev, cPrev) -> h
//	                                                     -> Trunk.Wpi,Bpi -> Softmax -> policy
//	                                                     -> Trunk.Wv,Bv   -> value
//
// LSTM's hidden width always equals Trunk.HiddenSize() — see
// NewRecurrentParams. The hidden/cell state (h, c) itself is not part of
// RecurrentParams: it's per-episode, per-environment runtime state the
// caller carries across steps (see pkg/ppo's recurrent rollout
// collection), not a learned parameter.
//
// mu is the sole lock for this bundle: Trunk's own embedded mutex is
// never used once wrapped here (see Lock/Unlock/Snapshot), so a caller
// never needs to reason about two separate locks for one RecurrentParams.
type RecurrentParams struct {
	mu sync.RWMutex

	Trunk *Params
	LSTM  *LSTMLayer
}

// NewRecurrentParams allocates a RecurrentParams for a network with the
// given layer sizes: an inputSize/hiddenSize/outputSize/numHiddenLayers
// Trunk (see NewParams) plus an LSTM layer sized hiddenSize x hiddenSize.
func NewRecurrentParams(rng *rand.Rand, inputSize, hiddenSize, outputSize, numHiddenLayers int) *RecurrentParams {
	return &RecurrentParams{
		Trunk: NewParams(rng, inputSize, hiddenSize, outputSize, numHiddenLayers),
		LSTM:  NewLSTMLayer(rng, hiddenSize, hiddenSize),
	}
}

// WarmStartRecurrentParams builds a RecurrentParams by reusing trunk
// as-is (a deep copy, via Snapshot — trunk itself is never mutated,
// matching Net2WiderNet/Net2DeeperNet's own contract) and adding a
// freshly, randomly initialized LSTM layer sized to match.
//
// Unlike Net2WiderNet/Net2DeeperNet, this is NOT function-preserving,
// and cannot be made so for a standard LSTM cell: Net2DeeperNet's
// inserted layer is an exact identity because ReLU(I·a + 0) = a for any
// a >= 0 (ReLU's own output is already non-negative), but an LSTM's
// hidden output is hOut = o*tanh(cOut), with both o (sigmoid) and tanh
// bounded into (0,1)/(-1,1) — there is no setting of the LSTM's weights
// that makes a bounded quantity equal an arbitrary *unbounded* trunk
// output for every input. So wrapping a trained trunk in a fresh LSTM
// changes what the combined network computes from the very first
// forward pass; this is a warm start (keep the trunk's learned
// features, let the LSTM learn its own role from scratch), not a
// lossless upgrade — expect a real retraining period before the
// combined network's behavior stabilizes, exactly as it would training
// a recurrent network from scratch, just with a head start on the
// per-step feature extraction the trunk already learned.
func WarmStartRecurrentParams(trunk *Params, rng *rand.Rand) *RecurrentParams {
	return &RecurrentParams{
		Trunk: trunk.Snapshot(),
		LSTM:  NewLSTMLayer(rng, trunk.HiddenSize(), trunk.HiddenSize()),
	}
}

// InputSize, HiddenSize, OutputSize, and NumHiddenLayers delegate to
// Trunk; see Params' own identically-named methods.
func (p *RecurrentParams) InputSize() int       { return p.Trunk.InputSize() }
func (p *RecurrentParams) HiddenSize() int      { return p.Trunk.HiddenSize() }
func (p *RecurrentParams) OutputSize() int      { return p.Trunk.OutputSize() }
func (p *RecurrentParams) NumHiddenLayers() int { return p.Trunk.NumHiddenLayers() }

// Lock acquires p's write lock; see Params.Lock's own doc comment — the
// same concurrent-snapshot-safety reasoning applies here, covering both
// Trunk's and LSTM's matrices together as one unit.
func (p *RecurrentParams) Lock() { p.mu.Lock() }

// Unlock releases the write lock acquired by Lock.
func (p *RecurrentParams) Unlock() { p.mu.Unlock() }

// Snapshot returns a deep copy of p's current weights (both Trunk's and
// LSTM's matrices), sharing nothing with p; see Params.Snapshot's own
// doc comment. Clones Trunk's matrices directly, rather than calling
// Trunk.Snapshot(), to keep p.mu the only lock that matters for this
// bundle.
func (p *RecurrentParams) Snapshot() *RecurrentParams {
	p.mu.RLock()
	defer p.mu.RUnlock()

	hidden := make([]Layer, len(p.Trunk.Hidden))
	for i, layer := range p.Trunk.Hidden {
		hidden[i] = Layer{W: cloneMatrix(layer.W), B: cloneMatrix(layer.B)}
	}

	return &RecurrentParams{
		Trunk: &Params{
			Hidden: hidden,
			Wpi:    cloneMatrix(p.Trunk.Wpi),
			Bpi:    cloneMatrix(p.Trunk.Bpi),
			Wv:     cloneMatrix(p.Trunk.Wv),
			Bv:     cloneMatrix(p.Trunk.Bv),
		},
		LSTM: p.LSTM.clone(),
	}
}
