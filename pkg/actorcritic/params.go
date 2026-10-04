// Package actorcritic builds a 3-layer actor-critic MLP: the same
// shared trunk shape as pkg/policy's network, feeding two independent
// heads instead of one — a softmax policy head (matching pkg/policy's
// only head) and a scalar value head predicting expected return.
//
// This is a separate package rather than an extension of pkg/policy so
// that pkg/policy can remain exactly what docs/05-porting-notes.md's
// "Naming: no implied critic network" section says it is: a
// REINFORCE-only network with no critic. pkg/reinforce and its tests
// depend on that shape and are unmodified by this package's existence.
package actorcritic

import (
	"math"
	"math/rand/v2"
	"sync"

	"github.com/reallyoldfogie/cRL-go/pkg/mat"
)

// Layer is one hidden layer's weight matrix and bias vector.
type Layer struct {
	W, B *mat.Matrix
}

// Params holds the learnable weights and biases of the actor-critic MLP:
//
//	                                        -> Wpi,Bpi -> Softmax -> policy output
//	input -> Hidden[0] -> ReLU -> ... -> Hidden[n-1] -> ReLU
//	                                        -> Wv,Bv  -> value output
//
// Hidden is the shared trunk, one or more layers deep; every layer has
// the same width (HiddenSize), matching Net2WiderNet's existing
// assumption that the whole trunk widens together — per-layer widths
// are a non-goal until a real caller needs them (see
// docs/plans/20-configurable-depth-step-memory-and-dynamic-architecture.md,
// Part A). Wpi,Bpi and Wv,Bv are independent linear heads applied to
// the last hidden layer's output.
//
// mu guards concurrent access to the matrices below whenever a live,
// possibly-being-trained Params is read from a different goroutine than
// the one applying gradient updates (see Lock/Unlock and Snapshot,
// and docs/plans/09-concurrency-safe-live-inference.md). Callers that
// only ever touch a Params from a single goroutine, or only after
// training has finished, don't need to think about mu at all.
type Params struct {
	mu sync.RWMutex

	Hidden   []Layer
	Wpi, Bpi *mat.Matrix
	Wv, Bv   *mat.Matrix
}

// NewParams allocates Params for a network with the given layer sizes
// and numHiddenLayers hidden layers (every one HiddenSize wide), using
// the same Xavier/Glorot uniform initialization as pkg/policy.NewParams
// for every weight matrix, including the value head (bound = sqrt(6 /
// (fanIn + fanOut)), fanOut = 1 for Wv). Biases start at zero.
// numHiddenLayers must be at least 1.
func NewParams(rng *rand.Rand, inputSize, hiddenSize, outputSize, numHiddenLayers int) *Params {
	if numHiddenLayers < 1 {
		panic("actorcritic: NewParams: numHiddenLayers must be at least 1")
	}

	valueHeadSize := 1

	hidden := make([]Layer, numHiddenLayers)
	for i := range hidden {
		fanIn := hiddenSize
		if i == 0 {
			fanIn = inputSize
		}
		w := mat.New(hiddenSize, fanIn)
		b := mat.New(hiddenSize, 1)
		w.FillRand(rng, -xavierBound(fanIn, hiddenSize), xavierBound(fanIn, hiddenSize))
		hidden[i] = Layer{W: w, B: b}
	}

	p := &Params{
		Hidden: hidden,
		Wpi:    mat.New(outputSize, hiddenSize),
		Bpi:    mat.New(outputSize, 1),
		Wv:     mat.New(valueHeadSize, hiddenSize),
		Bv:     mat.New(valueHeadSize, 1),
	}

	p.Wpi.FillRand(rng, -xavierBound(hiddenSize, outputSize), xavierBound(hiddenSize, outputSize))
	p.Wv.FillRand(rng, -xavierBound(hiddenSize, valueHeadSize), xavierBound(hiddenSize, valueHeadSize))

	return p
}

func xavierBound(fanIn, fanOut int) float32 {
	return float32(math.Sqrt(6.0 / float64(fanIn+fanOut)))
}

// InputSize, HiddenSize, and OutputSize report the network's layer
// sizes. OutputSize is the policy head's width (the action space size);
// the value head is always width 1 regardless of OutputSize. HiddenSize
// is every hidden layer's width (see Params' doc comment); use
// NumHiddenLayers for how many of them there are.
func (p *Params) InputSize() int       { return p.Hidden[0].W.Cols }
func (p *Params) HiddenSize() int      { return p.Hidden[0].W.Rows }
func (p *Params) OutputSize() int      { return p.Wpi.Rows }
func (p *Params) NumHiddenLayers() int { return len(p.Hidden) }

// Lock acquires p's write lock. Callers applying a gradient update to a
// Params that other goroutines may concurrently Snapshot (e.g. a live
// inference Actor sharing this Params with a trainer) must hold this
// lock for the duration of that update (see pkg/ppo.Trainer.RunEpoch),
// so Snapshot never observes a partially-updated matrix. Callers that
// never share a Params across goroutines don't need to call
// Lock/Unlock at all.
func (p *Params) Lock() { p.mu.Lock() }

// Unlock releases the write lock acquired by Lock.
func (p *Params) Unlock() { p.mu.Unlock() }

// Snapshot returns a deep copy of p's current weights: every matrix's
// data is copied into freshly allocated storage, so the result shares
// nothing with p and is unaffected by any gradient update applied to p
// afterward. The read lock is held only long enough to perform the
// copy, not for the snapshot's entire lifetime, so a caller holding a
// snapshot never blocks a concurrent writer (see Lock) — this is what
// lets Actor (see actor.go) infer against a stable snapshot instead of
// serializing every decision behind p's lock.
func (p *Params) Snapshot() *Params {
	p.mu.RLock()
	defer p.mu.RUnlock()

	hidden := make([]Layer, len(p.Hidden))
	for i, layer := range p.Hidden {
		hidden[i] = Layer{W: cloneMatrix(layer.W), B: cloneMatrix(layer.B)}
	}

	return &Params{
		Hidden: hidden,
		Wpi:    cloneMatrix(p.Wpi),
		Bpi:    cloneMatrix(p.Bpi),
		Wv:     cloneMatrix(p.Wv),
		Bv:     cloneMatrix(p.Bv),
	}
}

// cloneMatrix returns a deep copy of m, sharing no storage with it.
// Duplicated from pkg/policy's identically-named helper rather than
// shared, for the same reason this package's chain type is duplicated
// (see network.go): no dependency on pkg/policy's unexported internals.
func cloneMatrix(m *mat.Matrix) *mat.Matrix {
	data := make([]float32, len(m.Data))
	copy(data, m.Data)
	return &mat.Matrix{Rows: m.Rows, Cols: m.Cols, Data: data}
}
