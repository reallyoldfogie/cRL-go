package actorcritic

import (
	"fmt"
	"math/rand/v2"

	"github.com/reallyoldfogie/cRL-go/pkg/mat"
)

// Net2WiderNet returns a new Params whose hidden width is newHiddenSize
// (every hidden layer's output — see NewParams' diagram; this network
// always keeps every hidden layer the same width, so widening applies
// to all of them together), computing the same function p does on any
// input, up to noiseStd of independent Gaussian noise added to each
// newly-created unit's weights (0 for exact preservation). p itself is
// unmodified.
//
// This is Net2Net's "widening" transform (Chen, Goodfellow, Shlens 2016,
// "Net2Net: Accelerating Learning via Knowledge Transfer",
// https://arxiv.org/abs/1511.05641): rather than reinitializing a bigger
// network from scratch and losing everything a checkpoint has already
// learned, every existing unit is kept exactly where it was (at its
// original index) and newHiddenSize-p.HiddenSize() new units are added,
// each an independent copy of a uniformly-chosen existing unit. Whichever
// layer CONSUMES a duplicated unit's output has that unit's incoming
// weight column split across every position now representing it — the
// original index plus however many new units happened to clone it — so
// their combined contribution to the next layer's pre-activation is
// unchanged. Two boundaries are widened this way: W0/B0 (producer) into
// W1 (consumer) on its input side, and W1/B1 (producer, its output side)
// into Wpi and Wv (consumers).
//
// A duplicated unit computes an identical function to its clone-parent
// until something breaks the tie between them, since they'd otherwise
// receive identical gradients forever and the added capacity would go
// unused — that's what noiseStd is for: a small perturbation (relative to
// typical Xavier-scale weight magnitudes) to each new unit's own weight
// row, not to what already existed. This makes the transform only
// approximately, not exactly, function-preserving when noiseStd > 0 — the
// tradeoff Chen et al. accept for the same reason.
//
// Exists because a checkpoint's hidden size is otherwise fixed for its
// whole life: Load reconstructs whatever width the file itself stored
// (see checkpoint.go), ignoring anything a caller might otherwise want.
// Built as general infrastructure ahead of any specific run needing it —
// see mc-rsi-trainer's cmd/widen-checkpoint for the tool that applies
// this to a saved lineage generation.
//
// newHiddenSize must be strictly greater than p.HiddenSize(); use
// Snapshot to copy a Params without changing its size.
func Net2WiderNet(p *Params, newHiddenSize int, noiseStd float32, rng *rand.Rand) (*Params, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	oldHiddenSize := p.HiddenSize()
	if newHiddenSize <= oldHiddenSize {
		return nil, fmt.Errorf("actorcritic: Net2WiderNet: newHiddenSize %d must be greater than the current hidden size %d", newHiddenSize, oldHiddenSize)
	}

	mapTo, count := widenMapping(rng, oldHiddenSize, newHiddenSize)

	// Every hidden layer sits at a producer boundary (its own output
	// feeds the next layer or a head, so its rows/bias always widen) and
	// every layer but the first also sits at a consumer boundary (its
	// input is the previous layer's now-wider output, so its columns
	// widen too — layer 0's input is the actual observation, untouched
	// by this). Each axis is widened independently — order between them
	// doesn't matter, mirroring W1's original two-boundary handling.
	hidden := make([]Layer, len(p.Hidden))
	for i, layer := range p.Hidden {
		w := layer.W
		if i > 0 {
			w = widenConsumerAxis(w, mapTo, count)
		}
		w = widenProducerAxis(w, mapTo, noiseStd, rng)
		b := widenProducerAxis(layer.B, mapTo, noiseStd, rng)
		hidden[i] = Layer{W: w, B: b}
	}

	wpi := widenConsumerAxis(p.Wpi, mapTo, count)
	wv := widenConsumerAxis(p.Wv, mapTo, count)

	return &Params{
		Hidden: hidden,
		Wpi:    wpi, Bpi: cloneMatrix(p.Bpi),
		Wv: wv, Bv: cloneMatrix(p.Bv),
	}, nil
}

// Net2DeeperNet returns a new Params with one extra hidden layer
// inserted at atDepth — the number of existing hidden layers that come
// before it, so atDepth must be between 1 and p.NumHiddenLayers()
// inclusive (atDepth == p.NumHiddenLayers() appends the new layer right
// before the heads) — computing exactly the same function p does on
// any input: the inserted layer's weight is the HiddenSize x HiddenSize
// identity matrix and its bias is zero, so for the preceding layer's
// ReLU output a (which is >= 0 elementwise), ReLU(I·a + 0) = a exactly.
// p itself is unmodified.
//
// This is Net2Net's "deepening" transform (same 2016 paper as
// Net2WiderNet's own doc comment cites). Unlike widening, it needs no
// noise parameter to break symmetry: an inserted identity layer
// computes a function no other layer in the network computes, so
// there's no tied-gradient problem the way two widened units sharing
// one clone-parent have.
//
// atDepth == 0 is rejected rather than treated as "insert before
// Hidden[0]": that boundary's input is the raw observation, not a ReLU
// output, and ReLU(I·a) == a only holds for a >= 0 — a guarantee only
// an existing ReLU provides. Every other insertion point sits strictly
// between two things that already satisfy that (two hidden layers, or
// the last hidden layer and the heads), which is why they all work the
// same way.
func Net2DeeperNet(p *Params, atDepth int) (*Params, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if atDepth < 1 || atDepth > len(p.Hidden) {
		return nil, fmt.Errorf("actorcritic: Net2DeeperNet: atDepth %d must be between 1 and %d (the current number of hidden layers)", atDepth, len(p.Hidden))
	}

	hiddenSize := p.HiddenSize()
	identityW := mat.New(hiddenSize, hiddenSize)
	for i := 0; i < hiddenSize; i++ {
		identityW.Data[i*hiddenSize+i] = 1
	}
	identity := Layer{W: identityW, B: mat.New(hiddenSize, 1)}

	hidden := make([]Layer, 0, len(p.Hidden)+1)
	for i, layer := range p.Hidden {
		if i == atDepth {
			hidden = append(hidden, identity)
		}
		hidden = append(hidden, Layer{W: cloneMatrix(layer.W), B: cloneMatrix(layer.B)})
	}
	if atDepth == len(p.Hidden) {
		hidden = append(hidden, identity)
	}

	return &Params{
		Hidden: hidden,
		Wpi:    cloneMatrix(p.Wpi),
		Bpi:    cloneMatrix(p.Bpi),
		Wv:     cloneMatrix(p.Wv),
		Bv:     cloneMatrix(p.Bv),
	}, nil
}

// widenMapping returns, for each of newSize final positions, which of the
// original [0, size) indices it represents — identity for the first size
// positions (every existing unit keeps its own index unchanged), then a
// uniform random draw from [0, size) for each new position beyond that,
// matching Net2Net's own widening scheme. count[j] is how many total
// positions (the original index itself, plus however many new positions
// happened to draw it) ended up representing original unit j; a consumer
// of a widened layer splits that unit's original weight count[j] ways
// (see widenConsumerAxis) so the two are always used together.
func widenMapping(rng *rand.Rand, size, newSize int) (mapTo []int, count []int) {
	mapTo = make([]int, newSize)
	count = make([]int, size)
	for j := 0; j < size; j++ {
		mapTo[j] = j
		count[j] = 1
	}
	for k := size; k < newSize; k++ {
		j := rng.IntN(size)
		mapTo[k] = j
		count[j]++
	}
	return mapTo, count
}

// widenProducerAxis returns a new matrix with len(mapTo) rows (m's Cols
// unchanged): row k is a copy of m's row mapTo[k], the unit it represents.
// Every row at or beyond m's original row count is a genuinely new unit —
// its copied weights get noiseStd of independent Gaussian noise added (see
// Net2WiderNet's own doc comment for why); every row below that is an
// existing unit kept exactly as it was, untouched by noise.
func widenProducerAxis(m *mat.Matrix, mapTo []int, noiseStd float32, rng *rand.Rand) *mat.Matrix {
	out := mat.New(len(mapTo), m.Cols)
	for k, j := range mapTo {
		copy(out.Data[k*m.Cols:(k+1)*m.Cols], m.Data[j*m.Cols:(j+1)*m.Cols])
		if k >= m.Rows && noiseStd > 0 {
			row := out.Data[k*m.Cols : (k+1)*m.Cols]
			for c := range row {
				row[c] += float32(rng.NormFloat64()) * noiseStd
			}
		}
	}
	return out
}

// widenConsumerAxis returns a new matrix with len(mapTo) columns (m's Rows
// unchanged): column k is m's column mapTo[k] scaled by 1/count[mapTo[k]].
// Every position representing original input unit j — its own kept index
// plus every new position that cloned it — carries 1/count[j] of that
// unit's original weight, so their combined contribution to this matrix's
// output (a sum over its now-larger input) reproduces unit j's original
// single contribution exactly; see Net2WiderNet's own doc comment.
func widenConsumerAxis(m *mat.Matrix, mapTo []int, count []int) *mat.Matrix {
	out := mat.New(m.Rows, len(mapTo))
	for r := 0; r < m.Rows; r++ {
		for k, j := range mapTo {
			out.Data[r*len(mapTo)+k] = m.Data[r*m.Cols+j] / float32(count[j])
		}
	}
	return out
}
