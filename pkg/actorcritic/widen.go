package actorcritic

import (
	"fmt"
	"math/rand/v2"

	"github.com/reallyoldfogie/cRL-go/pkg/mat"
)

// Net2WiderNet returns a new Params whose hidden width is newHiddenSize
// (both W0/B0's output and W1/B1's output — see NewParams' diagram; this
// network always keeps the two hidden layers the same width, so widening
// applies to both), computing the same function p does on any input, up
// to noiseStd of independent Gaussian noise added to each newly-created
// unit's weights (0 for exact preservation). p itself is unmodified.
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

	oldHiddenSize := p.W0.Rows
	if newHiddenSize <= oldHiddenSize {
		return nil, fmt.Errorf("actorcritic: Net2WiderNet: newHiddenSize %d must be greater than the current hidden size %d", newHiddenSize, oldHiddenSize)
	}

	mapTo, count := widenMapping(rng, oldHiddenSize, newHiddenSize)

	w0 := widenProducerAxis(p.W0, mapTo, noiseStd, rng)
	b0 := widenProducerAxis(p.B0, mapTo, noiseStd, rng)

	// W1 sits at both boundaries this widens: its columns are the first
	// boundary's consumer side (fed by W0/B0's now-wider output), and its
	// rows are the second boundary's producer side (feeding Wpi/Wv). Each
	// axis is widened independently — order between them doesn't matter.
	w1 := widenProducerAxis(widenConsumerAxis(p.W1, mapTo, count), mapTo, noiseStd, rng)
	b1 := widenProducerAxis(p.B1, mapTo, noiseStd, rng)

	wpi := widenConsumerAxis(p.Wpi, mapTo, count)
	wv := widenConsumerAxis(p.Wv, mapTo, count)

	return &Params{
		W0: w0, B0: b0,
		W1: w1, B1: b1,
		Wpi: wpi, Bpi: cloneMatrix(p.Bpi),
		Wv: wv, Bv: cloneMatrix(p.Bv),
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
