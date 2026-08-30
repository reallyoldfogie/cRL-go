// Package policy builds the 3-layer MLP policy network (and its REINFORCE
// cost graph) used to choose actions in snakeenv.Env, mirroring
// create_actor_model from model.c in github.com/harshbhatt7585/cRL. See
// docs/01-neural-networks-and-forward-pass.md for what a "policy network"
// is and why it's shaped the way it is.
//
// Params (the learnable weights/biases) is kept separate from the graph
// that computes with them, so that a single set of Params can back many
// independent, concurrently-usable InferenceNetworks during rollout
// collection while a single TrainingNetwork accumulates gradients into
// the same underlying matrices. See docs/05-porting-notes.md for the full
// rationale.
package policy

import (
	"math"
	"math/rand/v2"
	"sync"

	"github.com/reallyoldfogie/cRL-go/pkg/mat"
)

// Params holds the learnable weights and biases of the 3-layer MLP:
//
//	input -> W0,B0 -> ReLU -> W1,B1 -> ReLU -> W2,B2 -> Softmax -> output
//
// mu guards concurrent access to the matrices below whenever a live,
// possibly-being-trained Params is read from a different goroutine than
// the one applying gradient updates (see Lock/Unlock and Snapshot,
// and docs/plans/09-concurrency-safe-live-inference.md). Callers that
// only ever touch a Params from a single goroutine, or only after
// training has finished, don't need to think about mu at all.
type Params struct {
	mu sync.RWMutex

	W0, B0 *mat.Matrix
	W1, B1 *mat.Matrix
	W2, B2 *mat.Matrix
}

// NewParams allocates Params for a network with the given layer sizes and
// initializes the weight matrices using Glorot/Xavier uniform
// initialization: each weight is drawn uniformly from
// [-bound, bound] where bound = sqrt(6 / (fanIn + fanOut)).
// See docs/01-neural-networks-and-forward-pass.md for why this
// initialization scheme (rather than, say, all-zeros or unscaled random
// values) is used. Biases start at zero.
func NewParams(rng *rand.Rand, inputSize, hiddenSize, outputSize int) *Params {
	p := &Params{
		W0: mat.New(hiddenSize, inputSize),
		B0: mat.New(hiddenSize, 1),
		W1: mat.New(hiddenSize, hiddenSize),
		B1: mat.New(hiddenSize, 1),
		W2: mat.New(outputSize, hiddenSize),
		B2: mat.New(outputSize, 1),
	}

	p.W0.FillRand(rng, -xavierBound(inputSize, hiddenSize), xavierBound(inputSize, hiddenSize))
	p.W1.FillRand(rng, -xavierBound(hiddenSize, hiddenSize), xavierBound(hiddenSize, hiddenSize))
	p.W2.FillRand(rng, -xavierBound(hiddenSize, outputSize), xavierBound(hiddenSize, outputSize))

	return p
}

func xavierBound(fanIn, fanOut int) float32 {
	return float32(math.Sqrt(6.0 / float64(fanIn+fanOut)))
}

// InputSize, HiddenSize, and OutputSize report the network's layer sizes.
func (p *Params) InputSize() int  { return p.W0.Cols }
func (p *Params) HiddenSize() int { return p.W0.Rows }
func (p *Params) OutputSize() int { return p.W2.Rows }

// Lock acquires p's write lock. Callers applying a gradient update to a
// Params that other goroutines may concurrently Snapshot (e.g. a live
// inference Actor sharing this Params with a trainer) must hold this
// lock for the duration of that update (see
// pkg/reinforce.Trainer.RunEpoch), so Snapshot never observes a
// partially-updated matrix. Callers that never share a Params across
// goroutines don't need to call Lock/Unlock at all.
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

	return &Params{
		W0: cloneMatrix(p.W0),
		B0: cloneMatrix(p.B0),
		W1: cloneMatrix(p.W1),
		B1: cloneMatrix(p.B1),
		W2: cloneMatrix(p.W2),
		B2: cloneMatrix(p.B2),
	}
}

// cloneMatrix returns a deep copy of m, sharing no storage with it.
func cloneMatrix(m *mat.Matrix) *mat.Matrix {
	data := make([]float32, len(m.Data))
	copy(data, m.Data)
	return &mat.Matrix{Rows: m.Rows, Cols: m.Cols, Data: data}
}
