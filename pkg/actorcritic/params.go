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

// Params holds the learnable weights and biases of the actor-critic MLP:
//
//	                          -> Wpi,Bpi -> Softmax -> policy output
//	input -> W0,B0 -> ReLU -> W1,B1 -> ReLU
//	                          -> Wv,Bv  -> value output
//
// W0,B0/W1,B1 are the shared trunk (identical in shape and role to
// pkg/policy.Params' W0,B0/W1,B1); Wpi,Bpi and Wv,Bv are independent
// linear heads applied to the same trunk output.
//
// mu guards concurrent access to the matrices below whenever a live,
// possibly-being-trained Params is read from a different goroutine than
// the one applying gradient updates (see Lock/Unlock and Snapshot,
// and docs/plans/09-concurrency-safe-live-inference.md). Callers that
// only ever touch a Params from a single goroutine, or only after
// training has finished, don't need to think about mu at all.
type Params struct {
	mu sync.RWMutex

	W0, B0   *mat.Matrix
	W1, B1   *mat.Matrix
	Wpi, Bpi *mat.Matrix
	Wv, Bv   *mat.Matrix
}

// NewParams allocates Params for a network with the given layer sizes,
// using the same Xavier/Glorot uniform initialization as
// pkg/policy.NewParams for every weight matrix, including the value
// head (bound = sqrt(6 / (fanIn + fanOut)), fanOut = 1 for Wv). Biases
// start at zero.
func NewParams(rng *rand.Rand, inputSize, hiddenSize, outputSize int) *Params {
	valueHeadSize := 1

	p := &Params{
		W0:  mat.New(hiddenSize, inputSize),
		B0:  mat.New(hiddenSize, 1),
		W1:  mat.New(hiddenSize, hiddenSize),
		B1:  mat.New(hiddenSize, 1),
		Wpi: mat.New(outputSize, hiddenSize),
		Bpi: mat.New(outputSize, 1),
		Wv:  mat.New(valueHeadSize, hiddenSize),
		Bv:  mat.New(valueHeadSize, 1),
	}

	p.W0.FillRand(rng, -xavierBound(inputSize, hiddenSize), xavierBound(inputSize, hiddenSize))
	p.W1.FillRand(rng, -xavierBound(hiddenSize, hiddenSize), xavierBound(hiddenSize, hiddenSize))
	p.Wpi.FillRand(rng, -xavierBound(hiddenSize, outputSize), xavierBound(hiddenSize, outputSize))
	p.Wv.FillRand(rng, -xavierBound(hiddenSize, valueHeadSize), xavierBound(hiddenSize, valueHeadSize))

	return p
}

func xavierBound(fanIn, fanOut int) float32 {
	return float32(math.Sqrt(6.0 / float64(fanIn+fanOut)))
}

// InputSize, HiddenSize, and OutputSize report the network's layer
// sizes. OutputSize is the policy head's width (the action space size);
// the value head is always width 1 regardless of OutputSize.
func (p *Params) InputSize() int  { return p.W0.Cols }
func (p *Params) HiddenSize() int { return p.W0.Rows }
func (p *Params) OutputSize() int { return p.Wpi.Rows }

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

	return &Params{
		W0:  cloneMatrix(p.W0),
		B0:  cloneMatrix(p.B0),
		W1:  cloneMatrix(p.W1),
		B1:  cloneMatrix(p.B1),
		Wpi: cloneMatrix(p.Wpi),
		Bpi: cloneMatrix(p.Bpi),
		Wv:  cloneMatrix(p.Wv),
		Bv:  cloneMatrix(p.Bv),
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
