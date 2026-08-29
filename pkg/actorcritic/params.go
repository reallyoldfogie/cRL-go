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
type Params struct {
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
