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

	"github.com/reallyoldfogie/cRL-go/pkg/mat"
)

// Params holds the learnable weights and biases of the 3-layer MLP:
//
//	input -> W0,B0 -> ReLU -> W1,B1 -> ReLU -> W2,B2 -> Softmax -> output
type Params struct {
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
