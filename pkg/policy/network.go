package policy

import (
	"github.com/reallyoldfogie/cRL-go/pkg/autograd"
	"github.com/reallyoldfogie/cRL-go/pkg/mat"
)

// paramVars wraps a Params' six matrices as autograd.Var leaves.
type paramVars struct {
	W0, B0 *autograd.Var
	W1, B1 *autograd.Var
	W2, B2 *autograd.Var
}

func (p paramVars) all() []*autograd.Var {
	return []*autograd.Var{p.W0, p.B0, p.W1, p.B1, p.W2, p.B2}
}

// chain builds a sequence of autograd ops, short-circuiting on the first
// error so callers don't need to check an error after every intermediate
// step (mirroring the "sticky error" pattern used by e.g. bufio.Writer).
type chain struct{ err error }

func (c *chain) matMul(a, b *autograd.Var) *autograd.Var {
	if c.err != nil {
		return nil
	}
	v, err := autograd.MatMul(a, b)
	c.err = err
	return v
}

func (c *chain) add(a, b *autograd.Var) *autograd.Var {
	if c.err != nil {
		return nil
	}
	v, err := autograd.Add(a, b)
	c.err = err
	return v
}

func (c *chain) relu(a *autograd.Var) *autograd.Var {
	if c.err != nil {
		return nil
	}
	v, err := autograd.ReLU(a)
	c.err = err
	return v
}

func (c *chain) softmax(a *autograd.Var) *autograd.Var {
	if c.err != nil {
		return nil
	}
	v, err := autograd.Softmax(a)
	c.err = err
	return v
}

// buildForward wires up input -> W0,B0 -> ReLU -> W1,B1 -> ReLU -> W2,B2
// -> Softmax -> output, returning the output Var.
func buildForward(input *autograd.Var, p paramVars) (*autograd.Var, error) {
	c := &chain{}

	z0 := c.matMul(p.W0, input)
	z0b := c.add(z0, p.B0)
	a0 := c.relu(z0b)

	z1 := c.matMul(p.W1, a0)
	z1b := c.add(z1, p.B1)
	a1 := c.relu(z1b)

	z2 := c.matMul(p.W2, a1)
	z2b := c.add(z2, p.B2)

	output := c.softmax(z2b)
	return output, c.err
}

// InferenceNetwork is a forward-only computation graph over a shared,
// read-only Params: its weight/bias Vars alias Params' matrices directly
// (see autograd.Constant) rather than copying them, and no gradients are
// tracked. Building one is cheap (a handful of small allocations for
// activation buffers), so a fresh InferenceNetwork can be built per
// rollout-collection goroutine per epoch, letting many goroutines run
// forward passes concurrently against the same frozen weights while a
// separate TrainingNetwork later updates those weights sequentially.
type InferenceNetwork struct {
	Input  *autograd.Var
	Output *autograd.Var
	Graph  *autograd.Graph
}

// NewInferenceNetwork builds an InferenceNetwork over params.
func NewInferenceNetwork(params *Params) (*InferenceNetwork, error) {
	input := autograd.NewVar(params.InputSize(), 1, autograd.FlagNone)

	pv := paramVars{
		W0: autograd.Constant(params.W0),
		B0: autograd.Constant(params.B0),
		W1: autograd.Constant(params.W1),
		B1: autograd.Constant(params.B1),
		W2: autograd.Constant(params.W2),
		B2: autograd.Constant(params.B2),
	}

	output, err := buildForward(input, pv)
	if err != nil {
		return nil, err
	}

	return &InferenceNetwork{
		Input:  input,
		Output: output,
		Graph:  autograd.BuildGraph(output),
	}, nil
}

// TrainingNetwork is the forward-pass-plus-REINFORCE-loss computation
// graph used to compute gradients: its weight/bias Vars alias Params'
// matrices (see autograd.Parameter) and accumulate gradients into them
// across repeated Graph.Backward calls, matching model_state's cost_graph
// in the original C code. Loss also includes an entropy bonus (see
// buildEntropyBonus) working against premature collapse into a
// deterministic distribution, the same role it plays in pkg/ppo's loss
// (see docs/08-ppo-clipped-objective.md) — REINFORCE has no clipping to
// slow a single large step, so a policy that saturates its softmax has
// nothing else here to reintroduce exploration.
type TrainingNetwork struct {
	Input     *autograd.Var
	Output    *autograd.Var
	Advantage *autograd.Var
	Loss      *autograd.Var
	Graph     *autograd.Graph // rooted at Loss; a Forward call also computes Output.

	params paramVars
}

// NewTrainingNetwork builds a TrainingNetwork over params. entropyCoef
// scales the entropy bonus subtracted from the REINFORCE loss (see
// buildEntropyBonus); pass 0 to disable it and match this package's
// pre-entropy-bonus loss exactly (a coefficient of 0 makes every
// entropy-bonus term and its gradient exactly zero, so the composed
// graph is mathematically identical to the plain REINFORCE loss, not
// merely close to it).
func NewTrainingNetwork(params *Params, entropyCoef float32) (*TrainingNetwork, error) {
	input := autograd.NewVar(params.InputSize(), 1, autograd.FlagNone)

	pv := paramVars{
		W0: autograd.Parameter(params.W0),
		B0: autograd.Parameter(params.B0),
		W1: autograd.Parameter(params.W1),
		B1: autograd.Parameter(params.B1),
		W2: autograd.Parameter(params.W2),
		B2: autograd.Parameter(params.B2),
	}

	output, err := buildForward(input, pv)
	if err != nil {
		return nil, err
	}

	advantage := autograd.NewVar(params.OutputSize(), 1, autograd.FlagNone)
	reinforceLoss, err := autograd.ReinforceLoss(output, advantage)
	if err != nil {
		return nil, err
	}

	entropyBonus, err := buildEntropyBonus(output, entropyCoef)
	if err != nil {
		return nil, err
	}
	negatedEntropyBonus, err := autograd.Neg(entropyBonus)
	if err != nil {
		return nil, err
	}
	loss, err := autograd.Add(reinforceLoss, negatedEntropyBonus)
	if err != nil {
		return nil, err
	}

	return &TrainingNetwork{
		Input:     input,
		Output:    output,
		Advantage: advantage,
		Loss:      loss,
		Graph:     autograd.BuildGraph(loss),
		params:    pv,
	}, nil
}

// buildEntropyBonus computes entropyCoef * sum_a(-probs[a]*log(probs[a])),
// i.e. entropyCoef times the policy's Shannon entropy, expressed as a
// per-action (outputSize, 1) vector whose entries sum (via Backward's
// implicit-sum trick — see autograd.Graph's doc comment) to that scalar.
// Mirrors pkg/ppo/loss.go's identically-named function; unlike that one,
// this recomputes log(probs) itself rather than accepting a caller's
// copy, since NewTrainingNetwork has no other reason to compute it.
func buildEntropyBonus(probs *autograd.Var, entropyCoef float32) (*autograd.Var, error) {
	logProbs, err := autograd.Log(probs)
	if err != nil {
		return nil, err
	}
	perActionEntropy, err := autograd.Mul(probs, logProbs)
	if err != nil {
		return nil, err
	}
	positiveEntropy, err := autograd.Neg(perActionEntropy)
	if err != nil {
		return nil, err
	}

	entropyCoefVar := autograd.Constant(filledColumn(probs.Val.Rows, entropyCoef))
	return autograd.Mul(positiveEntropy, entropyCoefVar)
}

// filledColumn returns a (rows, 1) matrix with every entry set to value.
func filledColumn(rows int, value float32) *mat.Matrix {
	m := mat.New(rows, 1)
	m.Fill(value)
	return m
}

// ZeroGrad clears every parameter's accumulated gradient. Call this once
// per training batch, after applying the previous batch's gradient step.
func (n *TrainingNetwork) ZeroGrad() {
	for _, p := range n.params.all() {
		p.Grad.Clear()
	}
}

// ApplyGradientStep performs one manual (vanilla) SGD step:
// param -= learningRate/sampleCount * accumulatedGradient, for every
// parameter. sampleCount is normally the number of individual (state,
// action) steps whose gradients were accumulated into this batch, so the
// effective step is scaled by the *average* gradient per sample rather
// than the raw sum. If sampleCount is 0, no update is applied.
func (n *TrainingNetwork) ApplyGradientStep(learningRate float32, sampleCount int) {
	if sampleCount == 0 {
		return
	}

	scale := learningRate / float32(sampleCount)
	for _, p := range n.params.all() {
		p.Grad.Scale(scale)
		_ = p.Val.Sub(p.Val, p.Grad)
	}
}
