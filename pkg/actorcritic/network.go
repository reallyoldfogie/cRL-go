package actorcritic

import (
	"github.com/reallyoldfogie/cRL-go/pkg/autograd"
)

// paramVars wraps a Params' eight matrices as autograd.Var leaves.
type paramVars struct {
	W0, B0   *autograd.Var
	W1, B1   *autograd.Var
	Wpi, Bpi *autograd.Var
	Wv, Bv   *autograd.Var
}

func (p paramVars) all() []*autograd.Var {
	return []*autograd.Var{p.W0, p.B0, p.W1, p.B1, p.Wpi, p.Bpi, p.Wv, p.Bv}
}

// chain mirrors pkg/policy/network.go's sticky-error op-chaining helper.
// It's duplicated here, rather than shared, so this package has no
// dependency on pkg/policy's unexported internals — see this package's
// doc comment for why pkg/policy and pkg/actorcritic are kept separate.
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

// buildForward wires up the shared trunk (input -> W0,B0 -> ReLU ->
// W1,B1 -> ReLU) and both heads (Wpi,Bpi -> Softmax for the policy
// output; Wv,Bv for the value output), returning both output Vars.
func buildForward(input *autograd.Var, p paramVars) (policyOutput, valueOutput *autograd.Var, err error) {
	c := &chain{}

	z0 := c.matMul(p.W0, input)
	z0b := c.add(z0, p.B0)
	a0 := c.relu(z0b)

	z1 := c.matMul(p.W1, a0)
	z1b := c.add(z1, p.B1)
	a1 := c.relu(z1b)

	zpi := c.matMul(p.Wpi, a1)
	zpib := c.add(zpi, p.Bpi)
	policyOutput = c.softmax(zpib)

	zv := c.matMul(p.Wv, a1)
	valueOutput = c.add(zv, p.Bv)

	return policyOutput, valueOutput, c.err
}

// InferenceNetwork is a forward-only computation graph over a shared,
// read-only Params (its weight/bias Vars alias Params' matrices via
// autograd.Constant, exactly as pkg/policy.InferenceNetwork does, for
// the same reason: many goroutines can build one InferenceNetwork each
// and run forward passes concurrently against the same frozen weights).
// Its Graph is built with autograd.BuildGraphMulti over both outputs, so
// a single Forward call computes the policy distribution and the value
// estimate together without recomputing the shared trunk twice.
type InferenceNetwork struct {
	Input        *autograd.Var
	PolicyOutput *autograd.Var
	ValueOutput  *autograd.Var
	Graph        *autograd.Graph
}

// NewInferenceNetwork builds an InferenceNetwork over params.
func NewInferenceNetwork(params *Params) (*InferenceNetwork, error) {
	input := autograd.NewVar(params.InputSize(), 1, autograd.FlagNone)

	pv := paramVars{
		W0:  autograd.Constant(params.W0),
		B0:  autograd.Constant(params.B0),
		W1:  autograd.Constant(params.W1),
		B1:  autograd.Constant(params.B1),
		Wpi: autograd.Constant(params.Wpi),
		Bpi: autograd.Constant(params.Bpi),
		Wv:  autograd.Constant(params.Wv),
		Bv:  autograd.Constant(params.Bv),
	}

	policyOutput, valueOutput, err := buildForward(input, pv)
	if err != nil {
		return nil, err
	}

	return &InferenceNetwork{
		Input:        input,
		PolicyOutput: policyOutput,
		ValueOutput:  valueOutput,
		Graph:        autograd.BuildGraphMulti(policyOutput, valueOutput),
	}, nil
}

// TrainingNetwork wraps a Params' eight matrices as gradient-accumulating
// (autograd.Parameter) leaves feeding the same shared trunk built by
// buildForward, exposing both PolicyOutput and ValueOutput so a loss can
// be composed from them.
//
// Unlike pkg/policy.TrainingNetwork, TrainingNetwork does not build a
// Graph or own a Loss: the actual PPO objective (combining
// PolicyOutput, ValueOutput, an advantage signal, an old
// action-log-probability, and a value target) depends on rollout data
// this package doesn't produce, so it's composed on top of this network
// by its trainer instead. ZeroGrad and ApplyGradientStep only touch
// parameter Grad/Val buffers directly, so they work regardless of what
// loss graph a caller builds on top of PolicyOutput/ValueOutput.
type TrainingNetwork struct {
	Input        *autograd.Var
	PolicyOutput *autograd.Var
	ValueOutput  *autograd.Var

	params paramVars
}

// NewTrainingNetwork builds a TrainingNetwork over params.
func NewTrainingNetwork(params *Params) (*TrainingNetwork, error) {
	input := autograd.NewVar(params.InputSize(), 1, autograd.FlagNone)

	pv := paramVars{
		W0:  autograd.Parameter(params.W0),
		B0:  autograd.Parameter(params.B0),
		W1:  autograd.Parameter(params.W1),
		B1:  autograd.Parameter(params.B1),
		Wpi: autograd.Parameter(params.Wpi),
		Bpi: autograd.Parameter(params.Bpi),
		Wv:  autograd.Parameter(params.Wv),
		Bv:  autograd.Parameter(params.Bv),
	}

	policyOutput, valueOutput, err := buildForward(input, pv)
	if err != nil {
		return nil, err
	}

	return &TrainingNetwork{
		Input:        input,
		PolicyOutput: policyOutput,
		ValueOutput:  valueOutput,
		params:       pv,
	}, nil
}

// ZeroGrad clears every parameter's accumulated gradient. Call this once
// per training batch, after applying the previous batch's gradient step.
func (n *TrainingNetwork) ZeroGrad() {
	for _, p := range n.params.all() {
		p.Grad.Clear()
	}
}

// ApplyGradientStep performs one manual (vanilla) SGD step, identical in
// shape to pkg/policy.TrainingNetwork.ApplyGradientStep:
// param -= learningRate/sampleCount * accumulatedGradient, for every
// parameter. If sampleCount is 0, no update is applied.
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
