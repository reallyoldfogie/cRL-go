package actorcritic

import (
	"math"

	"github.com/reallyoldfogie/cRL-go/pkg/autograd"
	"github.com/reallyoldfogie/cRL-go/pkg/mat"
)

// AdamCoefficient is a fixed Adam hyperparameter (a moment-decay rate or
// numerical-stability epsilon), given its own type so these constants
// can't be mixed up with an ordinary float32 (e.g. a learning rate) at a
// call site.
type AdamCoefficient float32

const (
	// adamBeta1 is the exponential decay rate for the first moment
	// (mean) estimate, matching the original Adam paper's (Kingma &
	// Ba, 2014) default.
	adamBeta1 AdamCoefficient = 0.9
	// adamBeta2 is the exponential decay rate for the second moment
	// (uncentered variance) estimate.
	adamBeta2 AdamCoefficient = 0.999
	// adamEpsilon prevents division by (near) zero when a parameter's
	// second-moment estimate is still very small.
	adamEpsilon AdamCoefficient = 1e-8
)

// Adam implements the Adam optimizer over a fixed set of parameters
// (typically TrainingNetwork.Parameters()): per-parameter first- and
// second-moment estimates, bias-corrected against the number of Step
// calls so far, applied as one update per Step call. Unlike
// TrainingNetwork.ApplyGradientStep's plain SGD, Adam adapts each
// parameter's effective step size individually from its own gradient
// history, which is what lets pkg/ppo's trainer reuse the same
// (potentially small) rollout batch for several minibatch passes
// without the clipped-surrogate objective's benefit being swamped by a
// too-aggressive fixed step size.
type Adam struct {
	learningRate float32
	parameters   []*autograd.Var
	moment1      []*mat.Matrix // first-moment (mean) estimate, one per parameter
	moment2      []*mat.Matrix // second-moment (uncentered variance) estimate, one per parameter
	stepCount    int
}

// NewAdam builds an Adam optimizer for parameters (e.g.
// actor.Parameters()), with moment estimates initialized to zero
// (matching the reference algorithm's initialization) and
// learningRate as the fixed base step size.
func NewAdam(parameters []*autograd.Var, learningRate float32) *Adam {
	moment1 := make([]*mat.Matrix, len(parameters))
	moment2 := make([]*mat.Matrix, len(parameters))
	for i, p := range parameters {
		moment1[i] = mat.New(p.Val.Rows, p.Val.Cols)
		moment2[i] = mat.New(p.Val.Rows, p.Val.Cols)
	}

	return &Adam{
		learningRate: learningRate,
		parameters:   parameters,
		moment1:      moment1,
		moment2:      moment2,
	}
}

// Step performs one Adam update using each parameter's currently
// accumulated gradient (see autograd.Graph.Backward), averaged over
// sampleCount (the number of individual steps whose gradients were
// accumulated), mirroring ApplyGradientStep's sampleCount convention.
// If sampleCount is 0, no update is applied and the step counter (used
// for bias correction) does not advance.
func (a *Adam) Step(sampleCount int) {
	if sampleCount == 0 {
		return
	}
	a.stepCount++

	beta1 := float32(adamBeta1)
	beta2 := float32(adamBeta2)
	epsilon := float32(adamEpsilon)

	beta1Correction := 1 - float32(math.Pow(float64(beta1), float64(a.stepCount)))
	beta2Correction := 1 - float32(math.Pow(float64(beta2), float64(a.stepCount)))

	for i, p := range a.parameters {
		moment1 := a.moment1[i]
		moment2 := a.moment2[i]

		for j := range p.Val.Data {
			gradient := p.Grad.Data[j] / float32(sampleCount)

			moment1.Data[j] = beta1*moment1.Data[j] + (1-beta1)*gradient
			moment2.Data[j] = beta2*moment2.Data[j] + (1-beta2)*gradient*gradient

			moment1Hat := moment1.Data[j] / beta1Correction
			moment2Hat := moment2.Data[j] / beta2Correction

			p.Val.Data[j] -= a.learningRate * moment1Hat / (float32(math.Sqrt(float64(moment2Hat))) + epsilon)
		}
	}
}
