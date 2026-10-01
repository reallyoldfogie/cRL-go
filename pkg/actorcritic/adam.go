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
	// maxGradNorm, when > 0, is the max L2 norm Step rescales the whole
	// averaged-gradient vector (across every parameter together, not
	// per-parameter) down to before applying it. 0 disables clipping,
	// matching every existing NewAdam caller's prior behavior exactly.
	maxGradNorm float32
	parameters  []*autograd.Var
	moment1     []*mat.Matrix // first-moment (mean) estimate, one per parameter
	moment2     []*mat.Matrix // second-moment (uncentered variance) estimate, one per parameter
	stepCount   int
}

// NewAdam builds an Adam optimizer for parameters (e.g.
// actor.Parameters()), with moment estimates initialized to zero
// (matching the reference algorithm's initialization), learningRate as
// the fixed base step size, and no gradient-norm clipping — see
// NewAdamWithGradClip for that.
func NewAdam(parameters []*autograd.Var, learningRate float32) *Adam {
	return NewAdamWithGradClip(parameters, learningRate, 0)
}

// NewAdamWithGradClip builds an Adam optimizer exactly like NewAdam, but
// Step additionally rescales the whole averaged-gradient vector (its L2
// norm across every parameter together) down to at most maxGradNorm
// before applying it, whenever maxGradNorm > 0 - the standard PPO
// stability guard (OpenAI Baselines, Stable-Baselines3 and CleanRL all
// default max_grad_norm to 0.5) this package was missing. Adam's own
// per-parameter adaptive scaling bounds an individual parameter's
// *step size* to roughly learningRate once its moment estimates have
// settled, but nothing bounded the raw gradient *magnitude* feeding
// into that: pkg/ppo's clipped-surrogate ratio only limits how far the
// policy is allowed to move once probabilities have already shifted,
// and normalizeAdvantages only rescales advantages per-batch, so a
// batch whose minibatches happen to disagree sharply (high per-step
// variance even after that normalization) can still sum to an outsized
// gradient — and moment2's slow-moving (beta2=0.999) EMA under-reacts to
// it for several Step calls, during which the oversized update applies
// at close to its raw, un-adapted scale. Found live: PPO rounds with
// properly normalized advantages and correctly clipped ratios still
// collapsed from a healthy ~30 return to catastrophically negative
// within a handful of late-training epochs — independent of
// entropy_coef (reproduced identically at 0.01, 0.02 and 0.05) —
// episode lengths collapsing at the same moment as the broken policy
// started failing fast.
func NewAdamWithGradClip(parameters []*autograd.Var, learningRate, maxGradNorm float32) *Adam {
	moment1 := make([]*mat.Matrix, len(parameters))
	moment2 := make([]*mat.Matrix, len(parameters))
	for i, p := range parameters {
		moment1[i] = mat.New(p.Val.Rows, p.Val.Cols)
		moment2[i] = mat.New(p.Val.Rows, p.Val.Cols)
	}

	return &Adam{
		learningRate: learningRate,
		maxGradNorm:  maxGradNorm,
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

	gradScale := 1 / float32(sampleCount)
	if a.maxGradNorm > 0 {
		gradScale *= a.clipScale(gradScale)
	}

	for i, p := range a.parameters {
		moment1 := a.moment1[i]
		moment2 := a.moment2[i]

		for j := range p.Val.Data {
			gradient := p.Grad.Data[j] * gradScale

			moment1.Data[j] = beta1*moment1.Data[j] + (1-beta1)*gradient
			moment2.Data[j] = beta2*moment2.Data[j] + (1-beta2)*gradient*gradient

			moment1Hat := moment1.Data[j] / beta1Correction
			moment2Hat := moment2.Data[j] / beta2Correction

			p.Val.Data[j] -= a.learningRate * moment1Hat / (float32(math.Sqrt(float64(moment2Hat))) + epsilon)
		}
	}
}

// clipScale returns the additional factor (<=1; 1 if already within
// bounds) Step must multiply preScale by so the L2 norm of the
// resulting gradient, summed across every parameter together, is at
// most a.maxGradNorm. preScale is whatever Step has already applied
// (sampleCount's averaging) before this call, so the norm this computes
// matches the gradient Step is actually about to apply.
func (a *Adam) clipScale(preScale float32) float32 {
	var sumSquares float64
	for _, p := range a.parameters {
		for _, g := range p.Grad.Data {
			scaled := float64(g * preScale)
			sumSquares += scaled * scaled
		}
	}
	norm := float32(math.Sqrt(sumSquares))
	if norm <= a.maxGradNorm || norm == 0 {
		return 1
	}
	return a.maxGradNorm / norm
}
