package ppo

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/autograd"
	"github.com/reallyoldfogie/cRL-go/pkg/mat"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- Gradient checking ---
//
// Duplicated from pkg/autograd/gradcheck_test.go's helpers (test files
// can't be imported across packages), following the same pattern
// pkg/actorcritic/network_test.go already uses.

const (
	gradCheckEpsilon   = 1e-2
	gradCheckTolerance = 5e-2
)

func sumMatrix(m *mat.Matrix) float64 {
	var sum float64
	for _, v := range m.Data {
		sum += float64(v)
	}
	return sum
}

func numericalGradient(x, output *autograd.Var, forward func()) *mat.Matrix {
	grad := mat.New(x.Val.Rows, x.Val.Cols)

	for i := range x.Val.Data {
		original := x.Val.Data[i]

		x.Val.Data[i] = original + gradCheckEpsilon
		forward()
		plus := sumMatrix(output.Val)

		x.Val.Data[i] = original - gradCheckEpsilon
		forward()
		minus := sumMatrix(output.Val)

		x.Val.Data[i] = original
		grad.Data[i] = float32((plus - minus) / (2 * gradCheckEpsilon))
	}

	forward()
	return grad
}

func assertMatricesClose(t *testing.T, want, got *mat.Matrix, tolerance float64) {
	t.Helper()
	require.Equal(t, want.Rows, got.Rows)
	require.Equal(t, want.Cols, got.Cols)

	for i := range want.Data {
		assert.InDelta(t, want.Data[i], got.Data[i], tolerance, "element %d differs", i)
	}
}

// TestGradientCheckPPOLoss gradient-checks the full PPO loss (policy +
// value + entropy) end to end: from the shared input, through both the
// policy and value heads, to the combined loss, verifying every one of
// the network's eight parameter matrices receives a correct gradient.
func TestGradientCheckPPOLoss(t *testing.T) {
	// Seed (1, 4) is used instead of an arbitrary one because it keeps
	// every hidden-layer pre-activation comfortably away from ReLU's
	// kink at zero for this network shape and input; landing exactly on
	// a kink makes gradient-checking that unit unstable (see
	// TestGradientCheckReLU's doc comment in pkg/autograd for the same
	// caveat) without indicating any actual bug.
	rng := rand.New(rand.NewPCG(1, 4))
	actorParams := actorcritic.NewParams(rng, 4, 5, 3)

	net, err := NewTrainingNetwork(actorParams, LossConfig{ClipEpsilon: 0.2, EntropyCoef: 0.01, ValueCoef: 0.5})
	require.NoError(t, err)

	net.Actor.Input.Val.FillRand(rng, -1, 1)
	net.SetStep(rl.Action(1), -1.0, 0.7, 0.5)

	net.Graph.Forward()
	net.Graph.Backward()

	for _, p := range net.Actor.Parameters() {
		assertMatricesClose(t, p.Grad, numericalGradient(p, net.Loss, net.Graph.Forward), gradCheckTolerance)
	}
}

// TestBuildLossClipsRatioOutsideBounds confirms the clip actually clips:
// pushing the probability ratio far outside [1-eps, 1+eps] must leave
// the resulting loss reflecting the *clamped* surrogate, not the raw
// (unclamped) one. entropyCoef/valueCoef are set to 0 here so the
// combined loss reduces to exactly the policy surrogate term at the
// sampled action's index, making the expected value computable by hand.
func TestBuildLossClipsRatioOutsideBounds(t *testing.T) {
	rng := rand.New(rand.NewPCG(419, 421))
	actorParams := actorcritic.NewParams(rng, 4, 5, 3)
	clipEpsilon := float32(0.2)

	net, err := NewTrainingNetwork(actorParams, LossConfig{ClipEpsilon: clipEpsilon, EntropyCoef: 0, ValueCoef: 0})
	require.NoError(t, err)

	net.Actor.Input.Val.FillRand(rng, -1, 1)
	action := rl.Action(1)
	advantage := float32(1.0)

	// First, forward once to read off the network's actual current
	// log-probability for action, so oldLogProb can be set relative to
	// it (SetStep must run before Forward populates PolicyOutput from
	// this exact Input, so run Forward once beforehand to read it).
	net.Graph.Forward()
	currentLogProb := actionLogProb(net.Actor.PolicyOutput.Val, action)

	// Push oldLogProb far below the current log-probability, forcing
	// ratio = exp(currentLogProb - oldLogProb) to be huge (well beyond
	// 1+clipEpsilon).
	oldLogProb := currentLogProb - 10
	net.SetStep(action, oldLogProb, advantage, 0)
	net.Graph.Forward()

	// With the ratio clamped to 1+clipEpsilon, and a positive advantage,
	// min(raw*advantage, clamped*advantage) == clamped*advantage (the
	// clamped surrogate is smaller), so the expected policy loss is
	// -(1+clipEpsilon)*advantage.
	expected := -(1 + clipEpsilon) * advantage
	actual := net.Loss.Val.Data[action]
	assert.InDelta(t, expected, actual, 1e-3)

	// Confirm this genuinely differs from what the *unclamped* ratio
	// would have produced, so the assertion above isn't accidentally
	// satisfied by both being equal.
	rawRatio := float32(math.Exp(float64(currentLogProb - oldLogProb)))
	unclampedLoss := -rawRatio * advantage
	assert.Greater(t, math.Abs(float64(unclampedLoss-actual)), 1.0)
}

// TestBuildLossValueTermIsSquaredError confirms the value loss term
// behaves like valueCoef*(value-returnTarget)^2, isolated by setting
// entropyCoef to 0 and clipEpsilon large enough that the policy
// surrogate term is unclipped, with advantage set to 0 so the policy
// term itself vanishes (min(0,0)=0), leaving only the value term.
func TestBuildLossValueTermIsSquaredError(t *testing.T) {
	rng := rand.New(rand.NewPCG(431, 433))
	actorParams := actorcritic.NewParams(rng, 4, 5, 3)
	valueCoef := float32(2.0)

	net, err := NewTrainingNetwork(actorParams, LossConfig{ClipEpsilon: 1.0, EntropyCoef: 0, ValueCoef: valueCoef})
	require.NoError(t, err)

	net.Actor.Input.Val.FillRand(rng, -1, 1)
	action := rl.Action(0)

	net.Graph.Forward()
	value := net.Actor.ValueOutput.Val.Data[0]

	returnTarget := value - 0.5 // a fixed, known difference from value
	net.SetStep(action, actionLogProb(net.Actor.PolicyOutput.Val, action), 0 /* advantage */, returnTarget)
	net.Graph.Forward()

	expected := valueCoef * 0.5 * 0.5
	assert.InDelta(t, expected, net.Loss.Val.Data[action], 1e-3)
}
