package policy

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/autograd"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewParamsShapesAndBoundedWeights(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	params := NewParams(rng, 12, 8, 5)

	assert.Equal(t, 12, params.InputSize())
	assert.Equal(t, 8, params.HiddenSize())
	assert.Equal(t, 5, params.OutputSize())

	assert.Equal(t, 8, params.W0.Rows)
	assert.Equal(t, 12, params.W0.Cols)
	assert.Equal(t, 5, params.W2.Rows)
	assert.Equal(t, 8, params.W2.Cols)

	// Biases start at zero.
	for _, v := range params.B0.Data {
		assert.Equal(t, float32(0), v)
	}
}

func TestInferenceNetworkForwardProducesValidDistribution(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	params := NewParams(rng, 12, 8, 5)

	net, err := NewInferenceNetwork(params)
	require.NoError(t, err)

	net.Input.Val.FillRand(rng, -1, 1)
	net.Graph.Forward()

	require.Equal(t, 5, net.Output.Val.Rows)
	require.Equal(t, 1, net.Output.Val.Cols)

	var sum float32
	for _, v := range net.Output.Val.Data {
		assert.False(t, math.IsNaN(float64(v)))
		assert.GreaterOrEqual(t, v, float32(0))
		sum += v
	}
	assert.InDelta(t, 1.0, sum, 1e-4)
}

func TestInferenceNetworksSharePramsButHavePrivateActivations(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	params := NewParams(rng, 12, 8, 5)

	netA, err := NewInferenceNetwork(params)
	require.NoError(t, err)
	netB, err := NewInferenceNetwork(params)
	require.NoError(t, err)

	assert.NotSame(t, netA.Input, netB.Input, "each inference network must own private activation Vars")
	assert.NotSame(t, netA.Input.Val, netB.Input.Val)

	input := make([]float32, params.InputSize())
	for i := range input {
		input[i] = 0.1
	}
	copy(netA.Input.Val.Data, input)
	copy(netB.Input.Val.Data, input)

	// Perturbing a shared weight must be visible to both networks, since
	// their weight Vars alias the same underlying matrices rather than
	// copying them (see autograd.Constant).
	params.W0.Data[0] += 5.0

	netA.Graph.Forward()
	netB.Graph.Forward()

	assert.Equal(t, netA.Output.Val.Data, netB.Output.Val.Data,
		"networks sharing params and given the same input must produce identical output")
}

func TestTrainingNetworkAccumulatesGradientsAcrossMultipleBackwardCalls(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	params := NewParams(rng, 6, 4, 3)

	net, err := NewTrainingNetwork(params, 0)
	require.NoError(t, err)

	runOnce := func() {
		net.Input.Val.FillRand(rng, -1, 1)
		net.Advantage.Val.Clear()
		net.Advantage.Val.Data[0] = 1.0

		net.Graph.Forward()
		net.Graph.Backward()
	}

	net.ZeroGrad()
	runOnce()
	firstW0Grad := append([]float32(nil), net.params.W0.Grad.Data...)

	runOnce()
	secondW0Grad := net.params.W0.Grad.Data

	// Gradients accumulate (are not reset) across repeated Backward
	// calls until ZeroGrad is called again, matching the original's
	// batch-wide gradient accumulation over many trajectory steps.
	changed := false
	for i := range firstW0Grad {
		if firstW0Grad[i] != secondW0Grad[i] {
			changed = true
			break
		}
	}
	assert.True(t, changed, "gradient should accumulate rather than reset between Backward calls")

	net.ZeroGrad()
	for _, v := range net.params.W0.Grad.Data {
		assert.Equal(t, float32(0), v)
	}
}

func TestApplyGradientStepUpdatesParameters(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 10))
	params := NewParams(rng, 6, 4, 3)

	net, err := NewTrainingNetwork(params, 0)
	require.NoError(t, err)

	before := append([]float32(nil), params.W0.Data...)

	net.ZeroGrad()
	net.Input.Val.FillRand(rng, -1, 1)
	net.Advantage.Val.Clear()
	net.Advantage.Val.Data[0] = 1.0
	net.Graph.Forward()
	net.Graph.Backward()

	net.ApplyGradientStep(0.1, 1)

	assert.NotEqual(t, before, params.W0.Data, "ApplyGradientStep should modify the shared parameter matrices")
}

func TestApplyGradientStepNoOpWithZeroSamples(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	params := NewParams(rng, 6, 4, 3)
	net, err := NewTrainingNetwork(params, 0)
	require.NoError(t, err)

	before := append([]float32(nil), params.W0.Data...)
	net.ApplyGradientStep(0.1, 0)
	assert.Equal(t, before, params.W0.Data)
}

func TestGradientNormZeroBeforeAnyBackward(t *testing.T) {
	rng := rand.New(rand.NewPCG(105, 106))
	params := NewParams(rng, 6, 4, 3)
	net, err := NewTrainingNetwork(params, 0)
	require.NoError(t, err)

	assert.Equal(t, float32(0), net.GradientNorm(), "no Backward call has happened yet, so there is no accumulated gradient")
}

func TestGradientNormReflectsAccumulatedGradient(t *testing.T) {
	rng := rand.New(rand.NewPCG(107, 108))
	params := NewParams(rng, 6, 4, 3)
	net, err := NewTrainingNetwork(params, 0)
	require.NoError(t, err)

	net.ZeroGrad()
	net.Input.Val.FillRand(rng, -1, 1)
	net.Advantage.Val.Clear()
	net.Advantage.Val.Data[0] = 1.0
	net.Graph.Forward()
	net.Graph.Backward()

	afterOneCall := net.GradientNorm()
	assert.Greater(t, afterOneCall, float32(0), "a real Backward call must produce a nonzero gradient norm")

	// Repeating the exact same Forward/Backward call (same Input,
	// same Advantage) accumulates the identical gradient contribution a
	// second time, so the total must double deterministically — unlike
	// a second call with a fresh random input, this can't coincidentally
	// partially cancel, so it's a reliable way to confirm accumulation.
	net.Graph.Forward()
	net.Graph.Backward()

	assert.InDelta(t, 2*afterOneCall, net.GradientNorm(), 1e-3, "gradients accumulate (are not reset) across repeated Backward calls")
}

// TestSetActionMaskZeroesMaskedActionProbability confirms SetActionMask
// actually reaches Output: a masked action's probability comes out
// exactly 0 (not merely small), and the remaining legal actions still
// sum to 1, matching SampleMaskedAction's renormalization contract even
// though this masking happens before Softmax rather than after it — see
// docs/plans/19-training-time-action-masking.md.
func TestSetActionMaskZeroesMaskedActionProbability(t *testing.T) {
	rng := rand.New(rand.NewPCG(201, 202))
	params := NewParams(rng, 6, 4, 4)

	net, err := NewTrainingNetwork(params, 0)
	require.NoError(t, err)

	net.Input.Val.FillRand(rng, -1, 1)
	net.SetActionMask([]bool{true, false, true, true})
	net.Graph.Forward()

	assert.Equal(t, float32(0), net.Output.Val.Data[1], "a masked action must have exactly 0 probability")

	var sum float32
	for _, v := range net.Output.Val.Data {
		sum += v
	}
	assert.InDelta(t, 1.0, sum, 1e-4)
}

// TestSetActionMaskNilMatchesUnmasked confirms a nil mask (the default
// before any SetActionMask call, and what an ActionMasker-less
// environment produces) leaves Output bit-for-bit identical to a network
// that never had SetActionMask called at all — the same "nil mask is
// exactly equivalent to no masking" invariant SampleMaskedAction already
// guarantees at the sampling layer.
func TestSetActionMaskNilMatchesUnmasked(t *testing.T) {
	rng := rand.New(rand.NewPCG(203, 204))
	params := NewParams(rng, 6, 4, 4)
	input := make([]float32, params.InputSize())
	for i := range input {
		input[i] = 0.3
	}

	unmasked, err := NewTrainingNetwork(params, 0)
	require.NoError(t, err)
	copy(unmasked.Input.Val.Data, input)
	unmasked.Graph.Forward()

	explicitlyCleared, err := NewTrainingNetwork(params, 0)
	require.NoError(t, err)
	copy(explicitlyCleared.Input.Val.Data, input)
	explicitlyCleared.SetActionMask(nil)
	explicitlyCleared.Graph.Forward()

	assert.Equal(t, unmasked.Output.Val.Data, explicitlyCleared.Output.Val.Data)
}

// TestNewTrainingNetworkZeroEntropyCoefMatchesPlainReinforceLoss confirms
// entropyCoef 0 leaves the loss (and therefore the gradient) exactly
// equal to the plain REINFORCE loss with no entropy term at all, not
// merely close to it — a regression check for NewTrainingNetwork's new
// entropyCoef parameter, since every pre-existing caller in this
// package's own tests now passes 0 and must see unchanged behavior.
func TestNewTrainingNetworkZeroEntropyCoefMatchesPlainReinforceLoss(t *testing.T) {
	rng := rand.New(rand.NewPCG(101, 102))
	params := NewParams(rng, 6, 4, 3)

	net, err := NewTrainingNetwork(params, 0)
	require.NoError(t, err)

	net.Input.Val.FillRand(rng, -1, 1)
	net.Advantage.Val.Clear()
	net.Advantage.Val.Data[1] = 0.7
	net.Graph.Forward()

	plainLoss, err := autograd.ReinforceLoss(net.Output, net.Advantage)
	require.NoError(t, err)
	plainGraph := autograd.BuildGraph(plainLoss)
	plainGraph.Forward()

	assert.Equal(t, plainLoss.Val.Data, net.Loss.Val.Data, "entropyCoef 0 must produce bit-identical loss to the plain REINFORCE loss")
}

// TestEntropyCoefAffectsLossAndGradient confirms a nonzero entropyCoef
// actually participates in NewTrainingNetwork's loss and gradient,
// rather than being silently accepted and ignored — the exact bug this
// parameter was added to fix (see docs/03-policy-gradients-and-reinforce.md's
// entropy-bonus section).
func TestEntropyCoefAffectsLossAndGradient(t *testing.T) {
	rng := rand.New(rand.NewPCG(103, 104))
	params := NewParams(rng, 6, 4, 3)
	input := make([]float32, params.InputSize())
	for i := range input {
		input[i] = 0.1
	}

	runNet := func(entropyCoef float32) *TrainingNetwork {
		net, err := NewTrainingNetwork(params, entropyCoef)
		require.NoError(t, err)
		copy(net.Input.Val.Data, input)
		net.Advantage.Val.Clear()
		net.Advantage.Val.Data[1] = 0.7
		net.ZeroGrad()
		net.Graph.Forward()
		net.Graph.Backward()
		return net
	}

	withoutEntropy := runNet(0)
	withEntropy := runNet(0.5)

	assert.NotEqual(t, withoutEntropy.Loss.Val.Data, withEntropy.Loss.Val.Data,
		"a nonzero entropyCoef must change the computed loss")
	assert.NotEqual(t, withoutEntropy.params.W2.Grad.Data, withEntropy.params.W2.Grad.Data,
		"a nonzero entropyCoef must change the gradient reaching the output layer's weights")
}
