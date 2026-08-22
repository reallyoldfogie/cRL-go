package policy

import (
	"math"
	"math/rand/v2"
	"testing"

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

	net, err := NewTrainingNetwork(params)
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

	net, err := NewTrainingNetwork(params)
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
	net, err := NewTrainingNetwork(params)
	require.NoError(t, err)

	before := append([]float32(nil), params.W0.Data...)
	net.ApplyGradientStep(0.1, 0)
	assert.Equal(t, before, params.W0.Data)
}
