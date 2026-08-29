package actorcritic

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/autograd"
	"github.com/reallyoldfogie/cRL-go/pkg/mat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestInferenceNetworkForwardProducesValidPolicyDistributionAndScalarValue(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	params := NewParams(rng, 12, 8, 5)

	net, err := NewInferenceNetwork(params)
	require.NoError(t, err)

	net.Input.Val.FillRand(rng, -1, 1)
	net.Graph.Forward()

	require.Equal(t, 5, net.PolicyOutput.Val.Rows)
	require.Equal(t, 1, net.PolicyOutput.Val.Cols)
	require.Equal(t, 1, net.ValueOutput.Val.Rows)
	require.Equal(t, 1, net.ValueOutput.Val.Cols)

	var sum float32
	for _, v := range net.PolicyOutput.Val.Data {
		assert.False(t, math.IsNaN(float64(v)))
		assert.GreaterOrEqual(t, v, float32(0))
		sum += v
	}
	assert.InDelta(t, 1.0, sum, 1e-4)
	assert.False(t, math.IsNaN(float64(net.ValueOutput.Val.Data[0])))
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

	assert.Equal(t, netA.PolicyOutput.Val.Data, netB.PolicyOutput.Val.Data,
		"networks sharing params and given the same input must produce identical policy output")
	assert.Equal(t, netA.ValueOutput.Val.Data, netB.ValueOutput.Val.Data,
		"networks sharing params and given the same input must produce identical value output")
}

func TestZeroGradClearsAccumulatedGradients(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	params := NewParams(rng, 6, 4, 3)

	net, err := NewTrainingNetwork(params)
	require.NoError(t, err)

	for _, p := range net.params.all() {
		p.Grad.Fill(1.0)
	}

	net.ZeroGrad()
	for _, p := range net.params.all() {
		for _, v := range p.Grad.Data {
			assert.Equal(t, float32(0), v)
		}
	}
}

func TestApplyGradientStepUpdatesParameters(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 10))
	params := NewParams(rng, 6, 4, 3)

	net, err := NewTrainingNetwork(params)
	require.NoError(t, err)

	before := append([]float32(nil), params.W0.Data...)
	for _, p := range net.params.all() {
		p.Grad.Fill(1.0)
	}
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

// --- Gradient checking ---
//
// Duplicated from pkg/autograd/gradcheck_test.go's helpers (test files
// can't be imported across packages) so this package's forward wiring
// can be validated the same way every autograd op already is.

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

	forward() // leave output.Val in its unperturbed state
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

// TestGradientCheckTrainingNetworkStandInObjective builds a stand-in
// combined objective from TrainingNetwork's two heads — the policy
// head's first action probability, reduced to a scalar via a dot
// product (the same vector-to-scalar trick
// TestGradientCheckSoftmax in pkg/autograd/gradcheck_test.go uses),
// plus the value head's squared error against a fixed target — and
// gradient-checks every one of the network's eight parameter matrices
// against it.
//
// This is deliberately not the real PPO loss (GAE and the
// clipped-surrogate objective arrive separately); it only needs to
// exercise every parameter's gradient path through the shared trunk and
// both heads together.
func TestGradientCheckTrainingNetworkStandInObjective(t *testing.T) {
	rng := rand.New(rand.NewPCG(101, 103))
	params := NewParams(rng, 4, 5, 3)

	net, err := NewTrainingNetwork(params)
	require.NoError(t, err)
	net.Input.Val.FillRand(rng, -1, 1)

	selectFirstAction := &autograd.Var{Val: &mat.Matrix{Rows: 1, Cols: 3, Data: []float32{1, 0, 0}}}
	selectedProb, err := autograd.MatMul(selectFirstAction, net.PolicyOutput)
	require.NoError(t, err)

	returnTarget := &autograd.Var{Val: &mat.Matrix{Rows: 1, Cols: 1, Data: []float32{0.5}}}
	valueError, err := autograd.Sub(net.ValueOutput, returnTarget)
	require.NoError(t, err)
	valueLoss, err := autograd.Mul(valueError, valueError)
	require.NoError(t, err)

	combined, err := autograd.Add(selectedProb, valueLoss)
	require.NoError(t, err)

	graph := autograd.BuildGraph(combined)
	graph.Forward()
	graph.Backward()

	for _, p := range net.params.all() {
		assertMatricesClose(t, p.Grad, numericalGradient(p, combined, graph.Forward), gradCheckTolerance)
	}
}
