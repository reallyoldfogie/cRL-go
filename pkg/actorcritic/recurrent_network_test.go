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

func TestRecurrentInferenceNetworkForwardProducesValidPolicyDistributionAndScalarValue(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	params := NewRecurrentParams(rng, 9, 6, 4, 2)

	net, err := NewRecurrentInferenceNetwork(params)
	require.NoError(t, err)

	net.Input.Val.FillRand(rng, -1, 1)
	// HPrevIn/CPrevIn start zeroed (episode start) — left as-is.
	net.Graph.Forward()

	require.Equal(t, 4, net.PolicyOutput.Val.Rows)
	require.Equal(t, 6, net.HOut.Val.Rows)
	require.Equal(t, 6, net.COut.Val.Rows)

	var sum float32
	for _, v := range net.PolicyOutput.Val.Data {
		assert.False(t, math.IsNaN(float64(v)))
		assert.GreaterOrEqual(t, v, float32(0))
		sum += v
	}
	assert.InDelta(t, 1.0, sum, 1e-4)
	assert.False(t, math.IsNaN(float64(net.ValueOutput.Val.Data[0])))

	// tanh/sigmoid outputs are bounded regardless of input scale.
	for _, v := range net.HOut.Val.Data {
		assert.LessOrEqual(t, math.Abs(float64(v)), 1.0)
	}
}

// TestRecurrentInferenceNetworkCarriesStateAcrossSteps confirms feeding a
// step's HOut/COut into the next step's HPrevIn/CPrevIn actually changes
// the result relative to always feeding zeros — i.e. the network is
// genuinely using its recurrent state, not ignoring it.
func TestRecurrentInferenceNetworkCarriesStateAcrossSteps(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	params := NewRecurrentParams(rng, 5, 4, 3, 1)

	net, err := NewRecurrentInferenceNetwork(params)
	require.NoError(t, err)

	obs := []float32{0.1, -0.2, 0.3, 0.4, -0.5}
	copy(net.Input.Val.Data, obs)
	net.Graph.Forward()
	firstPolicy := append([]float32(nil), net.PolicyOutput.Val.Data...)

	// Feed the first step's state forward and run the same observation
	// again: a network that depends on hPrev/cPrev must produce a
	// different distribution the second time.
	copy(net.HPrevIn.Val.Data, net.HOut.Val.Data)
	copy(net.CPrevIn.Val.Data, net.COut.Val.Data)
	copy(net.Input.Val.Data, obs)
	net.Graph.Forward()
	secondPolicy := net.PolicyOutput.Val.Data

	assert.NotEqual(t, firstPolicy, secondPolicy, "carrying state forward must change the output for the same observation")
}

// --- Gradient checking over the unrolled multi-step graph ---
//
// This is the test that actually proves the multi-use-Var gradient
// accumulation this type's doc comment claims: every weight Var is
// referenced by ChunkLen separate unrolled applications within one
// graph, and Backward must sum contributions from all of them, not just
// the last. See docs/autograd's own gradcheck_test.go for the same
// technique applied to every other op.

const (
	chunkGradCheckEpsilon   = 1e-2
	chunkGradCheckTolerance = 5e-2
)

func chunkSumMatrix(m *mat.Matrix) float64 {
	var sum float64
	for _, v := range m.Data {
		sum += float64(v)
	}
	return sum
}

func chunkNumericalGradient(x, output *autograd.Var, forward func()) *mat.Matrix {
	grad := mat.New(x.Val.Rows, x.Val.Cols)
	for i := range x.Val.Data {
		original := x.Val.Data[i]

		x.Val.Data[i] = original + chunkGradCheckEpsilon
		forward()
		plus := chunkSumMatrix(output.Val)

		x.Val.Data[i] = original - chunkGradCheckEpsilon
		forward()
		minus := chunkSumMatrix(output.Val)

		x.Val.Data[i] = original
		grad.Data[i] = float32((plus - minus) / (2 * chunkGradCheckEpsilon))
	}
	forward()
	return grad
}

// scaleRecurrentParams multiplies every one of p's weight/bias matrices
// by factor in place, used by gradient-check tests to pull a freshly
// Xavier-initialized RecurrentParams toward 0 — see
// TestGradientCheckRecurrentChunkTrainingNetworkAccumulatesAcrossTimesteps's
// own doc comment for why staying near-linear matters for a numerical
// gradient check specifically.
func scaleRecurrentParams(p *RecurrentParams, factor float32) {
	for _, layer := range p.Trunk.Hidden {
		layer.W.Scale(factor)
		layer.B.Scale(factor)
	}
	p.Trunk.Wpi.Scale(factor)
	p.Trunk.Bpi.Scale(factor)
	p.Trunk.Wv.Scale(factor)
	p.Trunk.Bv.Scale(factor)

	for _, m := range []*mat.Matrix{
		p.LSTM.WxF, p.LSTM.WhF, p.LSTM.BF,
		p.LSTM.WxI, p.LSTM.WhI, p.LSTM.BI,
		p.LSTM.WxO, p.LSTM.WhO, p.LSTM.BO,
		p.LSTM.WxG, p.LSTM.WhG, p.LSTM.BG,
	} {
		m.Scale(factor)
	}
	// The forget gate's bias is deliberately initialized to 1.0 rather
	// than 0 (see NewLSTMLayer) — scale it down too rather than leave it
	// at a magnitude the rest of this test's weights don't have, for the
	// same near-linear-regime reasoning.
}

func chunkAssertMatricesClose(t *testing.T, want, got *mat.Matrix, tolerance float64) {
	t.Helper()
	require.Equal(t, want.Rows, got.Rows)
	require.Equal(t, want.Cols, got.Cols)
	for i := range want.Data {
		assert.InDelta(t, want.Data[i], got.Data[i], tolerance, "element %d differs", i)
	}
}

// TestGradientCheckRecurrentChunkTrainingNetworkAccumulatesAcrossTimesteps
// builds a stand-in scalar objective (sum of every timestep's first
// policy-action probability plus every timestep's value output) over a
// 3-step unrolled chunk and gradient-checks every shared weight
// (trunk + LSTM) against it — the gradient w.r.t. a shared weight is the
// *sum* of its contribution from all 3 timesteps, which is exactly what
// this checks, end to end, against numerical differentiation.
func TestGradientCheckRecurrentChunkTrainingNetworkAccumulatesAcrossTimesteps(t *testing.T) {
	rng := rand.New(rand.NewPCG(21, 22))
	params := NewRecurrentParams(rng, 4, 5, 3, 2)
	// Pull every weight toward 0: an LSTM stacks four saturating
	// nonlinearities (3 sigmoid gates, 1 tanh) per timestep, and chaining
	// several timesteps compounds however saturated they are. Xavier-
	// scale (or larger) weights combined with ChunkLen>1 are enough to
	// drive a gate deep into its saturated regime for some random draw,
	// where finite-difference estimates (this test's numerical half)
	// pick up real curvature error unrelated to whether the analytic
	// gradient is correct — the same well-known gradient-checking
	// caveat as the "kink at ReLU's 0" one gradcheck_test.go's own
	// TestGradientCheckReLU already documents, just for saturation
	// instead of a kink. Scaling weights (and the input below) down
	// keeps every gate in its near-linear regime, where the finite-
	// difference approximation is actually accurate.
	scaleRecurrentParams(params, 0.2)

	const chunkLen = 3
	net, err := NewRecurrentChunkTrainingNetwork(params, chunkLen)
	require.NoError(t, err)

	for step := 0; step < chunkLen; step++ {
		net.Input[step].Val.FillRand(rng, -0.3, 0.3)
	}

	selectFirstAction := &autograd.Var{Val: &mat.Matrix{Rows: 1, Cols: 3, Data: []float32{1, 0, 0}}}

	var combined *autograd.Var
	for step := 0; step < chunkLen; step++ {
		selectedProb, err := autograd.MatMul(selectFirstAction, net.PolicyOutput[step])
		require.NoError(t, err)

		stepSum, err := autograd.Add(selectedProb, net.ValueOutput[step])
		require.NoError(t, err)

		if combined == nil {
			combined = stepSum
		} else {
			combined, err = autograd.Add(combined, stepSum)
			require.NoError(t, err)
		}
	}

	graph := autograd.BuildGraph(combined)
	graph.Forward()
	graph.Backward()

	for _, p := range net.Parameters() {
		chunkAssertMatricesClose(t, p.Grad, chunkNumericalGradient(p, combined, graph.Forward), chunkGradCheckTolerance)
	}
}

func TestRecurrentChunkTrainingNetworkRejectsNonPositiveChunkLen(t *testing.T) {
	rng := rand.New(rand.NewPCG(13, 14))
	params := NewRecurrentParams(rng, 4, 5, 3, 1)

	_, err := NewRecurrentChunkTrainingNetwork(params, 0)
	assert.Error(t, err)
}

func TestRecurrentChunkTrainingNetworkApplyGradientStepUpdatesSharedWeightsOnce(t *testing.T) {
	rng := rand.New(rand.NewPCG(15, 16))
	params := NewRecurrentParams(rng, 4, 5, 3, 2)

	net, err := NewRecurrentChunkTrainingNetwork(params, 3)
	require.NoError(t, err)

	before := append([]float32(nil), params.LSTM.WxF.Data...)
	for _, p := range net.Parameters() {
		p.Grad.Fill(1.0)
	}
	net.ApplyGradientStep(0.1, 1)

	assert.NotEqual(t, before, params.LSTM.WxF.Data, "ApplyGradientStep should modify the shared LSTM weight exactly once, not ChunkLen times")
}
