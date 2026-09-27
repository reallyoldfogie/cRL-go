package actorcritic

import (
	"math/rand/v2"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/mat"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestNet2WiderNetWithZeroNoisePreservesTheFunctionExactly is Net2WiderNet's
// central claim: widening a network computes the same policy and value
// output on any input, when nothing perturbs the newly-created units.
// Checked against real inference (NewInferenceNetwork), not hand-rolled
// arithmetic, so this exercises the same path training/eval actually use.
func TestNet2WiderNetWithZeroNoisePreservesTheFunctionExactly(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	inputSize, hiddenSize, outputSize := 17, 8, 4
	p := NewParams(rng, inputSize, hiddenSize, outputSize)

	wide, err := Net2WiderNet(p, 20, 0, rand.New(rand.NewPCG(9, 9)))
	require.NoError(t, err)
	require.Equal(t, 20, wide.HiddenSize())
	assert.Equal(t, inputSize, wide.InputSize())
	assert.Equal(t, outputSize, wide.OutputSize())

	for trial := 0; trial < 20; trial++ {
		obs := rl.Observation{Values: randomVector(rng, inputSize)}

		orig, err := forward(p, obs)
		require.NoError(t, err)
		got, err := forward(wide, obs)
		require.NoError(t, err)

		for i := range orig.policy {
			assert.InDelta(t, orig.policy[i], got.policy[i], 1e-5, "policy[%d] on trial %d", i, trial)
		}
		assert.InDelta(t, orig.value, got.value, 1e-5, "value on trial %d", trial)
	}
}

// TestNet2WiderNetLeavesPureProducerParametersAtTheirOriginalIndexUntouched
// confirms the part of the transform noiseStd never touches: a parameter
// that is only ever a producer in this network — W0/B0 (nothing widens
// their input side) and B1 (a bias has no "consumer" of its own) — keeps
// every original unit's row copied byte-for-byte at its original index in
// the widened network. W1 is deliberately not checked here: it is also a
// consumer (see Net2WiderNet's own doc comment on its two boundaries), so
// even its "kept" rows change content whenever the column they read from
// was also cloned — TestWidenConsumerAxisSplitsADuplicatedUnitsWeightSoTheSumIsUnchanged
// and the end-to-end forward-pass equivalence test above are what actually
// pin W1's correctness.
func TestNet2WiderNetLeavesPureProducerParametersAtTheirOriginalIndexUntouched(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	p := NewParams(rng, 5, 6, 2)

	wide, err := Net2WiderNet(p, 11, 0.5, rand.New(rand.NewPCG(5, 6)))
	require.NoError(t, err)

	assertRowsEqual(t, p.W0, wide.W0, p.W0.Rows)
	assertRowsEqual(t, p.B0, wide.B0, p.B0.Rows)
	assertRowsEqual(t, p.B1, wide.B1, p.B1.Rows)

	// Wpi/Wv's own rows (the action/value outputs) are unchanged in count;
	// only their columns (hidden inputs) grow, and every original column j
	// must still carry a nonzero share of that unit's original weight, never
	// zero (a unit's contribution never gets silently dropped).
	assert.Equal(t, p.Wpi.Rows, wide.Wpi.Rows)
	assert.Equal(t, p.Wv.Rows, wide.Wv.Rows)
	for c := 0; c < p.W0.Rows; c++ {
		for r := 0; r < wide.Wpi.Rows; r++ {
			if p.Wpi.Data[r*p.Wpi.Cols+c] != 0 {
				assert.NotZero(t, wide.Wpi.Data[r*wide.Wpi.Cols+c])
			}
		}
	}
}

func assertRowsEqual(t *testing.T, orig, wide *mat.Matrix, n int) {
	t.Helper()
	require.LessOrEqual(t, n, wide.Rows)
	for r := 0; r < n; r++ {
		assert.Equal(t, orig.Data[r*orig.Cols:(r+1)*orig.Cols], wide.Data[r*wide.Cols:(r+1)*wide.Cols], "row %d", r)
	}
}

// TestWidenProducerAxisCopiesOriginalRowsExactlyAndOnlyNoisesNewOnes pins
// widenProducerAxis directly (the function W1's row-widening pass shares
// with W0/B0/B1, but which TestNet2WiderNetLeavesPureProducerParameters...
// above can't isolate for W1 since its input is already consumer-scaled by
// then): every row below the original count is copied verbatim; every row
// at or beyond it is that same source row plus noise (nonzero, since a
// Gaussian draw landing on exactly zero is negligible).
func TestWidenProducerAxisCopiesOriginalRowsExactlyAndOnlyNoisesNewOnes(t *testing.T) {
	m := &mat.Matrix{Rows: 2, Cols: 2, Data: []float32{1, 2, 3, 4}}
	mapTo := []int{0, 1, 0} // kept 0, kept 1, one new clone of 0
	rng := rand.New(rand.NewPCG(1, 1))

	out := widenProducerAxis(m, mapTo, 0.3, rng)

	require.Equal(t, 3, out.Rows)
	assert.Equal(t, []float32{1, 2}, out.Data[0:2], "row 0 kept exactly")
	assert.Equal(t, []float32{3, 4}, out.Data[2:4], "row 1 kept exactly")
	assert.NotEqual(t, []float32{1, 2}, out.Data[4:6], "row 2 (new) must differ from its source by the noise")
}

// TestWidenConsumerAxisSplitsADuplicatedUnitsWeightSoTheSumIsUnchanged pins
// the scaling math directly, on numbers simple enough to check by hand:
// widening [1, 2] to 4 positions where the new positions both clone index 0
// must produce [1/3, 1/3, 1/3, 2] (three copies of unit 0's weight summing
// back to the original 1, unit 1 untouched since nothing cloned it).
func TestWidenConsumerAxisSplitsADuplicatedUnitsWeightSoTheSumIsUnchanged(t *testing.T) {
	m := &mat.Matrix{Rows: 1, Cols: 2, Data: []float32{1, 2}}
	mapTo := []int{0, 1, 0, 0} // kept 0, kept 1, two new clones of 0
	count := []int{3, 1}       // unit 0: itself + 2 clones; unit 1: itself only

	out := widenConsumerAxis(m, mapTo, count)

	require.Equal(t, 1, out.Rows)
	require.Equal(t, 4, out.Cols)
	assert.InDelta(t, float32(1)/3, out.Data[0], 1e-6)
	assert.InDelta(t, float32(2), out.Data[1], 1e-6)
	assert.InDelta(t, float32(1)/3, out.Data[2], 1e-6)
	assert.InDelta(t, float32(1)/3, out.Data[3], 1e-6)

	var sum float32
	for _, k := range []int{0, 2, 3} { // every position representing unit 0
		sum += out.Data[k]
	}
	assert.InDelta(t, float32(1), sum, 1e-6, "the three positions representing unit 0 must sum back to its original weight")
}

func TestNet2WiderNetRejectsShrinkingOrEqualSize(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 1))
	p := NewParams(rng, 5, 8, 2)

	_, err := Net2WiderNet(p, 8, 0, rng)
	assert.Error(t, err, "equal size must be rejected, not silently a no-op")

	_, err = Net2WiderNet(p, 4, 0, rng)
	assert.Error(t, err, "shrinking must be rejected")
}

// TestNet2WiderNetDoesNotMutateTheOriginal guards against a widen call
// corrupting the checkpoint it started from, which would be especially bad
// for a tool applying this to a saved generation in place.
func TestNet2WiderNetDoesNotMutateTheOriginal(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 2))
	p := NewParams(rng, 5, 8, 2)
	before := p.Snapshot()

	_, err := Net2WiderNet(p, 12, 0.1, rand.New(rand.NewPCG(3, 3)))
	require.NoError(t, err)

	assert.Equal(t, before.W0.Data, p.W0.Data)
	assert.Equal(t, before.W1.Data, p.W1.Data)
	assert.Equal(t, before.Wpi.Data, p.Wpi.Data)
	assert.Equal(t, before.Wv.Data, p.Wv.Data)
}

func randomVector(rng *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	return v
}

type forwardResult struct {
	policy []float32
	value  float32
}

func forward(p *Params, obs rl.Observation) (forwardResult, error) {
	net, err := NewInferenceNetwork(p)
	if err != nil {
		return forwardResult{}, err
	}
	copy(net.Input.Val.Data, obs.Values)
	net.Graph.Forward()
	policy := make([]float32, len(net.PolicyOutput.Val.Data))
	copy(policy, net.PolicyOutput.Val.Data)
	return forwardResult{policy: policy, value: net.ValueOutput.Val.Data[0]}, nil
}
