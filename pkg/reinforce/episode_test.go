package reinforce

import (
	"math/rand/v2"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/mat"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newProbs(values ...float32) *mat.Matrix {
	m := mat.New(len(values), 1)
	copy(m.Data, values)
	return m
}

// TestSampleMaskedActionWithNilMaskMatchesSampleAction confirms a nil
// mask reproduces SampleAction's output bit-for-bit for the same rng
// draw, across many seeds, rather than merely agreeing on average.
func TestSampleMaskedActionWithNilMaskMatchesSampleAction(t *testing.T) {
	probs := newProbs(0.1, 0.2, 0.3, 0.4)

	for seed := range 20 {
		want := SampleAction(probs, rand.New(rand.NewPCG(uint64(seed), uint64(seed+1))))

		got, err := SampleMaskedAction(probs, nil, rand.New(rand.NewPCG(uint64(seed), uint64(seed+1))))
		require.NoError(t, err)

		assert.Equal(t, want, got)
	}
}

// TestSampleMaskedActionWithAllTrueMaskMatchesSampleAction confirms an
// explicit all-true mask behaves identically to a nil mask.
func TestSampleMaskedActionWithAllTrueMaskMatchesSampleAction(t *testing.T) {
	probs := newProbs(0.1, 0.2, 0.3, 0.4)
	mask := []bool{true, true, true, true}

	for seed := range 20 {
		want := SampleAction(probs, rand.New(rand.NewPCG(uint64(seed), uint64(seed+1))))

		got, err := SampleMaskedAction(probs, mask, rand.New(rand.NewPCG(uint64(seed), uint64(seed+1))))
		require.NoError(t, err)

		assert.Equal(t, want, got)
	}
}

// TestSampleMaskedActionRenormalizesOverAllowedActions hand-computes
// the renormalized distribution over a mask that excludes some actions:
// allowed actions 0 and 2 have raw probabilities 0.1 and 0.3, which sum
// to 0.4, so their renormalized probabilities are 0.25 and 0.75.
func TestSampleMaskedActionRenormalizesOverAllowedActions(t *testing.T) {
	probs := newProbs(0.1, 0.2, 0.3, 0.4)
	mask := []bool{true, false, true, false}

	for seed := range 20 {
		draw := rand.New(rand.NewPCG(uint64(seed), uint64(seed+1))).Float32()

		action, err := SampleMaskedAction(probs, mask, rand.New(rand.NewPCG(uint64(seed), uint64(seed+1))))
		require.NoError(t, err)

		want := rl.Action(2)
		if draw <= 0.25 {
			want = rl.Action(0)
		}
		assert.Equal(t, want, action, "seed %d: draw=%v", seed, draw)
	}
}

func TestSampleMaskedActionRejectsAllFalseMask(t *testing.T) {
	probs := newProbs(0.25, 0.25, 0.25, 0.25)
	mask := []bool{false, false, false, false}

	_, err := SampleMaskedAction(probs, mask, rand.New(rand.NewPCG(1, 2)))
	assert.Error(t, err)
}

// TestSampleMaskedActionFallsBackToUniformWhenAllowedProbabilityUnderflows
// exercises the live-confirmed 2026-09-18 crash this guards against: mask
// allows one action whose own probability is exactly 0 (simulating a
// float32 softmax underflow, not a genuine zero-probability policy
// decision) alongside disallowed actions with real probability mass —
// SampleMaskedAction must return that allowed action rather than erroring,
// since it is mathematically the only legal choice regardless of how
// small (unrepresentably small, even) the policy's real preference for it
// is.
func TestSampleMaskedActionFallsBackToUniformWhenAllowedProbabilityUnderflows(t *testing.T) {
	probs := newProbs(0, 0.4, 0.6, 0)
	mask := []bool{true, false, false, false}

	action, err := SampleMaskedAction(probs, mask, rand.New(rand.NewPCG(1, 2)))
	require.NoError(t, err)
	assert.Equal(t, rl.Action(0), action)
}

// TestSampleMaskedActionFallsBackToUniformAmongSeveralUnderflowedActions
// is the same scenario with more than one allowed action, confirming the
// fallback distribution is uniform among all of them (not just the
// first) by checking every seed in a wide range lands on an allowed
// index.
func TestSampleMaskedActionFallsBackToUniformAmongSeveralUnderflowedActions(t *testing.T) {
	probs := newProbs(0, 0, 0.5, 0.5)
	mask := []bool{true, true, false, false}

	seen := map[rl.Action]bool{}
	for seed := range uint64(50) {
		action, err := SampleMaskedAction(probs, mask, rand.New(rand.NewPCG(seed, 0)))
		require.NoError(t, err)
		if action != 0 && action != 1 {
			t.Fatalf("action %d is not allowed by mask %v", action, mask)
		}
		seen[action] = true
	}
	assert.Len(t, seen, 2, "expected both allowed actions to be sampled across 50 seeds")
}

// TestSampleMaskedActionStillRejectsAllFalseMask confirms the new
// anyAllowed check didn't loosen the genuine "mask is entirely false"
// caller-error case TestSampleMaskedActionRejectsAllFalseMask already
// covers — kept as its own test, named to make the distinction from the
// underflow-fallback tests above explicit.
func TestSampleMaskedActionStillRejectsAllFalseMask(t *testing.T) {
	probs := newProbs(0.25, 0.25, 0.25, 0.25)
	mask := []bool{false, false, false, false}

	_, err := SampleMaskedAction(probs, mask, rand.New(rand.NewPCG(1, 2)))
	assert.Error(t, err)
}

func TestSampleMaskedActionRejectsWrongLengthMask(t *testing.T) {
	probs := newProbs(0.5, 0.5)
	mask := []bool{true, true, true}

	_, err := SampleMaskedAction(probs, mask, rand.New(rand.NewPCG(1, 2)))
	assert.Error(t, err)
}

func TestSampleMaskedActionWithProbabilitiesMatchesSampleMaskedAction(t *testing.T) {
	probs := newProbs(0.1, 0.2, 0.3, 0.4)
	mask := []bool{true, false, true, false}

	for seed := range 20 {
		want, err := SampleMaskedAction(probs, mask, rand.New(rand.NewPCG(uint64(seed), uint64(seed+1))))
		require.NoError(t, err)

		got, _, _, err := SampleMaskedActionWithProbabilities(probs, mask, rand.New(rand.NewPCG(uint64(seed), uint64(seed+1))))
		require.NoError(t, err)

		assert.Equal(t, want, got)
	}
}

func TestSampleMaskedActionWithProbabilitiesReturnsRawAndRenormalized(t *testing.T) {
	probs := newProbs(0.1, 0.2, 0.3, 0.4)
	mask := []bool{true, false, true, false}

	_, raw, renormalized, err := SampleMaskedActionWithProbabilities(probs, mask, rand.New(rand.NewPCG(1, 2)))
	require.NoError(t, err)

	assert.Equal(t, []float32{0.1, 0.2, 0.3, 0.4}, raw, "raw must be probs's own distribution, unaffected by mask")
	assert.Equal(t, []float32{0.25, 0, 0.75, 0}, renormalized, "allowed actions 0 and 2 (0.1, 0.3) renormalize to 0.25/0.75")
}

func TestSampleMaskedActionWithProbabilitiesNilMaskReturnsSameSliceForRawAndRenormalized(t *testing.T) {
	probs := newProbs(0.25, 0.25, 0.25, 0.25)

	_, raw, renormalized, err := SampleMaskedActionWithProbabilities(probs, nil, rand.New(rand.NewPCG(1, 2)))
	require.NoError(t, err)

	assert.Equal(t, probs.Data, raw)
	assert.Equal(t, raw, renormalized)
}

func TestSampleMaskedActionWithProbabilitiesRejectsAllFalseMask(t *testing.T) {
	probs := newProbs(0.25, 0.25, 0.25, 0.25)
	mask := []bool{false, false, false, false}

	_, _, _, err := SampleMaskedActionWithProbabilities(probs, mask, rand.New(rand.NewPCG(1, 2)))
	assert.Error(t, err)
}
