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

func TestSampleMaskedActionRejectsWrongLengthMask(t *testing.T) {
	probs := newProbs(0.5, 0.5)
	mask := []bool{true, true, true}

	_, err := SampleMaskedAction(probs, mask, rand.New(rand.NewPCG(1, 2)))
	assert.Error(t, err)
}
