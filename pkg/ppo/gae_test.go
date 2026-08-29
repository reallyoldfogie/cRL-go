package ppo

import (
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/stretchr/testify/assert"
)

// rolloutFromRewardsAndValues builds a Rollout whose rewards and value
// estimates are the given fixtures, ignoring every other rl.Transition
// field (computeGAE only reads Reward and the parallel Values slice).
func rolloutFromRewardsAndValues(rewards, values []float32) *Rollout {
	transitions := make([]rl.Transition, len(rewards))
	for i, reward := range rewards {
		transitions[i] = rl.Transition{Reward: reward}
	}
	return &Rollout{
		Episode: &rl.Episode{Transitions: transitions},
		Values:  values,
	}
}

// TestComputeGAEWithNoDiscountingReducesToPlainReturns checks the
// well-known identity that GAE(λ=1) with gamma=1 and all value
// estimates at 0 reduces to the plain (undiscounted) reward-to-go:
// advantages[t] == returns[t] == sum of all rewards from t onward.
func TestComputeGAEWithNoDiscountingReducesToPlainReturns(t *testing.T) {
	rollout := rolloutFromRewardsAndValues([]float32{1, 1}, []float32{0, 0})

	advantages, returns := computeGAE(rollout, 1.0, 1.0)

	assert.Equal(t, []float32{2, 1}, advantages)
	assert.Equal(t, []float32{2, 1}, returns)
}

// TestComputeGAEWithDiscountingAndValues checks the general recurrence
// against hand-computed values for a 3-step trajectory with nonzero
// value estimates and gamma, lambda both less than 1.
//
// rewards = [1, 1, 1], values = [0.5, 0.5, 0.5], gamma = 0.9, lambda = 0.95
//
//	step 2 (last): nextValue=0
//	  delta = 1 + 0.9*0 - 0.5 = 0.5
//	  advantage[2] = 0.5 + 0.9*0.95*0 = 0.5
//	  return[2] = 0.5 + 0.5 = 1.0
//	step 1: nextValue=values[2]=0.5
//	  delta = 1 + 0.9*0.5 - 0.5 = 0.95
//	  advantage[1] = 0.95 + 0.9*0.95*0.5 = 1.3775
//	  return[1] = 1.3775 + 0.5 = 1.8775
//	step 0: nextValue=values[1]=0.5
//	  delta = 1 + 0.9*0.5 - 0.5 = 0.95
//	  advantage[0] = 0.95 + 0.9*0.95*1.3775 = 2.1277625
//	  return[0] = 2.1277625 + 0.5 = 2.6277625
func TestComputeGAEWithDiscountingAndValues(t *testing.T) {
	rollout := rolloutFromRewardsAndValues([]float32{1, 1, 1}, []float32{0.5, 0.5, 0.5})

	advantages, returns := computeGAE(rollout, 0.9, 0.95)

	checks := assert.New(t)
	checks.InDelta(2.1277625, advantages[0], 1e-4)
	checks.InDelta(1.3775, advantages[1], 1e-4)
	checks.InDelta(0.5, advantages[2], 1e-4)

	checks.InDelta(2.6277625, returns[0], 1e-4)
	checks.InDelta(1.8775, returns[1], 1e-4)
	checks.InDelta(1.0, returns[2], 1e-4)
}

func TestComputeGAEHandlesSingleStepEpisode(t *testing.T) {
	rollout := rolloutFromRewardsAndValues([]float32{3}, []float32{1})

	advantages, returns := computeGAE(rollout, 0.99, 0.95)

	// A single step has no next value to bootstrap from, so
	// advantage[0] = reward - value = 3 - 1 = 2, and return[0] =
	// advantage[0] + value = 3.
	assert.InDelta(t, 2.0, advantages[0], 1e-6)
	assert.InDelta(t, 3.0, returns[0], 1e-6)
}
