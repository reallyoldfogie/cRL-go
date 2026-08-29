// Package ppo implements Proximal Policy Optimization on top of
// pkg/actorcritic's actor-critic network: collecting rollouts that also
// capture each step's action log-probability and value estimate,
// computing Generalized Advantage Estimation (GAE(λ)) advantages and
// value-target returns from them, and composing the clipped-surrogate
// policy loss, value loss, and entropy bonus into a single trainable
// objective. See docs/plans/03-gae-and-ppo-objective.md.
package ppo

import "github.com/reallyoldfogie/cRL-go/pkg/rl"

// Rollout pairs a collected rl.Episode with the per-step data PPO needs
// that a plain rl.Episode doesn't carry: the sampled action's
// log-probability under the policy that generated it (LogProbs[t]) and
// the value estimate at that step (Values[t]), both indexed the same
// way as Episode.Transitions[t]. This mirrors how pkg/reinforce's
// scoredEpisode pairs an *rl.Episode with algorithm-specific Returns
// rather than adding fields to rl.Transition itself.
type Rollout struct {
	Episode  *rl.Episode
	LogProbs []float32
	Values   []float32
}

// computeGAE computes Generalized Advantage Estimation (GAE(λ))
// advantages and value-target returns for every step of rollout, using
// the standard recurrence computed backward through the trajectory:
//
//	delta[t] = reward[t] + gamma*value[t+1] - value[t]
//	advantage[t] = delta[t] + gamma*lambda*advantage[t+1]
//	return[t] = advantage[t] + value[t]
//
// value[t+1] is treated as 0 at the last step (bootstrapping from a
// terminal or step-limit-truncated state, rather than from a further
// value estimate that was never computed). Setting lambda=1 reduces
// this to the plain discounted reward-to-go pkg/reinforce's
// computeReturns already implements (a well-known identity for GAE),
// which TestComputeGAEWithNoDiscountingReducesToPlainReturns checks
// directly.
func computeGAE(rollout *Rollout, gamma, lambda float32) (advantages, returns []float32) {
	stepCount := len(rollout.Episode.Transitions)
	advantages = make([]float32, stepCount)
	returns = make([]float32, stepCount)

	var runningAdvantage float32
	for step := stepCount - 1; step >= 0; step-- {
		var nextValue float32
		if step < stepCount-1 {
			nextValue = rollout.Values[step+1]
		}

		reward := rollout.Episode.Transitions[step].Reward
		value := rollout.Values[step]
		delta := reward + gamma*nextValue - value

		runningAdvantage = delta + gamma*lambda*runningAdvantage
		advantages[step] = runningAdvantage
		returns[step] = advantages[step] + value
	}
	return advantages, returns
}
