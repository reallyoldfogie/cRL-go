// Package reinforce implements the REINFORCE (vanilla policy-gradient)
// training loop: collecting rollouts from the current policy against any
// rl.Environment, computing discounted returns and a batch-wide
// baseline, and applying a gradient step. See
// docs/03-policy-gradients-and-reinforce.md for the algorithm this
// implements, and docs/05-porting-notes.md for how this differs
// structurally from the training loop in env.c of
// github.com/harshbhatt7585/cRL.
package reinforce

import (
	"math/rand/v2"

	"github.com/reallyoldfogie/cRL-go/pkg/mat"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

// scoredEpisode pairs a collected rl.Episode with its discounted
// reward-to-go at every step (Returns[t] corresponds to
// Episode.Transitions[t]). Returns are algorithm-specific (they depend
// on gamma), so they're kept alongside the raw Episode rather than as a
// field on rl.Transition itself.
type scoredEpisode struct {
	Episode *rl.Episode
	Returns []float32
}

// SampleAction draws an action from the categorical distribution in
// probs by sampling a uniform value and walking the cumulative
// distribution function, returning the first action whose cumulative
// probability meets or exceeds the sample. A rounding-error fallback
// returns the last action if none did (mirroring sample_action in the
// original env.c).
func SampleAction(probs *mat.Matrix, rng *rand.Rand) rl.Action {
	sample := rng.Float32()

	var cumulative float32
	size := len(probs.Data)
	for i, p := range probs.Data {
		cumulative += p
		if sample <= cumulative {
			return rl.Action(i)
		}
	}
	return rl.Action(size - 1)
}

// computeReturns computes the discounted reward-to-go at every step of
// episode: returns[t] = Rewards[t] + gamma*returns[t+1]. See
// docs/03-policy-gradients-and-reinforce.md for what "reward-to-go"
// means and why it's computed backward through the episode.
func computeReturns(episode *rl.Episode, gamma float32) []float32 {
	returns := make([]float32, len(episode.Transitions))

	var g float32
	for t := len(episode.Transitions) - 1; t >= 0; t-- {
		g = episode.Transitions[t].Reward + gamma*g
		returns[t] = g
	}
	return returns
}
