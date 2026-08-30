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
	"fmt"
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

// SampleMaskedAction draws an action from the categorical distribution
// in probs, exactly like SampleAction, except mask (when non-nil)
// excludes disallowed actions from consideration: probability mass is
// renormalized over only the allowed entries before sampling. A nil
// mask, or a mask where every entry is true, delegates straight to
// SampleAction, so it reproduces SampleAction's output bit-for-bit for
// the same rng draw rather than only approximately matching it through
// a renormalization that happens to divide by (nearly) 1.
//
// mask, when non-nil, must have the same length as probs.Data (i.e.
// len(mask) == ActionSpace()). Returns an error if mask is the wrong
// length, or if it excludes every action with nonzero probability (no
// legal action to sample).
func SampleMaskedAction(probs *mat.Matrix, mask []bool, rng *rand.Rand) (rl.Action, error) {
	if mask == nil {
		return SampleAction(probs, rng), nil
	}
	if len(mask) != len(probs.Data) {
		return 0, fmt.Errorf("reinforce: mask length %d does not match action space %d", len(mask), len(probs.Data))
	}

	allAllowed := true
	var maskedSum float32
	for i, allowed := range mask {
		if allowed {
			maskedSum += probs.Data[i]
		} else {
			allAllowed = false
		}
	}
	if allAllowed {
		return SampleAction(probs, rng), nil
	}
	if maskedSum <= 0 {
		return 0, fmt.Errorf("reinforce: no legal action: mask excludes every action with nonzero probability")
	}

	sample := rng.Float32()

	var cumulative float32
	lastAllowed := -1
	for i, allowed := range mask {
		if !allowed {
			continue
		}
		lastAllowed = i
		cumulative += probs.Data[i] / maskedSum
		if sample <= cumulative {
			return rl.Action(i), nil
		}
	}
	return rl.Action(lastAllowed), nil
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
