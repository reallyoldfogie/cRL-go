package hierarchical

import "github.com/reallyoldfogie/cRL-go/pkg/rl"

// augmentObservation returns obs with subgoal's one-hot encoding
// appended, mirroring mc-rl-go's AugmentObsWithSubgoal: a sub-policy
// needs to know which subgoal is currently active, since the same base
// observation can call for a different primitive action depending on
// it. The returned Observation's Values is a newly allocated slice;
// obs.Values is never mutated.
func augmentObservation(obs rl.Observation, subgoal Subgoal, numSubgoals int) rl.Observation {
	augmented := make([]float32, len(obs.Values)+numSubgoals)
	copy(augmented, obs.Values)
	if index := int(subgoal); index >= 0 && index < numSubgoals {
		augmented[len(obs.Values)+index] = 1
	}
	return rl.Observation{Values: augmented}
}
