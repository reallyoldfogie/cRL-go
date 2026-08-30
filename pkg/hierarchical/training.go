package hierarchical

import (
	"math"

	"github.com/reallyoldfogie/cRL-go/pkg/ppo"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

// trainingStep is one flattened (observation, action, ...) tuple ready
// for a minibatch Adam update, mirroring pkg/ppo's identically-named,
// unexported type (duplicated here since pkg/ppo's version is tied to
// its own private Trainer fields, not reusable across packages).
type trainingStep struct {
	Observation rl.Observation
	Action      rl.Action
	OldLogProb  float32
	Advantage   float32
	Return      float32
}

// scoredRollout pairs a collected *ppo.Rollout with its GAE advantages
// and value-target returns, mirroring pkg/ppo's identically-named type.
type scoredRollout struct {
	Rollout    *ppo.Rollout
	Advantages []float32
	Returns    []float32
}

// scoreRollouts computes GAE(λ) advantages and value-target returns for
// every rollout independently, via ppo.ComputeGAE. Each rollout is
// treated as its own bootstrap horizon: for sub-policy rollouts, that's
// exactly what makes splitting a subgoal's activity into one
// *ppo.Rollout per contiguous interval (see HierarchicalRollout)
// correct rather than merely convenient — GAE never bootstraps across a
// gap where a different subgoal was active.
func scoreRollouts(rollouts []*ppo.Rollout, gamma, lambda float32) []scoredRollout {
	scored := make([]scoredRollout, len(rollouts))
	for i, rollout := range rollouts {
		advantages, returns := ppo.ComputeGAE(rollout, gamma, lambda)
		scored[i] = scoredRollout{Rollout: rollout, Advantages: advantages, Returns: returns}
	}
	return scored
}

// flattenScoredRollouts gathers every step of every scoredRollout into
// one pool, mirroring pkg/ppo's identically-named function.
func flattenScoredRollouts(scored []scoredRollout) []trainingStep {
	var steps []trainingStep
	for _, s := range scored {
		for t, transition := range s.Rollout.Episode.Transitions {
			steps = append(steps, trainingStep{
				Observation: transition.Observation,
				Action:      transition.Action,
				OldLogProb:  s.Rollout.LogProbs[t],
				Advantage:   s.Advantages[t],
				Return:      s.Returns[t],
			})
		}
	}
	return steps
}

// normalizeAdvantages rescales every step's Advantage in place to
// batch-wide mean 0, standard deviation 1, mirroring pkg/ppo's
// identically-named function exactly (see its doc comment for why this
// is worthwhile on top of GAE's own formula).
func normalizeAdvantages(steps []trainingStep) {
	if len(steps) == 0 {
		return
	}

	const advantageEpsilon = 1e-8

	var sum, sumSquares float32
	for _, step := range steps {
		sum += step.Advantage
		sumSquares += step.Advantage * step.Advantage
	}

	mean := sum / float32(len(steps))
	variance := sumSquares/float32(len(steps)) - mean*mean
	std := float32(math.Sqrt(float64(max(variance, 0))))

	for i := range steps {
		steps[i].Advantage = (steps[i].Advantage - mean) / (std + advantageEpsilon)
	}
}
