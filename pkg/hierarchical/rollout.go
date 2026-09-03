package hierarchical

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/mat"
	"github.com/reallyoldfogie/cRL-go/pkg/ppo"
	"github.com/reallyoldfogie/cRL-go/pkg/reinforce"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

// logProbEpsilon mirrors pkg/ppo's identically-named constant.
const logProbEpsilon float32 = 1e-8

// HierarchicalRollout is one episode's worth of collected experience,
// split into the meta-controller's stream (one transition per
// subgoal-decision interval) and, per subgoal, one *ppo.Rollout per
// contiguous interval that subgoal was active.
//
// Sub-policy activity within a single episode is not always
// contiguous: the meta-controller can reactivate the same subgoal
// again later, with a different subgoal active in between. Each
// activation interval is kept as its own *ppo.Rollout — rather than one
// concatenated Rollout per subgoal — specifically so ppo.ComputeGAE
// bootstraps correctly at the end of each interval, instead of treating
// two chronologically-disjoint intervals as one continuous trajectory
// (which would bootstrap an interval's last step from an unrelated
// interval's first value estimate).
type HierarchicalRollout struct {
	MetaRollout *ppo.Rollout
	SubRollouts map[Subgoal][]*ppo.Rollout
	// MetaProbabilities is the meta-controller's full, length-NumSubgoals
	// probability distribution at each meta-decision, index-aligned with
	// MetaRollout.Episode.Transitions/LogProbs/Values (i.e.
	// MetaProbabilities[i] is the distribution
	// MetaRollout.Episode.Transitions[i].Action was sampled from). Kept
	// here rather than on *ppo.Rollout itself since that type is shared
	// with flat, non-hierarchical PPO, which has no equivalent use for
	// it. See docs/plans/17-meta-decision-distribution-recording.md.
	MetaProbabilities [][]float32
}

// selectFromNetwork runs one forward pass over obs through net and
// samples a category from its policy head — whether that category
// represents a Subgoal (the meta-controller) or a primitive rl.Action
// (a sub-policy) is entirely up to the caller, since both are scored
// identically. Returns the sampled index, its log-probability, the
// value head's estimate at obs, and the full distribution it was
// sampled from (an independent copy, safe to keep past the next
// Graph.Forward() call on net).
func selectFromNetwork(net *actorcritic.InferenceNetwork, obs rl.Observation, rng *rand.Rand) (index int, logProb, value float32, probabilities []float32, err error) {
	copy(net.Input.Val.Data, obs.Values)
	net.Graph.Forward()

	action, probabilities, _, err := reinforce.SampleMaskedActionWithProbabilities(net.PolicyOutput.Val, nil, rng)
	if err != nil {
		return 0, 0, 0, nil, err
	}
	return int(action), actionLogProb(net.PolicyOutput.Val, action), net.ValueOutput.Val.Data[0], probabilities, nil
}

// actionLogProb mirrors pkg/ppo's identically-named, unexported
// function (duplicated here for the same reason pkg/actorcritic's
// chain type is duplicated from pkg/policy's: no dependency on another
// package's unexported internals).
func actionLogProb(probs *mat.Matrix, action rl.Action) float32 {
	probability := probs.Data[action]
	if probability < logProbEpsilon {
		probability = logProbEpsilon
	}
	return float32(math.Log(float64(probability)))
}

// selectSubgoal runs one forward pass over obs through the
// meta-controller network and samples a Subgoal, additionally returning
// the full subgoal-probability distribution it was sampled from (see
// HierarchicalRollout.MetaProbabilities).
func selectSubgoal(net *actorcritic.InferenceNetwork, obs rl.Observation, rng *rand.Rand) (subgoal Subgoal, logProb, value float32, probabilities []float32, err error) {
	index, logProb, value, probabilities, err := selectFromNetwork(net, obs, rng)
	if err != nil {
		return 0, 0, 0, nil, fmt.Errorf("hierarchical: selecting subgoal: %w", err)
	}
	return Subgoal(index), logProb, value, probabilities, nil
}

// collectHierarchicalTrajectory runs one episode (up to episodeLen
// steps, ending early if the environment reports Done) against an
// environment built by envFactory, interleaving meta-controller
// decisions (every cfg.SubgoalInterval steps, or at episode start) with
// the currently-active sub-policy's primitive-action decisions. rng
// drives envFactory and every sampling decision.
func collectHierarchicalTrajectory(
	ctx context.Context,
	metaParams *actorcritic.Params,
	subParams map[Subgoal]*actorcritic.Params,
	cfg Config,
	envFactory reinforce.EnvFactory,
	episodeLen int,
	rng *rand.Rand,
) (*HierarchicalRollout, error) {
	env, err := envFactory(rng)
	if err != nil {
		return nil, fmt.Errorf("hierarchical: creating environment: %w", err)
	}

	metaNet, err := actorcritic.NewInferenceNetwork(metaParams)
	if err != nil {
		return nil, fmt.Errorf("hierarchical: building meta-controller inference network: %w", err)
	}

	subNets := make(map[Subgoal]*actorcritic.InferenceNetwork, cfg.NumSubgoals)
	for i := range cfg.NumSubgoals {
		subgoal := Subgoal(i)
		net, err := actorcritic.NewInferenceNetwork(subParams[subgoal])
		if err != nil {
			return nil, fmt.Errorf("hierarchical: building sub-policy inference network for subgoal %d: %w", subgoal, err)
		}
		subNets[subgoal] = net
	}

	observation, err := env.Reset(ctx)
	if err != nil {
		return nil, fmt.Errorf("hierarchical: resetting environment: %w", err)
	}

	metaEpisode := &rl.Episode{}
	var metaLogProbs, metaValues []float32
	var metaProbabilities [][]float32
	subRollouts := make(map[Subgoal][]*ppo.Rollout, cfg.NumSubgoals)

	// Interval state: which subgoal is active, the meta-decision that
	// activated it (needed to record the meta-transition once this
	// interval closes), and the sub-level transitions collected during
	// it (needed to record its own *ppo.Rollout segment once it
	// closes). These are function-level variables, reassigned (never
	// redeclared) as each interval closes and the next one begins, so
	// closeInterval's closure always sees the latest state.
	activeSubgoal, metaLogProb, metaValue, metaProbability, err := selectSubgoal(metaNet, observation, rng)
	if err != nil {
		return nil, err
	}
	intervalObservation := observation
	segmentEpisode := &rl.Episode{}
	var segmentLogProbs, segmentValues []float32
	var intervalReward float32
	stepsSinceDecision := 0

	closeInterval := func(done bool) {
		metaEpisode.Transitions = append(metaEpisode.Transitions, rl.Transition{
			Observation: intervalObservation,
			Action:      rl.Action(activeSubgoal),
			Reward:      intervalReward,
			Done:        done,
		})
		metaLogProbs = append(metaLogProbs, metaLogProb)
		metaValues = append(metaValues, metaValue)
		metaProbabilities = append(metaProbabilities, metaProbability)

		subRollouts[activeSubgoal] = append(subRollouts[activeSubgoal], &ppo.Rollout{
			Episode:  segmentEpisode,
			LogProbs: segmentLogProbs,
			Values:   segmentValues,
		})
	}

	for range episodeLen {
		augmented := augmentObservation(observation, activeSubgoal, cfg.NumSubgoals)
		actionIndex, subLogProb, subValue, _, err := selectFromNetwork(subNets[activeSubgoal], augmented, rng)
		if err != nil {
			return nil, fmt.Errorf("hierarchical: selecting action: %w", err)
		}
		action := rl.Action(actionIndex)

		result, err := env.Step(ctx, action)
		if err != nil {
			return nil, fmt.Errorf("hierarchical: stepping environment: %w", err)
		}

		segmentEpisode.Transitions = append(segmentEpisode.Transitions, rl.Transition{
			Observation: augmented,
			Action:      action,
			Reward:      result.Reward,
			Done:        result.Done,
		})
		segmentLogProbs = append(segmentLogProbs, subLogProb)
		segmentValues = append(segmentValues, subValue)

		intervalReward += result.Reward
		stepsSinceDecision++
		observation = result.Observation

		if result.Done {
			closeInterval(true)
			stepsSinceDecision = 0
			break
		}

		if stepsSinceDecision >= cfg.SubgoalInterval {
			closeInterval(false)

			intervalObservation = observation
			activeSubgoal, metaLogProb, metaValue, metaProbability, err = selectSubgoal(metaNet, intervalObservation, rng)
			if err != nil {
				return nil, err
			}
			segmentEpisode = &rl.Episode{}
			segmentLogProbs = nil
			segmentValues = nil
			intervalReward = 0
			stepsSinceDecision = 0
		}
	}

	// The loop above only closes an interval when the environment
	// reports Done or a full SubgoalInterval elapses; if episodeLen is
	// exhausted first, one more still-open, non-terminal interval needs
	// closing so its steps aren't silently dropped.
	if stepsSinceDecision > 0 {
		closeInterval(false)
	}

	return &HierarchicalRollout{
		MetaRollout:       &ppo.Rollout{Episode: metaEpisode, LogProbs: metaLogProbs, Values: metaValues},
		SubRollouts:       subRollouts,
		MetaProbabilities: metaProbabilities,
	}, nil
}
