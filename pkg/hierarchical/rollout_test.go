package hierarchical

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/reinforce"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// scriptedRewardEnv reports a reward equal to its step count (1, 2, 3,
// ...), regardless of the action taken, and never reports Done —
// letting a test predict the exact reward sequence a trajectory will
// see without needing to control which action gets sampled.
type scriptedRewardEnv struct {
	step int
}

func (e *scriptedRewardEnv) Reset(ctx context.Context) (rl.Observation, error) {
	e.step = 0
	return rl.Observation{Values: []float32{0, 0}}, nil
}

func (e *scriptedRewardEnv) Step(ctx context.Context, action rl.Action) (rl.StepResult, error) {
	e.step++
	return rl.StepResult{Observation: rl.Observation{Values: []float32{0, 0}}, Reward: float32(e.step), Done: false}, nil
}

func (e *scriptedRewardEnv) ObservationSize() int { return 2 }
func (e *scriptedRewardEnv) ActionSpace() int     { return 2 }

func testHierarchicalParams(rng *rand.Rand, cfg Config, obsSize, actionSpace int) (*actorcritic.Params, map[Subgoal]*actorcritic.Params) {
	metaParams := actorcritic.NewParams(rng, obsSize, cfg.MetaHiddenSize, cfg.NumSubgoals)
	subParams := make(map[Subgoal]*actorcritic.Params, cfg.NumSubgoals)
	for i := range cfg.NumSubgoals {
		subParams[Subgoal(i)] = actorcritic.NewParams(rng, obsSize+cfg.NumSubgoals, cfg.SubHiddenSize, actionSpace)
	}
	return metaParams, subParams
}

// TestCollectHierarchicalTrajectoryAccruesMetaRewardAcrossInterval
// confirms a meta-transition's reward is the sum of every per-step
// environment reward since the previous meta-decision, across a
// SubgoalInterval boundary, and that the final, partial interval left
// over when episodeLen is exhausted is still recorded (not dropped).
func TestCollectHierarchicalTrajectoryAccruesMetaRewardAcrossInterval(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	cfg := Config{NumSubgoals: 2, SubgoalInterval: 3, MetaHiddenSize: 4, SubHiddenSize: 4, MetaLearningRate: 0.01, SubLearningRate: 0.01}
	metaParams, subParams := testHierarchicalParams(rng, cfg, 2, 2)

	envFactory := reinforce.EnvFactory(func(rng *rand.Rand) (rl.Environment, error) {
		return &scriptedRewardEnv{}, nil
	})

	rollout, err := collectHierarchicalTrajectory(context.Background(), metaParams, subParams, cfg, envFactory, 7, rng)
	require.NoError(t, err)

	require.Len(t, rollout.MetaRollout.Episode.Transitions, 3)
	assert.Equal(t, float32(1+2+3), rollout.MetaRollout.Episode.Transitions[0].Reward)
	assert.Equal(t, float32(4+5+6), rollout.MetaRollout.Episode.Transitions[1].Reward)
	assert.Equal(t, float32(7), rollout.MetaRollout.Episode.Transitions[2].Reward, "the final partial interval (episodeLen exhausted before a full SubgoalInterval) must still be recorded")

	for i, transition := range rollout.MetaRollout.Episode.Transitions {
		assert.False(t, transition.Done, "transition %d: episodeLen exhaustion is not an environment Done", i)
	}

	totalSubSteps := 0
	for _, segments := range rollout.SubRollouts {
		for _, segment := range segments {
			totalSubSteps += len(segment.Episode.Transitions)
		}
	}
	assert.Equal(t, 7, totalSubSteps, "every step must be recorded in exactly one sub-policy segment")
}

// doneAfterStepEnv reports Done once its step count reaches doneAt,
// before a full SubgoalInterval necessarily elapses.
type doneAfterStepEnv struct {
	step, doneAt int
}

func (e *doneAfterStepEnv) Reset(ctx context.Context) (rl.Observation, error) {
	e.step = 0
	return rl.Observation{Values: []float32{0, 0}}, nil
}

func (e *doneAfterStepEnv) Step(ctx context.Context, action rl.Action) (rl.StepResult, error) {
	e.step++
	done := e.step >= e.doneAt
	return rl.StepResult{Observation: rl.Observation{Values: []float32{0, 0}}, Reward: float32(e.step), Done: done}, nil
}

func (e *doneAfterStepEnv) ObservationSize() int { return 2 }
func (e *doneAfterStepEnv) ActionSpace() int     { return 2 }

// TestCollectHierarchicalTrajectoryClosesIntervalOnEnvDone confirms an
// environment-reported Done closes the in-progress interval
// immediately (rather than waiting for the next SubgoalInterval
// boundary), and is recorded with Done=true.
func TestCollectHierarchicalTrajectoryClosesIntervalOnEnvDone(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	cfg := Config{NumSubgoals: 2, SubgoalInterval: 5, MetaHiddenSize: 4, SubHiddenSize: 4, MetaLearningRate: 0.01, SubLearningRate: 0.01}
	metaParams, subParams := testHierarchicalParams(rng, cfg, 2, 2)

	envFactory := reinforce.EnvFactory(func(rng *rand.Rand) (rl.Environment, error) {
		return &doneAfterStepEnv{doneAt: 2}, nil
	})

	rollout, err := collectHierarchicalTrajectory(context.Background(), metaParams, subParams, cfg, envFactory, 10, rng)
	require.NoError(t, err)

	require.Len(t, rollout.MetaRollout.Episode.Transitions, 1)
	assert.Equal(t, float32(1+2), rollout.MetaRollout.Episode.Transitions[0].Reward)
	assert.True(t, rollout.MetaRollout.Episode.Transitions[0].Done)
}

// TestCollectHierarchicalTrajectoryRecordsMetaProbabilities confirms
// MetaProbabilities is populated with one full, valid subgoal
// distribution per meta-decision — including the final, partial
// interval left over when episodeLen is exhausted (rollout.go's
// closeInterval call outside the main loop, the easiest one to
// accidentally miss) — and that each recorded distribution agrees with
// the already-trusted scalar LogProbs it was sampled alongside, rather
// than only checking shapes. See
// docs/plans/17-meta-decision-distribution-recording.md.
func TestCollectHierarchicalTrajectoryRecordsMetaProbabilities(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	cfg := Config{NumSubgoals: 3, SubgoalInterval: 3, MetaHiddenSize: 4, SubHiddenSize: 4, MetaLearningRate: 0.01, SubLearningRate: 0.01}
	metaParams, subParams := testHierarchicalParams(rng, cfg, 2, 2)

	envFactory := reinforce.EnvFactory(func(rng *rand.Rand) (rl.Environment, error) {
		return &scriptedRewardEnv{}, nil
	})

	rollout, err := collectHierarchicalTrajectory(context.Background(), metaParams, subParams, cfg, envFactory, 7, rng)
	require.NoError(t, err)

	require.Len(t, rollout.MetaRollout.Episode.Transitions, 3, "one entry per meta-decision, including the final partial interval")
	require.Len(t, rollout.MetaProbabilities, len(rollout.MetaRollout.Episode.Transitions))

	for i, probabilities := range rollout.MetaProbabilities {
		require.Len(t, probabilities, cfg.NumSubgoals, "meta-decision %d", i)

		var sum float32
		for _, p := range probabilities {
			sum += p
		}
		assert.InDelta(t, 1.0, sum, 1e-4, "meta-decision %d: distribution must sum to 1", i)

		sampledAction := rollout.MetaRollout.Episode.Transitions[i].Action
		wantProbability := float32(math.Exp(float64(rollout.MetaRollout.LogProbs[i])))
		assert.InDelta(t, wantProbability, probabilities[sampledAction], 1e-4,
			"meta-decision %d: recorded distribution at the sampled action must agree with the existing scalar LogProb", i)
	}
}
