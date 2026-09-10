package reinforce

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/policy"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// maskedEnv implements rl.ActionMasker, always reporting action index 1
// illegal, ending its episode after exactly 3 steps. Used to confirm a
// mask actually reaches rollout collection (never sampling the illegal
// action) and training (replaying without corrupting the gradient) — see
// docs/plans/19-training-time-action-masking.md.
type maskedEnv struct {
	steps int
}

func (e *maskedEnv) Reset(ctx context.Context) (rl.Observation, error) {
	e.steps = 0
	return rl.Observation{Values: []float32{0, 0}}, nil
}

func (e *maskedEnv) Step(ctx context.Context, action rl.Action) (rl.StepResult, error) {
	e.steps++
	return rl.StepResult{
		Observation: rl.Observation{Values: []float32{0, 0}},
		Reward:      1,
		Done:        e.steps >= 3,
	}, nil
}

func (*maskedEnv) ObservationSize() int { return 2 }
func (*maskedEnv) ActionSpace() int     { return 3 }
func (*maskedEnv) ActionMask() []bool   { return []bool{true, false, true} }

// TestActionMaskExcludesIllegalActionAndIsRecorded confirms
// collectTrajectoryFromEnv never samples an env.(rl.ActionMasker)'s
// illegal action, and records the mask that was in effect on every
// resulting Transition (not just the ones where masking mattered).
func TestActionMaskExcludesIllegalActionAndIsRecorded(t *testing.T) {
	env := &maskedEnv{}
	rng := rand.New(rand.NewPCG(1, 2))
	params := policy.NewParams(rng, env.ObservationSize(), 4, env.ActionSpace())

	episode, err := collectTrajectoryFromEnv(context.Background(), params, env, 20, rng)
	require.NoError(t, err)
	require.NotEmpty(t, episode.Transitions)

	for _, transition := range episode.Transitions {
		assert.NotEqual(t, rl.Action(1), transition.Action, "masked action must never be sampled")
		assert.Equal(t, []bool{true, false, true}, transition.Mask)
	}
}

// TestActionMaskWiredThroughTrainingWithoutNaN confirms a full RunEpoch
// (rollout collection through the replay-and-backward pass in
// trainOnRollouts) against an ActionMasker environment completes cleanly
// and produces a real, non-NaN gradient, exercising SetActionMask on the
// training-network side of the wiring, not just the sampling side.
func TestActionMaskWiredThroughTrainingWithoutNaN(t *testing.T) {
	settings := persistentTestSettings()
	settings.RolloutSize = 4
	settings.EpisodeLen = 3

	factory := func(rng *rand.Rand) (rl.Environment, error) {
		return &maskedEnv{}, nil
	}

	trainer, err := NewWithPersistentEnv(settings, factory, nil)
	require.NoError(t, err)

	stats, err := trainer.RunEpoch(context.Background(), 0)
	require.NoError(t, err)
	assert.False(t, math.IsNaN(float64(stats.AverageReturn)))
	assert.Greater(t, trainer.GradientNorm(), float32(0), "training against a masked environment must still produce a real gradient")
}
