package reinforce

import (
	"context"
	"errors"
	"math/rand/v2"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/config"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// cancelAwareEnv is a minimal rl.Environment whose Step reports ctx's
// error (if any) instead of ignoring ctx like the toy environments do,
// used to confirm context cancellation actually propagates through the
// rollout-collection path rather than being silently ignored.
type cancelAwareEnv struct{}

func (cancelAwareEnv) Reset(ctx context.Context) (rl.Observation, error) {
	return rl.Observation{Values: []float32{0, 0}}, ctx.Err()
}

func (cancelAwareEnv) Step(ctx context.Context, action rl.Action) (rl.StepResult, error) {
	if err := ctx.Err(); err != nil {
		return rl.StepResult{}, err
	}
	return rl.StepResult{Observation: rl.Observation{Values: []float32{0, 0}}, Reward: 0, Done: false}, nil
}

func (cancelAwareEnv) ObservationSize() int { return 2 }
func (cancelAwareEnv) ActionSpace() int     { return 2 }

func persistentTestSettings() config.Settings {
	settings := config.Default()
	settings.Epochs = 2
	settings.RolloutSize = 3
	settings.EpisodeLen = 4
	settings.Gamma = 0.99
	settings.LearningRate = 0.05
	settings.HiddenSize = 4
	settings.Seed = 1
	return settings
}

// TestNewWithPersistentEnvPropagatesContextCancellation confirms
// RunEpoch surfaces a canceled context's error through the sequential
// rollout path rather than swallowing or ignoring it.
func TestNewWithPersistentEnvPropagatesContextCancellation(t *testing.T) {
	settings := persistentTestSettings()

	factory := func(rng *rand.Rand) (rl.Environment, error) {
		return cancelAwareEnv{}, nil
	}

	trainer, err := NewWithPersistentEnv(settings, factory, nil)
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = trainer.RunEpoch(ctx, 0)
	require.Error(t, err)
	assert.True(t, errors.Is(err, context.Canceled), "expected error to wrap context.Canceled, got: %v", err)
}

// countingEnv counts how many times Reset was called (via a shared
// pointer), and ends every episode after exactly one step, so the
// number of episodes collected across epochs is precisely predictable.
type countingEnv struct {
	resetCount *int
}

func (e countingEnv) Reset(ctx context.Context) (rl.Observation, error) {
	*e.resetCount++
	return rl.Observation{Values: []float32{0, 0}}, nil
}

func (countingEnv) Step(ctx context.Context, action rl.Action) (rl.StepResult, error) {
	return rl.StepResult{Observation: rl.Observation{Values: []float32{0, 0}}, Reward: 1, Done: true}, nil
}

func (countingEnv) ObservationSize() int { return 2 }
func (countingEnv) ActionSpace() int     { return 2 }

// TestNewWithPersistentEnvConstructsEnvironmentOnce confirms
// NewWithPersistentEnv builds its environment exactly once, and every
// subsequent episode (across multiple RunEpoch calls, not just multiple
// episodes within one call) reuses that same instance via Reset,
// distinguishing this from EnvFactory's per-episode construction.
func TestNewWithPersistentEnvConstructsEnvironmentOnce(t *testing.T) {
	settings := persistentTestSettings()

	constructionCount := 0
	resetCount := 0
	factory := func(rng *rand.Rand) (rl.Environment, error) {
		constructionCount++
		return countingEnv{resetCount: &resetCount}, nil
	}

	trainer, err := NewWithPersistentEnv(settings, factory, nil)
	require.NoError(t, err)

	for epoch := range settings.Epochs {
		_, err := trainer.RunEpoch(context.Background(), epoch)
		require.NoError(t, err)
	}

	assert.Equal(t, 1, constructionCount, "the persistent environment must be constructed exactly once")
	assert.Equal(t, settings.Epochs*settings.RolloutSize, resetCount,
		"Reset should be called once per episode across every epoch")
}
