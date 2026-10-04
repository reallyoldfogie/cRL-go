package ppo

import (
	"context"
	"errors"
	"math/rand/v2"
	"sync/atomic"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/config"
	"github.com/reallyoldfogie/cRL-go/pkg/reinforce"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// recurrentPersistentTestSettings mirrors persistentTestSettings (see
// persistent_env_test.go) with recurrence turned on.
func recurrentPersistentTestSettings() config.Settings {
	settings := persistentTestSettings()
	settings.RecurrentChunkLen = 2
	return settings
}

// TestNewRecurrentWithPersistentEnvConstructsEnvironmentOnce mirrors
// TestNewWithPersistentEnvConstructsEnvironmentOnce exactly, confirming
// the recurrent persistent-env path reuses one environment instance
// across every episode of every epoch rather than rebuilding it.
func TestNewRecurrentWithPersistentEnvConstructsEnvironmentOnce(t *testing.T) {
	settings := recurrentPersistentTestSettings()

	constructionCount := 0
	resetCount := 0
	factory := func(rng *rand.Rand) (rl.Environment, error) {
		constructionCount++
		return countingEnv{resetCount: &resetCount}, nil
	}

	trainer, err := NewRecurrentWithPersistentEnv(settings, reinforce.PersistentEnvFactory(factory), nil)
	require.NoError(t, err)

	for epoch := range settings.Epochs {
		_, err := trainer.RunEpoch(context.Background(), epoch)
		require.NoError(t, err)
	}

	assert.Equal(t, 1, constructionCount, "the persistent environment must be constructed exactly once")
	assert.Equal(t, settings.Epochs*settings.RolloutSize, resetCount,
		"Reset should be called once per episode across every epoch")
}

// TestNewRecurrentWithPersistentEnvPropagatesContextCancellation mirrors
// TestNewWithPersistentEnvPropagatesContextCancellation.
func TestNewRecurrentWithPersistentEnvPropagatesContextCancellation(t *testing.T) {
	settings := recurrentPersistentTestSettings()

	factory := func(rng *rand.Rand) (rl.Environment, error) {
		return cancelAwareEnv{}, nil
	}

	trainer, err := NewRecurrentWithPersistentEnv(settings, reinforce.PersistentEnvFactory(factory), nil)
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = trainer.RunEpoch(ctx, 0)
	require.Error(t, err)
	assert.True(t, errors.Is(err, context.Canceled), "expected error to wrap context.Canceled, got: %v", err)
}

func TestNewRecurrentWithPersistentEnvPoolRejectsEmptyEnvs(t *testing.T) {
	settings := recurrentPersistentTestSettings()

	_, err := NewRecurrentWithPersistentEnvPool(settings, nil, nil)
	require.Error(t, err)
}

// TestRecurrentCollectRolloutsPoolDistributesAcrossAllEnvs mirrors
// TestCollectRolloutsPoolDistributesAcrossAllEnvs: RolloutSize episodes
// must be striped across every pooled env even when it doesn't divide
// evenly by the pool size, and no two goroutines ever touch the same
// pooled env concurrently.
func TestRecurrentCollectRolloutsPoolDistributesAcrossAllEnvs(t *testing.T) {
	settings := recurrentPersistentTestSettings()
	settings.RolloutSize = 7

	const n = 3
	envs := make([]rl.Environment, n)
	tracked := make([]*poolTrackingEnv, n)
	for i := range n {
		tracked[i] = newPoolTrackingEnv()
		envs[i] = tracked[i]
	}

	trainer, err := NewRecurrentWithPersistentEnvPool(settings, envs, nil)
	require.NoError(t, err)

	_, err = trainer.RunEpoch(context.Background(), 0)
	require.NoError(t, err)

	var totalResets int32
	for _, tr := range tracked {
		assert.LessOrEqual(t, atomic.LoadInt32(tr.maxSeen), int32(1), "no pooled env may be touched by two goroutines at once")
		totalResets += atomic.LoadInt32(tr.resetCount)
	}
	assert.EqualValues(t, settings.RolloutSize, totalResets, "every episode in RolloutSize must land on exactly one pooled env")
}

// TestNewRecurrentWithPersistentEnvPoolSurfacesPerEnvError mirrors
// TestNewWithPersistentEnvPoolSurfacesPerEnvError.
func TestNewRecurrentWithPersistentEnvPoolSurfacesPerEnvError(t *testing.T) {
	settings := recurrentPersistentTestSettings()
	envs := []rl.Environment{erroringEnv{}}

	trainer, err := NewRecurrentWithPersistentEnvPool(settings, envs, nil)
	require.NoError(t, err)

	_, err = trainer.RunEpoch(context.Background(), 0)
	assert.Error(t, err)
}
