package hierarchical

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/config"
	"github.com/reallyoldfogie/cRL-go/pkg/hierarchicalgridworld"
	"github.com/reallyoldfogie/cRL-go/pkg/reinforce"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func smallTestSettings() config.Settings {
	settings := config.Default()
	settings.Epochs = 3
	settings.RolloutSize = 4
	settings.EpisodeLen = 12
	settings.Gamma = 0.99
	settings.GridSize = 16
	settings.Seed = 1
	settings.Workers = 2
	settings.ClipEpsilon = 0.2
	settings.EntropyCoef = 0.01
	settings.ValueCoef = 0.5
	settings.GAELambda = 0.95
	settings.PPOEpochs = 2
	settings.MinibatchSize = 8
	return settings
}

func smallTestConfig() Config {
	return Config{
		NumSubgoals:      3,
		SubgoalInterval:  4,
		MetaHiddenSize:   8,
		SubHiddenSize:    8,
		MetaLearningRate: 0.01,
		SubLearningRate:  0.01,
	}
}

func hierarchicalGridWorldFactory(gridSize int) reinforce.EnvFactory {
	return func(rng *rand.Rand) (rl.Environment, error) {
		env, err := hierarchicalgridworld.New(gridSize, rng)
		if err != nil {
			return nil, err
		}
		return hierarchicalgridworld.NewAdapter(env), nil
	}
}

func TestRunEpochSmokeTestNoPanicsOrNaNs(t *testing.T) {
	settings := smallTestSettings()
	cfg := smallTestConfig()

	trainer, err := New(settings, cfg, hierarchicalGridWorldFactory(settings.GridSize), nil)
	require.NoError(t, err)

	for epoch := range settings.Epochs {
		stats, err := trainer.RunEpoch(context.Background(), epoch)
		require.NoError(t, err)

		assert.False(t, math.IsNaN(float64(stats.AverageReturn)), "epoch %d: average return is NaN", epoch)
		assert.GreaterOrEqual(t, stats.SampleCount, 0)
		assert.Equal(t, epoch, stats.Epoch)
	}
}

func TestRunEpochIsDeterministicForAFixedSeed(t *testing.T) {
	settings := smallTestSettings()
	cfg := smallTestConfig()

	trainerA, err := New(settings, cfg, hierarchicalGridWorldFactory(settings.GridSize), nil)
	require.NoError(t, err)
	statsA, err := trainerA.RunEpoch(context.Background(), 0)
	require.NoError(t, err)

	trainerB, err := New(settings, cfg, hierarchicalGridWorldFactory(settings.GridSize), nil)
	require.NoError(t, err)
	statsB, err := trainerB.RunEpoch(context.Background(), 0)
	require.NoError(t, err)

	// A fixed seed must reproduce identical results regardless of
	// goroutine scheduling order (see reinforce.WorkerRNG's
	// documentation), including every subgoal's minibatch shuffle order.
	assert.Equal(t, statsA, statsB)
}

func TestNewRejectsInvalidSettings(t *testing.T) {
	settings := smallTestSettings()
	settings.GridSize = 10 // not a perfect square
	cfg := smallTestConfig()

	_, err := New(settings, cfg, hierarchicalGridWorldFactory(settings.GridSize), nil)
	assert.Error(t, err)
}

func TestNewRejectsInvalidConfig(t *testing.T) {
	settings := smallTestSettings()
	cfg := smallTestConfig()
	cfg.NumSubgoals = 0

	_, err := New(settings, cfg, hierarchicalGridWorldFactory(settings.GridSize), nil)
	assert.Error(t, err)
}

// TestTrainingImprovesAverageReturn is a slower, qualitative "does it
// visibly learn" check, mirroring pkg/ppo's test of the same name: it
// runs enough epochs against pkg/hierarchicalgridworld's genuinely
// competing objectives that the meta-controller's average return
// should trend upward. Skipped under `go test -short`.
func TestTrainingImprovesAverageReturn(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping slow training-trend test in -short mode")
	}

	settings := config.Default()
	settings.Epochs = 200
	settings.RolloutSize = 32
	settings.EpisodeLen = 20
	settings.Gamma = 0.95
	settings.GridSize = 16
	settings.Seed = 42
	settings.Workers = 8
	settings.ClipEpsilon = 0.2
	settings.EntropyCoef = 0.02
	settings.ValueCoef = 0.5
	settings.GAELambda = 0.95
	settings.PPOEpochs = 4
	settings.MinibatchSize = 32

	cfg := Config{
		NumSubgoals:      3,
		SubgoalInterval:  5,
		MetaHiddenSize:   16,
		SubHiddenSize:    16,
		MetaLearningRate: 0.003,
		SubLearningRate:  0.003,
	}

	trainer, err := New(settings, cfg, hierarchicalGridWorldFactory(settings.GridSize), nil)
	require.NoError(t, err)

	const earlyWindow = 20
	var earlySum, lateSum float32

	for epoch := range settings.Epochs {
		stats, err := trainer.RunEpoch(context.Background(), epoch)
		require.NoError(t, err)

		if epoch < earlyWindow {
			earlySum += stats.AverageReturn
		}
		if epoch >= settings.Epochs-earlyWindow {
			lateSum += stats.AverageReturn
		}
	}

	earlyAverage := earlySum / earlyWindow
	lateAverage := lateSum / earlyWindow

	t.Logf("average return: early=%.3f late=%.3f", earlyAverage, lateAverage)
	assert.Greater(t, lateAverage, earlyAverage,
		"average return should improve from the first %d epochs to the last %d", earlyWindow, earlyWindow)
}
