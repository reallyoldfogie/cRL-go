package ppo

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/config"
	"github.com/reallyoldfogie/cRL-go/pkg/gridworldenv"
	"github.com/reallyoldfogie/cRL-go/pkg/reinforce"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/reallyoldfogie/cRL-go/pkg/snakeenv"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func smallTestSettings() config.Settings {
	return config.Settings{
		Epochs:        3,
		RolloutSize:   4,
		EpisodeLen:    5,
		Gamma:         0.99,
		LearningRate:  0.01,
		GridSize:      4, // 2x2 grid
		HiddenSize:    8,
		Seed:          1,
		Workers:       2,
		ClipEpsilon:   0.2,
		EntropyCoef:   0.01,
		ValueCoef:     0.5,
		GAELambda:     0.95,
		PPOEpochs:     2,
		MinibatchSize: 8,
	}
}

func gridWorldEnvFactory(gridSize int) reinforce.EnvFactory {
	return func(rng *rand.Rand) (rl.Environment, error) {
		env, err := gridworldenv.New(gridSize)
		if err != nil {
			return nil, err
		}
		return gridworldenv.NewAdapter(env), nil
	}
}

func TestRunEpochSmokeTestNoPanicsOrNaNs(t *testing.T) {
	settings := smallTestSettings()
	trainer, err := New(settings, snakeEnvFactory(settings.GridSize), nil)
	require.NoError(t, err)

	for epoch := range settings.Epochs {
		stats, err := trainer.RunEpoch(epoch)
		require.NoError(t, err)

		assert.False(t, math.IsNaN(float64(stats.AverageReturn)), "epoch %d: average return is NaN", epoch)
		assert.GreaterOrEqual(t, stats.SampleCount, 0)
		assert.Equal(t, epoch, stats.Epoch)
	}
}

func TestRunEpochIsDeterministicForAFixedSeed(t *testing.T) {
	settings := smallTestSettings()

	trainerA, err := New(settings, snakeEnvFactory(settings.GridSize), nil)
	require.NoError(t, err)
	statsA, err := trainerA.RunEpoch(0)
	require.NoError(t, err)

	trainerB, err := New(settings, snakeEnvFactory(settings.GridSize), nil)
	require.NoError(t, err)
	statsB, err := trainerB.RunEpoch(0)
	require.NoError(t, err)

	// A fixed seed must reproduce identical results regardless of
	// goroutine scheduling order (see reinforce.WorkerRNG's
	// documentation), including the minibatch shuffle order.
	assert.Equal(t, statsA, statsB)
}

func TestRunEpochRejectsInvalidSettings(t *testing.T) {
	settings := smallTestSettings()
	settings.GridSize = 10 // not a perfect square
	_, err := New(settings, snakeEnvFactory(settings.GridSize), nil)
	assert.Error(t, err)
}

// TestNewUsesProvidedInitialParams confirms that a non-nil initialParams
// is used as-is (not discarded in favor of a fresh initialization),
// which is what lets a checkpoint loaded via actorcritic.LoadFile
// actually resume training from its saved weights.
func TestNewUsesProvidedInitialParams(t *testing.T) {
	settings := smallTestSettings()

	rng := rand.New(rand.NewPCG(99, 99))
	initialParams := actorcritic.NewParams(rng, snakeenv.StateVectorSize(settings.GridSize), settings.HiddenSize, snakeenv.NumActions)

	trainer, err := New(settings, snakeEnvFactory(settings.GridSize), initialParams)
	require.NoError(t, err)

	assert.Same(t, initialParams, trainer.Params(), "New must use the provided initialParams rather than reinitializing")
}

// TestNewRejectsMismatchedInitialParams confirms a checkpoint saved for
// a different architecture is rejected rather than silently producing a
// broken network.
func TestNewRejectsMismatchedInitialParams(t *testing.T) {
	settings := smallTestSettings()

	rng := rand.New(rand.NewPCG(1, 1))
	mismatched := actorcritic.NewParams(rng, snakeenv.StateVectorSize(36), settings.HiddenSize, snakeenv.NumActions)

	_, err := New(settings, snakeEnvFactory(settings.GridSize), mismatched)
	assert.Error(t, err)
}

// TestRunEpochTrainsAgainstAnyEnvironment is the concrete proof that
// Trainer is genuinely environment-agnostic, exactly like
// pkg/reinforce.Trainer: gridworldenv has a different action count and
// observation layout than snakeenv, yet trains through the exact same
// Trainer/collectTrajectory code path with no pkg/ppo or pkg/actorcritic
// changes.
func TestRunEpochTrainsAgainstAnyEnvironment(t *testing.T) {
	settings := smallTestSettings()
	trainer, err := New(settings, gridWorldEnvFactory(settings.GridSize), nil)
	require.NoError(t, err)

	for epoch := range settings.Epochs {
		stats, err := trainer.RunEpoch(epoch)
		require.NoError(t, err)

		assert.False(t, math.IsNaN(float64(stats.AverageReturn)), "epoch %d: average return is NaN", epoch)
		assert.GreaterOrEqual(t, stats.SampleCount, 0)
	}
}

// TestTrainingImprovesAverageReturn is a slower, qualitative "does it
// visibly learn" check, mirroring pkg/reinforce's test of the same
// name: it runs enough epochs that the average return over the batch
// should trend upward. Skipped under `go test -short`.
func TestTrainingImprovesAverageReturn(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping slow training-trend test in -short mode")
	}

	settings := config.Settings{
		Epochs:        150,
		RolloutSize:   32,
		EpisodeLen:    10,
		Gamma:         0.9,
		LearningRate:  0.01,
		GridSize:      4,
		HiddenSize:    16,
		Seed:          42,
		Workers:       8,
		ClipEpsilon:   0.2,
		EntropyCoef:   0.01,
		ValueCoef:     0.5,
		GAELambda:     0.95,
		PPOEpochs:     4,
		MinibatchSize: 32,
	}

	trainer, err := New(settings, snakeEnvFactory(settings.GridSize), nil)
	require.NoError(t, err)

	const earlyWindow = 10
	var earlySum, lateSum float32

	for epoch := range settings.Epochs {
		stats, err := trainer.RunEpoch(epoch)
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
