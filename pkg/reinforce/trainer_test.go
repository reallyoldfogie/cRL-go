package reinforce

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/config"
	"github.com/reallyoldfogie/cRL-go/pkg/gridworldenv"
	"github.com/reallyoldfogie/cRL-go/pkg/policy"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/reallyoldfogie/cRL-go/pkg/snakeenv"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func smallTestSettings() config.Settings {
	return config.Settings{
		Epochs:       3,
		RolloutSize:  4,
		EpisodeLen:   5,
		Gamma:        0.99,
		LearningRate: 0.05,
		GridSize:     4, // 2x2 grid
		HiddenSize:   8,
		Seed:         1,
		Workers:      2,
	}
}

// snakeEnvFactory builds an EnvFactory for a snakeenv.Env over gridSize,
// used throughout this file so Trainer tests exercise the exact same
// rl.Environment-based path production code uses, rather than
// constructing snakeenv directly.
func snakeEnvFactory(gridSize int) EnvFactory {
	return func(rng *rand.Rand) (rl.Environment, error) {
		env, err := snakeenv.New(gridSize, rng)
		if err != nil {
			return nil, err
		}
		return snakeenv.NewAdapter(env), nil
	}
}

// gridWorldEnvFactory builds an EnvFactory for a gridworldenv.Env over
// gridSize.
func gridWorldEnvFactory(gridSize int) EnvFactory {
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
		assert.False(t, math.IsNaN(float64(stats.ReturnStd)), "epoch %d: return std is NaN", epoch)
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
	// goroutine scheduling order (see workerRNG's documentation).
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
// which is what lets a checkpoint loaded via policy.LoadFile actually
// resume training from its saved weights.
func TestNewUsesProvidedInitialParams(t *testing.T) {
	settings := smallTestSettings()

	rng := rand.New(rand.NewPCG(99, 99))
	initialParams := policy.NewParams(rng, snakeenv.StateVectorSize(settings.GridSize), settings.HiddenSize, snakeenv.NumActions)

	trainer, err := New(settings, snakeEnvFactory(settings.GridSize), initialParams)
	require.NoError(t, err)

	assert.Same(t, initialParams, trainer.Params(), "New must use the provided initialParams rather than reinitializing")
}

// TestNewRejectsMismatchedInitialParams confirms a checkpoint saved for
// a different architecture (e.g. a different grid size or hidden size)
// is rejected rather than silently producing a broken policy network.
func TestNewRejectsMismatchedInitialParams(t *testing.T) {
	settings := smallTestSettings()

	rng := rand.New(rand.NewPCG(1, 1))
	// Sized for a much larger grid than settings.GridSize describes.
	mismatched := policy.NewParams(rng, snakeenv.StateVectorSize(36), settings.HiddenSize, snakeenv.NumActions)

	_, err := New(settings, snakeEnvFactory(settings.GridSize), mismatched)
	assert.Error(t, err)
}

// TestRunEpochTrainsAgainstAnyEnvironment is the concrete proof that
// Trainer is genuinely environment-agnostic: gridworldenv has a
// different action count and observation layout than snakeenv (see
// pkg/gridworldenv's package doc), yet trains through the exact same
// Trainer/collectTrajectory code path with no reinforce or policy
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
// visibly learn" check: it runs enough epochs that the average return
// over the batch should trend upward as the policy learns to seek food
// and avoid the grid boundary. Skipped under `go test -short`.
func TestTrainingImprovesAverageReturn(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping slow training-trend test in -short mode")
	}

	// A small grid, short episodes, and an aggressive learning rate make
	// this converge in well under a second, while still exercising the
	// full rollout -> return/advantage -> gradient -> SGD pipeline
	// end-to-end. Slower, more "realistic" hyperparameters (e.g. the
	// original's 36-cell grid, 2000 epochs) show the same upward trend
	// but take far longer to plateau, which isn't a good fit for a unit
	// test.
	settings := config.Settings{
		Epochs:       200,
		RolloutSize:  64,
		EpisodeLen:   10,
		Gamma:        0.9,
		LearningRate: 2.0,
		GridSize:     4,
		HiddenSize:   16,
		Seed:         42,
		Workers:      8,
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

func BenchmarkRunEpoch(b *testing.B) {
	settings := config.Settings{
		Epochs:       1,
		RolloutSize:  16,
		EpisodeLen:   30,
		Gamma:        0.99,
		LearningRate: 0.05,
		GridSize:     36,
		HiddenSize:   32,
		Seed:         1,
		Workers:      4,
	}

	trainer, err := New(settings, snakeEnvFactory(settings.GridSize), nil)
	require.NoError(b, err)

	b.ResetTimer()
	for epoch := range b.N {
		if _, err := trainer.RunEpoch(epoch); err != nil {
			b.Fatal(err)
		}
	}
}
