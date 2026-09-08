package reinforce

import (
	"context"
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

// smallTestSettings starts from config.Default() (rather than a bare
// struct literal) and overrides only the fields this REINFORCE test
// suite cares about, so it automatically stays valid as config.Settings
// grows fields for other trainers (e.g. pkg/ppo's) that this suite
// neither sets nor needs.
func smallTestSettings() config.Settings {
	settings := config.Default()
	settings.Epochs = 3
	settings.RolloutSize = 4
	settings.EpisodeLen = 5
	settings.Gamma = 0.99
	settings.LearningRate = 0.05
	settings.GridSize = 4 // 2x2 grid
	settings.HiddenSize = 8
	settings.Seed = 1
	settings.Workers = 2
	return settings
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
		stats, err := trainer.RunEpoch(context.Background(), epoch)
		require.NoError(t, err)

		assert.False(t, math.IsNaN(float64(stats.AverageReturn)), "epoch %d: average return is NaN", epoch)
		assert.False(t, math.IsNaN(float64(stats.ReturnStd)), "epoch %d: return std is NaN", epoch)
		assert.GreaterOrEqual(t, stats.SampleCount, 0)
		assert.Equal(t, epoch, stats.Epoch)
	}
}

// TestTrainerGradientNormReflectsRunEpoch confirms Trainer.GradientNorm
// is 0 before any RunEpoch call (nothing accumulated yet) and nonzero
// after one, exercising the pass-through to policy.TrainingNetwork's own
// GradientNorm that a caller with no access to Trainer's unexported
// network field would otherwise have no way to reach — added as a
// diagnostic for exactly the situation an mc-agent peer session hit: a
// training run whose observable EpochStats/weights look completely flat,
// where a caller can now check whether that's a healthy-but-tiny
// gradient or a genuinely vanished one.
func TestTrainerGradientNormReflectsRunEpoch(t *testing.T) {
	settings := smallTestSettings()
	trainer, err := New(settings, snakeEnvFactory(settings.GridSize), nil)
	require.NoError(t, err)

	assert.Equal(t, float32(0), trainer.GradientNorm(), "no RunEpoch has happened yet")

	_, err = trainer.RunEpoch(context.Background(), 0)
	require.NoError(t, err)

	assert.Greater(t, trainer.GradientNorm(), float32(0), "RunEpoch must accumulate a nonzero gradient in the normal case")
}

// TestNewWiresEntropyCoefIntoTrainingNetwork confirms settings.EntropyCoef
// actually reaches the underlying policy.TrainingNetwork rather than
// being validated by config.Settings and then silently dropped — the
// exact bug an mc-agent peer session found live (a REINFORCE-trained
// policy collapsing to a frozen, deterministic distribution after one
// gradient step, independent of learning rate, because nothing opposed
// the softmax saturating). Two trainers built from the same Seed start
// with identical params; only EntropyCoef differs, so their training
// networks' losses must differ once entropy actually participates.
func TestNewWiresEntropyCoefIntoTrainingNetwork(t *testing.T) {
	settingsA := smallTestSettings()
	settingsA.EntropyCoef = 0
	trainerA, err := New(settingsA, snakeEnvFactory(settingsA.GridSize), nil)
	require.NoError(t, err)

	settingsB := smallTestSettings()
	settingsB.EntropyCoef = 0.5
	trainerB, err := New(settingsB, snakeEnvFactory(settingsB.GridSize), nil)
	require.NoError(t, err)

	require.Equal(t, trainerA.params.W0.Data, trainerB.params.W0.Data,
		"sanity: same Seed must produce identical initial params, so any Loss difference below comes from EntropyCoef alone")

	input := make([]float32, trainerA.network.Input.Val.Rows)
	for i := range input {
		input[i] = 0.1
	}
	copy(trainerA.network.Input.Val.Data, input)
	copy(trainerB.network.Input.Val.Data, input)
	trainerA.network.Advantage.Val.Clear()
	trainerA.network.Advantage.Val.Data[0] = 0.7
	trainerB.network.Advantage.Val.Clear()
	trainerB.network.Advantage.Val.Data[0] = 0.7

	trainerA.network.Graph.Forward()
	trainerB.network.Graph.Forward()

	assert.NotEqual(t, trainerA.network.Loss.Val.Data, trainerB.network.Loss.Val.Data,
		"settings.EntropyCoef must actually reach the training network's loss")
}

func TestRunEpochIsDeterministicForAFixedSeed(t *testing.T) {
	settings := smallTestSettings()

	trainerA, err := New(settings, snakeEnvFactory(settings.GridSize), nil)
	require.NoError(t, err)
	statsA, err := trainerA.RunEpoch(context.Background(), 0)
	require.NoError(t, err)

	trainerB, err := New(settings, snakeEnvFactory(settings.GridSize), nil)
	require.NoError(t, err)
	statsB, err := trainerB.RunEpoch(context.Background(), 0)
	require.NoError(t, err)

	// A fixed seed must reproduce identical results regardless of
	// goroutine scheduling order (see WorkerRNG's documentation).
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
		stats, err := trainer.RunEpoch(context.Background(), epoch)
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
	settings := config.Default()
	settings.Epochs = 200
	settings.RolloutSize = 64
	settings.EpisodeLen = 10
	settings.Gamma = 0.9
	settings.LearningRate = 2.0
	settings.GridSize = 4
	settings.HiddenSize = 16
	settings.Seed = 42
	settings.Workers = 8

	trainer, err := New(settings, snakeEnvFactory(settings.GridSize), nil)
	require.NoError(t, err)

	const earlyWindow = 10
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

func BenchmarkRunEpoch(b *testing.B) {
	settings := config.Default()
	settings.Epochs = 1
	settings.RolloutSize = 16
	settings.EpisodeLen = 30
	settings.Gamma = 0.99
	settings.LearningRate = 0.05
	settings.GridSize = 36
	settings.HiddenSize = 32
	settings.Seed = 1
	settings.Workers = 4

	trainer, err := New(settings, snakeEnvFactory(settings.GridSize), nil)
	require.NoError(b, err)

	ctx := context.Background()
	b.ResetTimer()
	for epoch := range b.N {
		if _, err := trainer.RunEpoch(ctx, epoch); err != nil {
			b.Fatal(err)
		}
	}
}
