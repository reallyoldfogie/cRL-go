package ppo

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/config"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/reallyoldfogie/cRL-go/pkg/snakeenv"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func smallRecurrentTestSettings() config.Settings {
	s := smallTestSettings()
	s.HiddenSize = 6
	s.RecurrentChunkLen = 3
	return s
}

// TestRecurrentRunEpochSmokeTestNoPanicsOrNaNs mirrors
// TestRunEpochSmokeTestNoPanicsOrNaNs for RecurrentTrainer: several
// epochs against a toy environment, with no NaNs or panics, confirming
// the whole chunked-BPTT pipeline (rollout collection, GAE, chunking,
// padding, shuffled-minibatch Adam updates) works end to end.
func TestRecurrentRunEpochSmokeTestNoPanicsOrNaNs(t *testing.T) {
	settings := smallRecurrentTestSettings()
	trainer, err := NewRecurrent(settings, snakeEnvFactory(settings.GridSize), nil)
	require.NoError(t, err)

	for epoch := range settings.Epochs {
		stats, err := trainer.RunEpoch(context.Background(), epoch)
		require.NoError(t, err)

		assert.False(t, math.IsNaN(float64(stats.AverageReturn)), "epoch %d: average return is NaN", epoch)
		assert.GreaterOrEqual(t, stats.SampleCount, 0)
		assert.Equal(t, epoch, stats.Epoch)
	}
}

func TestRecurrentRunEpochIsDeterministicForAFixedSeed(t *testing.T) {
	settings := smallRecurrentTestSettings()

	trainerA, err := NewRecurrent(settings, snakeEnvFactory(settings.GridSize), nil)
	require.NoError(t, err)
	statsA, err := trainerA.RunEpoch(context.Background(), 0)
	require.NoError(t, err)

	trainerB, err := NewRecurrent(settings, snakeEnvFactory(settings.GridSize), nil)
	require.NoError(t, err)
	statsB, err := trainerB.RunEpoch(context.Background(), 0)
	require.NoError(t, err)

	assert.Equal(t, statsA, statsB)
}

func TestNewRecurrentRejectsNonPositiveChunkLen(t *testing.T) {
	settings := smallRecurrentTestSettings()
	settings.RecurrentChunkLen = 0
	_, err := NewRecurrent(settings, snakeEnvFactory(settings.GridSize), nil)
	assert.ErrorContains(t, err, "RecurrentChunkLen")
}

func TestNewRecurrentUsesProvidedInitialParams(t *testing.T) {
	settings := smallRecurrentTestSettings()

	rng := rand.New(rand.NewPCG(99, 99))
	initialParams := actorcritic.NewRecurrentParams(rng, snakeenv.StateVectorSize(settings.GridSize), settings.HiddenSize, snakeenv.NumActions, 2)

	trainer, err := NewRecurrent(settings, snakeEnvFactory(settings.GridSize), initialParams)
	require.NoError(t, err)
	assert.Same(t, initialParams, trainer.Params())
}

// TestChunkRolloutPadsAShortFinalChunkWithZeroedZeroAdvantageSteps
// confirms a rollout whose length isn't a multiple of chunkLen pads its
// last chunk out, rather than dropping it or producing a short slice,
// and that every padded entry carries Advantage 0 (so
// normalizeRecurrentAdvantages, and the network's own StepMask, both
// have something well-defined to work with).
func TestChunkRolloutPadsAShortFinalChunkWithZeroedAdvantageSteps(t *testing.T) {
	rollout := &RecurrentRollout{
		Rollout: &Rollout{
			Episode:  episodeOfLength(5),
			LogProbs: []float32{-0.1, -0.2, -0.3, -0.4, -0.5},
			Values:   []float32{1, 2, 3, 4, 5},
		},
		HPrevs: [][]float32{{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}},
		CPrevs: [][]float32{{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}},
	}
	advantages := []float32{0.1, 0.2, 0.3, 0.4, 0.5}
	returns := []float32{1, 2, 3, 4, 5}

	chunks := chunkRollout(rollout, advantages, returns, 3, 4)
	require.Len(t, chunks, 2, "5 steps at chunkLen=3 must produce 2 chunks (3 + 2 padded to 3)")

	require.Len(t, chunks[0].Steps, 3)
	for _, step := range chunks[0].Steps {
		assert.True(t, step.Real)
	}
	assert.Equal(t, []float32{0, 0}, chunks[0].HPrevStart)

	require.Len(t, chunks[1].Steps, 3)
	assert.True(t, chunks[1].Steps[0].Real)
	assert.True(t, chunks[1].Steps[1].Real)
	assert.False(t, chunks[1].Steps[2].Real, "the 6th slot pads the 5-step episode out to chunkLen=3")
	assert.Equal(t, float32(0), chunks[1].Steps[2].Advantage)
	assert.Len(t, chunks[1].Steps[2].Observation.Values, 4, "a padded step's placeholder observation must still have the right InputSize")
	assert.Equal(t, []float32{3, 3}, chunks[1].HPrevStart, "the second chunk starts from step 3's recorded incoming state")
}

func episodeOfLength(n int) *rl.Episode {
	transitions := make([]rl.Transition, n)
	for i := range transitions {
		transitions[i] = rl.Transition{Observation: rl.Observation{Values: []float32{0, 0, 0, 0}}, Action: 0, Reward: 0}
	}
	return &rl.Episode{Transitions: transitions}
}
