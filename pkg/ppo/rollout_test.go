package ppo

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/reinforce"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/reallyoldfogie/cRL-go/pkg/snakeenv"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func snakeEnvFactory(gridSize int) reinforce.EnvFactory {
	return func(rng *rand.Rand) (rl.Environment, error) {
		env, err := snakeenv.New(gridSize, rng)
		if err != nil {
			return nil, err
		}
		return snakeenv.NewAdapter(env), nil
	}
}

func TestCollectTrajectoryProducesFiniteLogProbsAndValues(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	gridSize := 4
	params := actorcritic.NewParams(rng, snakeenv.StateVectorSize(gridSize), 8, snakeenv.NumActions)

	rollout, err := collectTrajectory(context.Background(), params, snakeEnvFactory(gridSize), 10, rng)
	require.NoError(t, err)

	require.NotEmpty(t, rollout.Episode.Transitions)
	require.Len(t, rollout.LogProbs, len(rollout.Episode.Transitions))
	require.Len(t, rollout.Values, len(rollout.Episode.Transitions))

	for i := range rollout.Episode.Transitions {
		assert.False(t, math.IsNaN(float64(rollout.LogProbs[i])), "log-probability must not be NaN")
		assert.False(t, math.IsInf(float64(rollout.LogProbs[i]), 0), "log-probability must not be Inf")
		// A log-probability can never be positive (probabilities are at
		// most 1).
		assert.LessOrEqual(t, rollout.LogProbs[i], float32(0))

		assert.False(t, math.IsNaN(float64(rollout.Values[i])), "value estimate must not be NaN")
	}
}

func TestCollectTrajectoryIsDeterministicForAFixedSeed(t *testing.T) {
	gridSize := 4
	build := func() (*Rollout, error) {
		rng := rand.New(rand.NewPCG(5, 6))
		params := actorcritic.NewParams(rand.New(rand.NewPCG(5, 6)), snakeenv.StateVectorSize(gridSize), 8, snakeenv.NumActions)
		return collectTrajectory(context.Background(), params, snakeEnvFactory(gridSize), 10, rng)
	}

	rolloutA, err := build()
	require.NoError(t, err)
	rolloutB, err := build()
	require.NoError(t, err)

	assert.Equal(t, rolloutA.LogProbs, rolloutB.LogProbs)
	assert.Equal(t, rolloutA.Values, rolloutB.Values)
}
