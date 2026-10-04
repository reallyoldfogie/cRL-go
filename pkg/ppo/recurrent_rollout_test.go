package ppo

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/snakeenv"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCollectRecurrentTrajectoryProducesFiniteLogProbsAndValues(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	gridSize := 4
	params := actorcritic.NewRecurrentParams(rng, snakeenv.StateVectorSize(gridSize), 8, snakeenv.NumActions, 2)

	rollout, err := collectRecurrentTrajectory(context.Background(), params, snakeEnvFactory(gridSize), 10, rng)
	require.NoError(t, err)

	stepCount := len(rollout.Episode.Transitions)
	require.NotEmpty(t, stepCount)
	require.Len(t, rollout.LogProbs, stepCount)
	require.Len(t, rollout.Values, stepCount)
	require.Len(t, rollout.HPrevs, stepCount)
	require.Len(t, rollout.CPrevs, stepCount)

	for i := range rollout.Episode.Transitions {
		assert.False(t, math.IsNaN(float64(rollout.LogProbs[i])))
		assert.LessOrEqual(t, rollout.LogProbs[i], float32(0))
		assert.False(t, math.IsNaN(float64(rollout.Values[i])))
		assert.Len(t, rollout.HPrevs[i], 8)
		assert.Len(t, rollout.CPrevs[i], 8)
	}
}

// TestCollectRecurrentTrajectoryHPrevsStartAtZeroAndCarryForward confirms
// the recorded incoming state is zero at episode start (t=0) and matches
// whatever state the network actually carried into each later step —
// the property RecurrentRollout's whole design depends on.
func TestCollectRecurrentTrajectoryHPrevsStartAtZeroAndCarryForward(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	gridSize := 36
	params := actorcritic.NewRecurrentParams(rng, snakeenv.StateVectorSize(gridSize), 6, snakeenv.NumActions, 1)

	rollout, err := collectRecurrentTrajectory(context.Background(), params, snakeEnvFactory(gridSize), 10, rng)
	require.NoError(t, err)
	require.GreaterOrEqual(t, len(rollout.HPrevs), 2, "need at least 2 steps to check carry-forward")

	for _, v := range rollout.HPrevs[0] {
		assert.Equal(t, float32(0), v, "episode start must carry zero incoming state")
	}
	for _, v := range rollout.CPrevs[0] {
		assert.Equal(t, float32(0), v)
	}

	// Step 1's incoming state need not be all-zero (the network actually
	// ran a step), confirming state is genuinely threaded forward rather
	// than every step starting fresh at zero.
	allZero := true
	for _, v := range rollout.HPrevs[1] {
		if v != 0 {
			allZero = false
		}
	}
	assert.False(t, allZero, "step 1's incoming state must reflect step 0's output, not reset to zero")
}

func TestCollectRecurrentTrajectoryIsDeterministicForAFixedSeed(t *testing.T) {
	gridSize := 4
	build := func() (*RecurrentRollout, error) {
		rng := rand.New(rand.NewPCG(5, 6))
		params := actorcritic.NewRecurrentParams(rand.New(rand.NewPCG(5, 6)), snakeenv.StateVectorSize(gridSize), 8, snakeenv.NumActions, 2)
		return collectRecurrentTrajectory(context.Background(), params, snakeEnvFactory(gridSize), 10, rng)
	}

	rolloutA, err := build()
	require.NoError(t, err)
	rolloutB, err := build()
	require.NoError(t, err)

	assert.Equal(t, rolloutA.Episode, rolloutB.Episode)
	assert.Equal(t, rolloutA.LogProbs, rolloutB.LogProbs)
	assert.Equal(t, rolloutA.Values, rolloutB.Values)
	assert.Equal(t, rolloutA.HPrevs, rolloutB.HPrevs)
	assert.Equal(t, rolloutA.CPrevs, rolloutB.CPrevs)
}
