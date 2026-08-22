package gridworldenv

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newTestEnv(t *testing.T) *Env {
	t.Helper()
	env, err := New(9) // 3x3 grid
	require.NoError(t, err)
	return env
}

func TestNewRejectsNonSquareGridSize(t *testing.T) {
	_, err := New(10)
	assert.Error(t, err)
}

func TestNewPlacesAgentAtOriginAndGoalAtOppositeCorner(t *testing.T) {
	env := newTestEnv(t)

	assert.Equal(t, Position{X: 0, Y: 0}, env.Agent)
	assert.Equal(t, Position{X: 2, Y: 2}, env.Goal) // 3x3 grid -> far corner (2,2)
}

func TestResetReturnsAgentToOrigin(t *testing.T) {
	env := newTestEnv(t)
	_, _ = env.Step(ActionRight)
	require.NotEqual(t, Position{X: 0, Y: 0}, env.Agent)

	env.Reset()

	assert.Equal(t, Position{X: 0, Y: 0}, env.Agent)
	assert.Equal(t, uint64(0), env.Steps)
	// The goal does not move between episodes.
	assert.Equal(t, Position{X: 2, Y: 2}, env.Goal)
}

func TestStepMovesInRequestedDirection(t *testing.T) {
	env := newTestEnv(t)
	start := env.Agent

	_, done := env.Step(ActionUp)

	assert.False(t, done)
	assert.Equal(t, Position{X: start.X, Y: start.Y + 1}, env.Agent)
}

func TestStepEveryMoveCostsAPenalty(t *testing.T) {
	env := newTestEnv(t)

	reward, done := env.Step(ActionRight)

	assert.False(t, done)
	assert.Equal(t, float32(-1), reward)
}

func TestStepReachingGoalEndsEpisodeWithBonus(t *testing.T) {
	env := newTestEnv(t)
	env.Agent = Position{X: 1, Y: 2}

	reward, done := env.Step(ActionRight) // (1,2) -> (2,2), the goal

	assert.True(t, done)
	assert.Equal(t, float32(19), reward) // -stepPenalty + goalReward
}

func TestStepOutOfBoundsEndsEpisodeWithPenalty(t *testing.T) {
	env := newTestEnv(t)
	env.Agent = Position{X: 0, Y: 0}

	reward, done := env.Step(ActionLeft) // (0,0) -> (-1,0), off the grid

	assert.True(t, done)
	assert.Equal(t, float32(-11), reward) // -stepPenalty - outOfBoundsPenalty
	assert.True(t, env.OutOfBounds())
}

func TestBuildObservationOneHotEncoding(t *testing.T) {
	gridSize, cols := 9, 3
	dst := make([]float32, ObservationSize(gridSize))

	err := BuildObservation(dst, Position{X: 1, Y: 0}, Position{X: 2, Y: 2}, cols, gridSize)
	require.NoError(t, err)

	agentIndex := 0*cols + 1
	goalIndex := gridSize + 2*cols + 2

	var onesAt []int
	for i, v := range dst {
		if v == 1 {
			onesAt = append(onesAt, i)
		}
	}
	assert.ElementsMatch(t, []int{agentIndex, goalIndex}, onesAt)
}

func TestObservationSizeMatchesEncodingLayout(t *testing.T) {
	// 2*gridSize (agent + goal one-hot segments), unlike snakeenv which
	// adds a third segment for direction of travel.
	assert.Equal(t, 2*9, ObservationSize(9))
}
