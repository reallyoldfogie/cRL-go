package snakeenv

import (
	"math/rand/v2"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/mat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newTestEnv(t *testing.T, seed uint64) *Env {
	t.Helper()
	rng := rand.New(rand.NewPCG(seed, seed))
	env, err := New(36, rng)
	require.NoError(t, err)
	return env
}

func TestNewRejectsNonSquareGridSize(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 1))
	_, err := New(10, rng)
	assert.Error(t, err)
}

func TestResetPlacesSnakeAtCenter(t *testing.T) {
	env := newTestEnv(t, 1)
	env.Reset()

	assert.Equal(t, Position{X: 3, Y: 3}, env.Snake) // 6x6 grid -> center (3,3)
	assert.NotEqual(t, env.Snake, env.Food)          // Reset must not overlap food with the snake
	assert.Equal(t, float32(0), env.Score)
	assert.Equal(t, 0, env.FoodsEaten)
	assert.Equal(t, ActionRight, env.POV)
}

func TestStepMovesInRequestedDirection(t *testing.T) {
	env := newTestEnv(t, 2)
	env.Reset()
	start := env.Snake

	_, done := env.Step(ActionUp)
	assert.False(t, done)
	assert.Equal(t, Position{X: start.X, Y: start.Y + 1}, env.Snake)
	assert.Equal(t, ActionUp, env.POV)
}

func TestStepNoneKeepsCurrentDirection(t *testing.T) {
	env := newTestEnv(t, 3)
	env.Reset()

	_, done := env.Step(ActionDown)
	require.False(t, done)
	afterDown := env.Snake

	_, done = env.Step(ActionNone)
	require.False(t, done)
	assert.Equal(t, Position{X: afterDown.X, Y: afterDown.Y - 1}, env.Snake)
	assert.Equal(t, ActionDown, env.POV)
}

func TestStepEatingFoodGrantsRewardAndRelocatesFood(t *testing.T) {
	env := newTestEnv(t, 4)
	env.Snake = Position{X: 2, Y: 2}
	env.Food = Position{X: 3, Y: 2}
	env.POV = ActionRight

	reward, done := env.Step(ActionRight)

	assert.False(t, done)
	assert.Equal(t, float32(20), reward)
	assert.Equal(t, 1, env.FoodsEaten)
	assert.NotEqual(t, Position{X: 3, Y: 2}, env.Food) // food must have relocated
}

func TestStepOutOfBoundsEndsEpisodeWithPenalty(t *testing.T) {
	env := newTestEnv(t, 5)
	env.Snake = Position{X: int32(env.Cols - 1), Y: 0}
	env.Food = Position{X: 0, Y: int32(env.Rows - 1)} // clearly not adjacent to the exit path
	env.POV = ActionRight

	reward, done := env.Step(ActionRight)

	assert.True(t, done)
	assert.Equal(t, float32(-10), reward)
	assert.True(t, env.GameOver())
}

func TestBuildStateVectorOneHotEncoding(t *testing.T) {
	gridSize := 36
	cols := 6
	dst := mat.New(StateVectorSize(gridSize), 1)

	err := BuildStateVector(dst, Position{X: 1, Y: 2}, Position{X: 4, Y: 5}, ActionUp, cols, gridSize)
	require.NoError(t, err)

	var onesAt []int
	for i, v := range dst.Data {
		if v == 1 {
			onesAt = append(onesAt, i)
		}
	}

	snakeIndex := 2*cols + 1 // y*cols + x
	foodIndex := gridSize + 5*cols + 4
	povIndex := 2*gridSize + int(ActionUp)

	assert.ElementsMatch(t, []int{snakeIndex, foodIndex, povIndex}, onesAt)

	var sum float32
	for _, v := range dst.Data {
		sum += v
	}
	assert.Equal(t, float32(3), sum, "exactly three entries should be set")
}

func TestStateVectorSizeMatchesEncodingLayout(t *testing.T) {
	// 2*gridSize (snake + food one-hot segments) + NumActions (pov segment).
	assert.Equal(t, 2*36+NumActions, StateVectorSize(36))
}

func TestRenderPlacesSnakeAndFoodMarkers(t *testing.T) {
	env := newTestEnv(t, 1)
	env.Snake = Position{X: 2, Y: 2}
	env.Food = Position{X: 4, Y: 4}

	lines := env.Render()
	require.Len(t, lines, env.Rows)
	for _, renderedLine := range lines {
		assert.Len(t, renderedLine, env.Cols)
	}

	// Lines()'s row 0 is the highest Y, so Y=2 is at index Rows-1-2.
	assert.Equal(t, uint8('@'), lineAt(lines, env.Rows, 2)[2])
	assert.Equal(t, uint8('F'), lineAt(lines, env.Rows, 4)[4])
}

func TestRenderOmitsSnakeMarkerOnceGameOver(t *testing.T) {
	env := newTestEnv(t, 2)
	env.Snake = Position{X: -1, Y: 0} // out of bounds

	for _, l := range env.Render() {
		assert.NotContains(t, l, "@")
	}
}

// lineAt returns lines[y]'s content, accounting for Render's
// highest-Y-first ordering (see pkg/gridrender.Grid.Lines).
func lineAt(lines []string, rows, y int) string {
	return lines[rows-1-y]
}
