package hierarchicalgridworld

import (
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newTestEnv(t *testing.T, seed uint64) *Env {
	t.Helper()
	rng := rand.New(rand.NewPCG(seed, seed))
	env, err := New(16, rng) // 4x4 grid
	require.NoError(t, err)
	return env
}

func TestNewRejectsNonSquareGridSize(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 1))
	_, err := New(10, rng)
	assert.Error(t, err)
}

func TestResetPlacesEntitiesAtDistinctPositions(t *testing.T) {
	env := newTestEnv(t, 1)

	assert.Equal(t, Position{X: 0, Y: 0}, env.Agent)
	assert.NotEqual(t, env.Agent, env.BuildTarget)
	assert.NotEqual(t, env.Agent, env.Resource)
	assert.NotEqual(t, env.Agent, env.Hazard)
	assert.NotEqual(t, env.BuildTarget, env.Resource)
	assert.NotEqual(t, env.BuildTarget, env.Hazard)
	assert.NotEqual(t, env.Resource, env.Hazard)
	if env.MobActive {
		assert.NotEqual(t, env.Agent, env.Mob)
		assert.NotEqual(t, env.BuildTarget, env.Mob)
		assert.NotEqual(t, env.Resource, env.Mob)
		assert.NotEqual(t, env.Hazard, env.Mob)
	}
	assert.Equal(t, 0, env.BuildProgress)
	assert.Equal(t, 0, env.ResourcesHeld)
	assert.Equal(t, uint64(0), env.Steps)
}

func TestStepMovesInRequestedDirection(t *testing.T) {
	env := newTestEnv(t, 2)
	env.Agent = Position{X: 1, Y: 1}

	reward, done := env.Step(ActionUp)

	assert.False(t, done)
	assert.Equal(t, Position{X: 1, Y: 2}, env.Agent)
	assert.Equal(t, -stepPenalty, reward)
}

func TestStepOutOfBoundsEndsEpisodeWithPenalty(t *testing.T) {
	env := newTestEnv(t, 3)
	env.Agent = Position{X: 0, Y: 0}

	reward, done := env.Step(ActionLeft)

	assert.True(t, done)
	assert.Equal(t, -stepPenalty-outOfBoundsPenalty, reward)
	assert.True(t, env.OutOfBounds())
}

func TestStepCollectingResourceGrantsRewardAndRespawnsIt(t *testing.T) {
	env := newTestEnv(t, 4)
	env.Agent = Position{X: 2, Y: 2}
	env.Resource = Position{X: 2, Y: 2}

	reward, done := env.Step(ActionCollect)

	assert.False(t, done)
	assert.Equal(t, -stepPenalty+collectReward, reward)
	assert.Equal(t, 1, env.ResourcesHeld)
	assert.NotEqual(t, Position{X: 2, Y: 2}, env.Resource, "resource must relocate after being collected")
}

func TestStepCollectingWithoutBeingOnResourceDoesNothingExtra(t *testing.T) {
	env := newTestEnv(t, 5)
	env.Agent = Position{X: 1, Y: 1}
	env.Resource = Position{X: 3, Y: 3}

	reward, done := env.Step(ActionCollect)

	assert.False(t, done)
	assert.Equal(t, -stepPenalty, reward)
	assert.Equal(t, 0, env.ResourcesHeld)
}

func TestStepBuildingWithoutResourceHeldDoesNothing(t *testing.T) {
	env := newTestEnv(t, 6)
	env.Agent = Position{X: 2, Y: 2}
	env.BuildTarget = Position{X: 2, Y: 2}
	env.ResourcesHeld = 0

	reward, done := env.Step(ActionBuild)

	assert.False(t, done)
	assert.Equal(t, -stepPenalty, reward)
	assert.Equal(t, 0, env.BuildProgress)
}

func TestStepBuildingWithResourceGrantsRewardAndProgress(t *testing.T) {
	env := newTestEnv(t, 7)
	env.Agent = Position{X: 2, Y: 2}
	env.BuildTarget = Position{X: 2, Y: 2}
	env.ResourcesHeld = 2

	reward, done := env.Step(ActionBuild)

	assert.False(t, done)
	assert.Equal(t, -stepPenalty+buildReward, reward)
	assert.Equal(t, 1, env.BuildProgress)
	assert.Equal(t, 1, env.ResourcesHeld)
}

func TestStepCompletingBuildEndsEpisodeWithBonus(t *testing.T) {
	env := newTestEnv(t, 8)
	env.Agent = Position{X: 2, Y: 2}
	env.BuildTarget = Position{X: 2, Y: 2}
	env.BuildGoal = 1
	env.ResourcesHeld = 1

	reward, done := env.Step(ActionBuild)

	assert.True(t, done)
	assert.Equal(t, -stepPenalty+buildReward+buildCompleteBonus, reward)
	assert.Equal(t, 1, env.BuildProgress)
}

func TestStepAttackingMobGrantsRewardAndDeactivatesIt(t *testing.T) {
	env := newTestEnv(t, 9)
	env.Agent = Position{X: 2, Y: 2}
	env.MobActive = true
	env.Mob = Position{X: 2, Y: 2}

	reward, done := env.Step(ActionAttack)

	assert.False(t, done)
	assert.Equal(t, -stepPenalty+attackReward, reward)
	assert.False(t, env.MobActive)
}

func TestStepSteppingOnHazardInflictsPenalty(t *testing.T) {
	env := newTestEnv(t, 10)
	env.Agent = Position{X: 1, Y: 1}
	env.Hazard = Position{X: 1, Y: 2}

	reward, done := env.Step(ActionUp)

	assert.False(t, done)
	assert.Equal(t, -stepPenalty-hazardPenalty, reward)
}

func TestStepStandingOnActiveMobInflictsPenalty(t *testing.T) {
	env := newTestEnv(t, 11)
	env.Agent = Position{X: 1, Y: 1}
	env.MobActive = true
	env.Mob = Position{X: 1, Y: 2}

	reward, _ := env.Step(ActionUp)

	assert.Equal(t, -stepPenalty-mobPenalty, reward)
}

func TestRenderPlacesEveryEntityMarker(t *testing.T) {
	env := newTestEnv(t, 1)
	env.Agent = Position{X: 0, Y: 0}
	env.BuildTarget = Position{X: 1, Y: 0}
	env.Resource = Position{X: 2, Y: 0}
	env.Hazard = Position{X: 3, Y: 0}
	env.MobActive = true
	env.Mob = Position{X: 0, Y: 1}

	lines := env.Render()
	require.Len(t, lines, env.Rows)
	for _, l := range lines {
		assert.Len(t, l, env.Cols)
	}

	row0 := lines[env.Rows-1] // Y=0
	assert.Equal(t, uint8('@'), row0[0])
	assert.Equal(t, uint8('T'), row0[1])
	assert.Equal(t, uint8('R'), row0[2])
	assert.Equal(t, uint8('H'), row0[3])
	assert.Equal(t, uint8('M'), lines[env.Rows-1-1][0])
}

func TestRenderOmitsMobMarkerWhenInactive(t *testing.T) {
	env := newTestEnv(t, 2)
	env.MobActive = false

	for _, l := range env.Render() {
		assert.NotContains(t, l, "M")
	}
}

func TestRenderDrawsAgentOnTopOfAnotherEntity(t *testing.T) {
	env := newTestEnv(t, 3)
	env.Agent = Position{X: 1, Y: 1}
	env.Hazard = Position{X: 1, Y: 1}

	lines := env.Render()
	assert.Equal(t, uint8('@'), lines[env.Rows-1-1][1])
}

func TestRenderOmitsAgentMarkerOnceOutOfBounds(t *testing.T) {
	env := newTestEnv(t, 4)
	env.Agent = Position{X: -1, Y: 0}

	for _, l := range env.Render() {
		assert.NotContains(t, l, "@")
	}
}

func TestBuildObservationOneHotEncoding(t *testing.T) {
	gridSize, cols := 16, 4
	dst := make([]float32, ObservationSize(gridSize))

	state := State{
		Agent:         Position{X: 1, Y: 0},
		BuildTarget:   Position{X: 2, Y: 0},
		Resource:      Position{X: 3, Y: 0},
		Hazard:        Position{X: 0, Y: 1},
		Mob:           Position{X: 1, Y: 1},
		MobActive:     true,
		ResourcesHeld: 1,
		BuildProgress: 2,
		BuildGoal:     3,
	}

	err := BuildObservation(dst, state, cols, gridSize)
	require.NoError(t, err)

	agentIndex := 0*gridSize + 0*cols + 1
	buildTargetIndex := 1*gridSize + 0*cols + 2
	resourceIndex := 2*gridSize + 0*cols + 3
	hazardIndex := 3*gridSize + 1*cols + 0
	mobIndex := 4*gridSize + 1*cols + 1

	var onesAt []int
	for i, v := range dst[:5*gridSize] { // exclude the trailing scalar features, which may also legitimately equal 1
		if v == 1 {
			onesAt = append(onesAt, i)
		}
	}
	assert.ElementsMatch(t, []int{agentIndex, buildTargetIndex, resourceIndex, hazardIndex, mobIndex}, onesAt)

	base := 5 * gridSize
	assert.InDelta(t, float32(1)/3, dst[base], 1e-6)
	assert.InDelta(t, float32(2)/3, dst[base+1], 1e-6)
	assert.Equal(t, float32(1), dst[base+2])
}

func TestBuildObservationLeavesMobSegmentZeroWhenInactive(t *testing.T) {
	gridSize, cols := 16, 4
	dst := make([]float32, ObservationSize(gridSize))

	state := State{
		Agent:       Position{X: 0, Y: 0},
		BuildTarget: Position{X: 1, Y: 0},
		Resource:    Position{X: 2, Y: 0},
		Hazard:      Position{X: 3, Y: 0},
		MobActive:   false,
		BuildGoal:   3,
	}

	err := BuildObservation(dst, state, cols, gridSize)
	require.NoError(t, err)

	mobSegment := dst[4*gridSize : 5*gridSize]
	for _, v := range mobSegment {
		assert.Equal(t, float32(0), v)
	}
	assert.Equal(t, float32(0), dst[5*gridSize+2], "mob-active flag must be 0 when no mob is present")
}

func TestObservationSizeMatchesEncodingLayout(t *testing.T) {
	// 5 one-hot position segments (agent, build target, resource,
	// hazard, mob) plus 3 scalar features.
	assert.Equal(t, 5*16+3, ObservationSize(16))
}
