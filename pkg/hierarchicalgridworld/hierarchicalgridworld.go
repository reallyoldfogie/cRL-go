// Package hierarchicalgridworld implements a genuinely multi-objective
// grid environment for validating pkg/hierarchical against: unlike
// pkg/snakeenv and pkg/gridworldenv (each built around a single
// objective), an agent here has to weigh collecting a resource,
// building at a fixed target (which needs a held resource), avoiding a
// fixed hazard cell, and handling an optional mob (fight it for a
// reward, or take repeated reward-free hits from lingering near it)
// against each other — exactly the kind of context-dependent
// prioritization a single flat policy may struggle with, which is what
// docs/plans/11-hierarchical-meta-controller-and-subpolicies.md's
// hierarchical design is meant to help with.
//
// This is a fresh implementation inspired by, not ported from,
// mc-rl-go's HierarchicalGridWorld — see that plan's Problem section
// for why (mc-rl-go's underlying PPO core is broken and untested).
package hierarchicalgridworld

import (
	"fmt"
	"math"
	"math/rand/v2"
)

// Action is one of the seven actions the agent can take: four
// movements, plus three interactions that only have an effect when the
// agent is positioned appropriately (Collect on the resource cell,
// Build on the build target while holding a resource, Attack on the
// mob's cell).
type Action int

const (
	ActionLeft Action = iota
	ActionRight
	ActionUp
	ActionDown
	ActionCollect
	ActionBuild
	ActionAttack

	// NumActions is the number of distinct actions, and therefore the
	// size of the policy network's output layer.
	NumActions = int(ActionAttack) + 1
)

const (
	stepPenalty        float32 = 1.0
	collectReward      float32 = 1.0
	buildReward        float32 = 2.0
	buildCompleteBonus float32 = 10.0
	attackReward       float32 = 5.0
	hazardPenalty      float32 = 5.0
	mobPenalty         float32 = 3.0
	// outOfBoundsPenalty is deliberately large relative to a typical
	// episode's accumulated stepPenalty: leaving the grid must always be
	// worse than simply surviving the rest of the episode doing nothing
	// productive, otherwise an untrained policy has a perverse incentive
	// to end the episode as early as possible rather than learn to
	// pursue any of the other objectives.
	outOfBoundsPenalty float32 = 25.0

	// mobMoveChance is the probability, per step, that an active mob
	// takes one step toward the agent.
	mobMoveChance float32 = 0.3

	// mobActiveChance is the probability, at Reset, that a mob is
	// present at all this episode.
	mobActiveChance float32 = 0.5

	// defaultBuildGoal is the number of resources the agent must
	// collect and deliver to the build target to complete the episode
	// successfully.
	defaultBuildGoal = 3
)

// Position is a location on the grid.
type Position struct {
	X, Y int
}

// Env is a square-grid environment with four simultaneously-live
// objectives: collect a resource, build at a fixed target using held
// resources, avoid a fixed hazard cell, and fight or avoid an optional
// roaming mob. Every entity's position is placed randomly (and the
// resource respawns elsewhere once collected), using rng for all of
// that randomness — Env owns its own random source rather than reading
// from a shared/global generator, mirroring pkg/snakeenv.Env.
type Env struct {
	rng *rand.Rand

	Rows, Cols, GridSize int

	Agent         Position
	BuildTarget   Position
	BuildProgress int
	BuildGoal     int
	Resource      Position
	ResourcesHeld int
	Hazard        Position
	Mob           Position
	MobActive     bool
	Steps         uint64
}

// Side returns the side length of a square grid with gridSize cells,
// reporting an error if gridSize is not a perfect square.
func Side(gridSize int) (int, error) {
	side := int(math.Sqrt(float64(gridSize)))
	if side*side != gridSize {
		return 0, fmt.Errorf("hierarchicalgridworld: gridSize %d is not a perfect square", gridSize)
	}
	return side, nil
}

// New creates an Env over a gridSize-cell square grid (gridSize must be
// a perfect square, e.g. 16 for a 4x4 grid) using rng for all
// randomness (entity placement, mob movement, resource respawning). The
// environment starts in a reset state.
func New(gridSize int, rng *rand.Rand) (*Env, error) {
	side, err := Side(gridSize)
	if err != nil {
		return nil, err
	}

	env := &Env{
		rng:       rng,
		Rows:      side,
		Cols:      side,
		GridSize:  gridSize,
		BuildGoal: defaultBuildGoal,
	}
	env.Reset()
	return env, nil
}

// ObservationSize returns the length of the observation vector produced
// by BuildObservation for a grid of the given size: one one-hot segment
// each for the agent, build target, resource, hazard, and mob
// positions, plus three scalar features (resources held, build
// progress, and whether a mob is currently active).
func ObservationSize(gridSize int) int {
	return 5*gridSize + 3
}

// randomPosition returns a uniformly random cell that doesn't coincide
// with any of avoid.
func (e *Env) randomPosition(avoid ...Position) Position {
	for {
		candidate := Position{X: e.rng.IntN(e.Cols), Y: e.rng.IntN(e.Rows)}
		if !containsPosition(avoid, candidate) {
			return candidate
		}
	}
}

func containsPosition(positions []Position, target Position) bool {
	for _, p := range positions {
		if p == target {
			return true
		}
	}
	return false
}

// Reset places the agent at the grid's origin and every other entity at
// a random, non-overlapping cell, and clears episode bookkeeping. A mob
// is present with probability mobActiveChance each episode.
func (e *Env) Reset() {
	e.Agent = Position{X: 0, Y: 0}
	e.BuildTarget = e.randomPosition(e.Agent)
	e.BuildProgress = 0
	e.Resource = e.randomPosition(e.Agent, e.BuildTarget)
	e.ResourcesHeld = 0
	e.Hazard = e.randomPosition(e.Agent, e.BuildTarget, e.Resource)
	e.Steps = 0

	e.MobActive = e.rng.Float32() < mobActiveChance
	if e.MobActive {
		e.Mob = e.randomPosition(e.Agent, e.BuildTarget, e.Resource, e.Hazard)
	}
}

// OutOfBounds reports whether the agent has left the grid.
func (e *Env) OutOfBounds() bool {
	return e.Agent.X < 0 || e.Agent.X >= e.Cols || e.Agent.Y < 0 || e.Agent.Y >= e.Rows
}

// Step applies action, updates the environment, and returns the
// resulting reward and whether the episode has ended (leaving the grid,
// or completing the build).
func (e *Env) Step(action Action) (reward float32, done bool) {
	switch action {
	case ActionLeft:
		e.Agent.X--
	case ActionRight:
		e.Agent.X++
	case ActionUp:
		e.Agent.Y++
	case ActionDown:
		e.Agent.Y--
	}
	e.Steps++

	if e.OutOfBounds() {
		return -stepPenalty - outOfBoundsPenalty, true
	}

	reward = -stepPenalty

	if action == ActionCollect && e.Agent == e.Resource {
		e.ResourcesHeld++
		e.Resource = e.randomPosition(e.Agent, e.BuildTarget, e.Hazard, e.mobOrElsewhere())
		reward += collectReward
	}

	if action == ActionBuild && e.Agent == e.BuildTarget && e.ResourcesHeld > 0 {
		e.ResourcesHeld--
		e.BuildProgress++
		reward += buildReward
		if e.BuildProgress >= e.BuildGoal {
			return reward + buildCompleteBonus, true
		}
	}

	if action == ActionAttack && e.MobActive && e.Agent == e.Mob {
		e.MobActive = false
		reward += attackReward
	}

	if e.Agent == e.Hazard {
		reward -= hazardPenalty
	}

	if e.MobActive {
		if e.Agent == e.Mob {
			reward -= mobPenalty
		}
		if e.rng.Float32() < mobMoveChance {
			e.moveMobToward(e.Agent)
		}
	}

	return reward, false
}

// mobOrElsewhere returns the mob's current position if active, or a
// position guaranteed to lie outside the grid, so randomPosition's
// avoid-list never needs to special-case an inactive mob.
func (e *Env) mobOrElsewhere() Position {
	if e.MobActive {
		return e.Mob
	}
	return Position{X: -1, Y: -1}
}

// moveMobToward moves the mob one cell closer to target along whichever
// axis is currently farther, a simple chase behavior.
func (e *Env) moveMobToward(target Position) {
	dx, dy := target.X-e.Mob.X, target.Y-e.Mob.Y
	if abs(dx) >= abs(dy) {
		switch {
		case dx > 0:
			e.Mob.X++
		case dx < 0:
			e.Mob.X--
		}
		return
	}
	switch {
	case dy > 0:
		e.Mob.Y++
	case dy < 0:
		e.Mob.Y--
	}
}

func abs(value int) int {
	if value < 0 {
		return -value
	}
	return value
}

// State is the subset of Env's fields BuildObservation needs, kept
// separate from *Env so BuildObservation stays a pure, independently
// testable function — mirroring pkg/gridworldenv.BuildObservation's
// plain-argument pattern, generalized here to more entities.
type State struct {
	Agent, BuildTarget, Resource, Hazard, Mob Position
	MobActive                                 bool
	ResourcesHeld, BuildProgress, BuildGoal   int
}

// BuildObservation fills dst with a one-hot encoding of state's agent,
// build target, resource, and hazard positions (each occupying its own
// gridSize-wide segment, in that order), followed by the mob's own
// gridSize-wide segment (left all zero if state.MobActive is false),
// followed by three scalar features: resources held and build progress
// (both normalized by state.BuildGoal), and whether a mob is currently
// active (0 or 1). dst must have length ObservationSize(gridSize).
func BuildObservation(dst []float32, state State, cols, gridSize int) error {
	for i := range dst {
		dst[i] = 0
	}

	positions := []Position{state.Agent, state.BuildTarget, state.Resource, state.Hazard}
	for segment, position := range positions {
		if err := setOneHot(dst, segment*gridSize, position, cols, gridSize); err != nil {
			return err
		}
	}
	if state.MobActive {
		const mobSegment = 4
		if err := setOneHot(dst, mobSegment*gridSize, state.Mob, cols, gridSize); err != nil {
			return err
		}
	}

	base := 5 * gridSize
	if state.BuildGoal > 0 {
		dst[base] = float32(state.ResourcesHeld) / float32(state.BuildGoal)
		dst[base+1] = float32(state.BuildProgress) / float32(state.BuildGoal)
	}
	if state.MobActive {
		dst[base+2] = 1
	}
	return nil
}

func setOneHot(dst []float32, segmentOffset int, position Position, cols, gridSize int) error {
	index := segmentOffset + position.Y*cols + position.X
	if index < segmentOffset || index >= segmentOffset+gridSize {
		return fmt.Errorf("hierarchicalgridworld: position %+v out of range for a %d-cell grid", position, gridSize)
	}
	dst[index] = 1
	return nil
}
