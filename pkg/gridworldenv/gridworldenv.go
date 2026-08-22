// Package gridworldenv implements a minimal, deterministic goal-seeking
// grid environment. Its only purpose is to validate that pkg/reinforce
// and pkg/policy are genuinely environment-agnostic rather than shaped
// around pkg/snakeenv in disguise: it has a different action count (4,
// with no "keep current direction" action), a different observation
// layout (agent + goal position, with no direction-of-travel segment),
// and a different reward shape (a per-step penalty plus a one-time goal
// bonus, rather than Snake's repeated food reward) than snakeenv, but
// trains through the exact same rl.Environment-based pipeline.
package gridworldenv

import (
	"fmt"
	"math"
)

// Action is one of the four moves the agent can take.
type Action int

const (
	ActionLeft Action = iota
	ActionRight
	ActionUp
	ActionDown

	// NumActions is the number of distinct actions, and therefore the
	// size of the policy network's output layer.
	NumActions = int(ActionDown) + 1
)

const (
	stepPenalty        float32 = 1.0
	goalReward         float32 = 20.0
	outOfBoundsPenalty float32 = 10.0
)

// Position is a location on the grid.
type Position struct {
	X, Y int
}

// Env is a square-grid environment: an agent starts in one corner and
// must reach a fixed goal cell in the opposite corner in as few steps as
// possible. Leaving the grid ends the episode with a penalty, mirroring
// snakeenv's boundary handling but via an entirely independent
// implementation (no shared state or types with snakeenv).
type Env struct {
	Rows, Cols int
	GridSize   int

	Agent Position
	Goal  Position
	Steps uint64
}

// Side returns the side length of a square grid with gridSize cells,
// reporting an error if gridSize is not a perfect square.
func Side(gridSize int) (int, error) {
	side := int(math.Sqrt(float64(gridSize)))
	if side*side != gridSize {
		return 0, fmt.Errorf("gridworldenv: gridSize %d is not a perfect square", gridSize)
	}
	return side, nil
}

// New creates an Env over a gridSize-cell square grid (gridSize must be a
// perfect square), with the goal fixed at the corner diagonally opposite
// the agent's starting corner.
func New(gridSize int) (*Env, error) {
	side, err := Side(gridSize)
	if err != nil {
		return nil, err
	}

	env := &Env{
		Rows:     side,
		Cols:     side,
		GridSize: gridSize,
		Goal:     Position{X: side - 1, Y: side - 1},
	}
	env.Reset()
	return env, nil
}

// ObservationSize returns the length of the one-hot state vector
// produced by BuildObservation for a grid of the given size: one segment
// for the agent's position, one for the goal's position.
func ObservationSize(gridSize int) int {
	return 2 * gridSize
}

// Reset places the agent back at its starting corner (0, 0) and clears
// episode bookkeeping. The goal position does not move between episodes.
func (e *Env) Reset() {
	e.Agent = Position{X: 0, Y: 0}
	e.Steps = 0
}

// OutOfBounds reports whether the agent has left the grid.
func (e *Env) OutOfBounds() bool {
	return e.Agent.X < 0 || e.Agent.X >= e.Cols || e.Agent.Y < 0 || e.Agent.Y >= e.Rows
}

// Step applies action, moving the agent one cell, and returns the
// resulting reward and whether the episode has ended (goal reached or
// grid boundary crossed). Every step costs stepPenalty, encouraging
// shorter paths to the goal.
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

	reward = -stepPenalty
	if e.OutOfBounds() {
		return reward - outOfBoundsPenalty, true
	}
	if e.Agent == e.Goal {
		return reward + goalReward, true
	}
	return reward, false
}

// BuildObservation fills dst with a one-hot encoding of the given agent
// and goal positions: a 1 at the agent's grid cell, and a 1 at
// (gridSize + the goal's grid cell). dst must have length
// ObservationSize(gridSize).
func BuildObservation(dst []float32, agent, goal Position, cols, gridSize int) error {
	for i := range dst {
		dst[i] = 0
	}

	agentIndex := agent.Y*cols + agent.X
	if err := setOneHot(dst, agentIndex, gridSize); err != nil {
		return fmt.Errorf("gridworldenv: agent position: %w", err)
	}

	goalIndex := goal.Y*cols + goal.X
	if err := setOneHot(dst, gridSize+goalIndex, 2*gridSize); err != nil {
		return fmt.Errorf("gridworldenv: goal position: %w", err)
	}
	return nil
}

func setOneHot(dst []float32, index, size int) error {
	if index < 0 || index >= size {
		return fmt.Errorf("index %d out of range [0, %d)", index, size)
	}
	dst[index] = 1.0
	return nil
}
