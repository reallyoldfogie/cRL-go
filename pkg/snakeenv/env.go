package snakeenv

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/reallyoldfogie/cRL-go/pkg/mat"
)

const (
	foodReward         float32 = 20.0
	outOfBoundsPenalty float32 = 10.0
	initialFoodX               = 5
	initialFoodY               = 5
)

// Position is a location on the grid.
type Position struct {
	X, Y int32
}

// Env is a square-grid foraging environment: Snake is the agent's
// position, and reaching Food earns a reward and relocates the food to a
// new random cell. Leaving the grid ends the episode.
//
// Each Env owns its own random source (rng), rather than reading from a
// shared/global generator. This is a deliberate departure from the
// original C environment (which mixed a global PCG stream for weight
// init/action sampling with a separate, unseeded libc RNG for food
// placement — see docs/05-porting-notes.md). Owning its own *rand.Rand
// also makes it safe to run many Envs concurrently, each with an
// independently-seeded generator, for parallel rollout collection.
type Env struct {
	rng *rand.Rand

	Rows, Cols int
	GridSize   int

	Snake      Position
	Food       Position
	Score      float32
	FoodsEaten int
	POV        Action
	Steps      uint64
}

// Side returns the side length of a square grid with gridSize cells,
// reporting an error if gridSize is not a perfect square. Exposed so
// callers (e.g. pkg/reinforce, when replaying stored trajectory data
// without a live Env) can compute grid dimensions without constructing an
// Env.
func Side(gridSize int) (int, error) {
	side := int(math.Sqrt(float64(gridSize)))
	if side*side != gridSize {
		return 0, fmt.Errorf("snakeenv: gridSize %d is not a perfect square", gridSize)
	}
	return side, nil
}

// New creates an Env over a gridSize-cell square grid (gridSize must be a
// perfect square, e.g. 36 for a 6x6 grid) using rng for all randomness
// (food placement). The environment starts in a reset state.
func New(gridSize int, rng *rand.Rand) (*Env, error) {
	side, err := Side(gridSize)
	if err != nil {
		return nil, err
	}

	env := &Env{
		rng:      rng,
		Rows:     side,
		Cols:     side,
		GridSize: gridSize,
		Snake:    Position{X: 0, Y: 0},
		Food:     Position{X: initialFoodX, Y: initialFoodY},
		POV:      ActionRight,
	}
	return env, nil
}

// StateVectorSize returns the length of the one-hot state vector produced
// by BuildStateVector for a grid of the given size: one segment for the
// agent's position, one for the food's position, and one for the current
// direction of travel. This is computed from gridSize rather than hardcoded
// (the original C model.c hardcoded 76, tightly and silently coupled to a
// grid size of 36 — see docs/05-porting-notes.md).
func StateVectorSize(gridSize int) int {
	return 2*gridSize + NumActions
}

// randomFoodLocation returns a uniformly random cell on the grid.
func (e *Env) randomFoodLocation() Position {
	return Position{
		X: int32(e.rng.IntN(e.Cols)),
		Y: int32(e.rng.IntN(e.Rows)),
	}
}

// randomNonOverlappingFoodLocation returns a random cell that isn't
// currently occupied by the snake.
func (e *Env) randomNonOverlappingFoodLocation() Position {
	for {
		food := e.randomFoodLocation()
		if food != e.Snake {
			return food
		}
	}
}

// GameOver reports whether the agent has left the grid.
func (e *Env) GameOver() bool {
	return e.Snake.X < 0 || e.Snake.X >= int32(e.Cols) ||
		e.Snake.Y < 0 || e.Snake.Y >= int32(e.Rows)
}

// Reset places the agent at the center of the grid and the food at a
// random non-overlapping cell, and clears episode bookkeeping.
func (e *Env) Reset() {
	e.Snake = Position{X: int32(e.Cols / 2), Y: int32(e.Rows / 2)}
	e.Food = e.randomNonOverlappingFoodLocation()
	e.Score = 0
	e.FoodsEaten = 0
	e.POV = ActionRight
	e.Steps = 0
}

// Step applies action (resolving ActionNone to the current direction of
// travel), updates the agent's position, and returns the resulting reward
// and whether the episode has ended.
//
// This consolidates what the original C code split across take_action,
// get_reward, and inline bookkeeping in the training loop (env.c) into a
// single method, matching the common step(action) -> (reward, done)
// shape used by most RL environment APIs.
func (e *Env) Step(action Action) (reward float32, done bool) {
	if action == ActionNone {
		action = e.POV
	}

	switch action {
	case ActionLeft:
		e.Snake.X--
	case ActionRight:
		e.Snake.X++
	case ActionUp:
		e.Snake.Y++
	case ActionDown:
		e.Snake.Y--
	}
	e.POV = action

	if e.Snake == e.Food {
		reward += foodReward
		e.FoodsEaten++
		e.Food = e.randomNonOverlappingFoodLocation()
	}

	done = e.GameOver()
	if done {
		reward -= outOfBoundsPenalty
	}

	e.Score += reward
	e.Steps++
	return reward, done
}

// BuildStateVector fills dst with a one-hot encoding of the given state:
// a 1 at the snake's grid cell, a 1 at (gridSize + the food's grid cell),
// and a 1 at (2*gridSize + pov). dst must have length StateVectorSize(gridSize).
func BuildStateVector(dst *mat.Matrix, snake, food Position, pov Action, cols, gridSize int) error {
	dst.Clear()

	snakeIndex := int(snake.Y)*cols + int(snake.X)
	foodIndex := int(food.Y)*cols + int(food.X)

	if err := setOneHot(dst, snakeIndex); err != nil {
		return fmt.Errorf("snakeenv: snake position: %w", err)
	}
	if err := setOneHot(dst, gridSize+foodIndex); err != nil {
		return fmt.Errorf("snakeenv: food position: %w", err)
	}
	if err := setOneHot(dst, 2*gridSize+int(pov)); err != nil {
		return fmt.Errorf("snakeenv: pov: %w", err)
	}
	return nil
}

func setOneHot(dst *mat.Matrix, index int) error {
	if index < 0 || index >= len(dst.Data) {
		return fmt.Errorf("index %d out of range [0, %d)", index, len(dst.Data))
	}
	dst.Data[index] = 1.0
	return nil
}
