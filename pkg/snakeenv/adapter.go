package snakeenv

import (
	"context"
	"fmt"

	"github.com/reallyoldfogie/cRL-go/pkg/mat"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

// Adapter wraps an *Env to satisfy rl.Environment, translating between
// Env's Position/Action-based Reset/Step API (kept as-is so it can still
// be unit tested directly, see env_test.go) and the generic
// Observation/Action/StepResult shapes pkg/reinforce trains against.
type Adapter struct {
	env *Env
}

// NewAdapter wraps env as an rl.Environment.
func NewAdapter(env *Env) *Adapter {
	return &Adapter{env: env}
}

// ObservationSize implements rl.Environment.
func (a *Adapter) ObservationSize() int {
	return StateVectorSize(a.env.GridSize)
}

// ActionSpace implements rl.Environment.
func (a *Adapter) ActionSpace() int {
	return NumActions
}

// Reset implements rl.Environment. ctx is accepted only to satisfy the
// interface; snakeenv is a cheap, synchronous toy environment with
// nothing to cancel or time out.
func (a *Adapter) Reset(ctx context.Context) (rl.Observation, error) {
	a.env.Reset()
	return a.observation()
}

// Step implements rl.Environment, translating action (an index into the
// policy's output layer) into the corresponding snakeenv Action. ctx is
// accepted only to satisfy the interface, for the same reason as Reset.
func (a *Adapter) Step(ctx context.Context, action rl.Action) (rl.StepResult, error) {
	reward, done := a.env.Step(Action(action))

	observation, err := a.observation()
	if err != nil {
		return rl.StepResult{}, err
	}
	return rl.StepResult{Observation: observation, Reward: reward, Done: done}, nil
}

// observation builds the current one-hot state vector. If the snake has
// just left the grid (Env.GameOver), it returns a zero vector instead of
// erroring: BuildStateVector can't encode an out-of-bounds position, but
// that terminal observation is never fed back into the policy (the
// caller stops on Done), it just needs to be a validly shaped
// Observation.
func (a *Adapter) observation() (rl.Observation, error) {
	values := make([]float32, StateVectorSize(a.env.GridSize))
	if a.env.GameOver() {
		return rl.Observation{Values: values}, nil
	}

	stateVector := &mat.Matrix{Rows: len(values), Cols: 1, Data: values}
	if err := BuildStateVector(stateVector, a.env.Snake, a.env.Food, a.env.POV, a.env.Cols, a.env.GridSize); err != nil {
		return rl.Observation{}, fmt.Errorf("snakeenv: building observation: %w", err)
	}
	return rl.Observation{Values: stateVector.Data}, nil
}
