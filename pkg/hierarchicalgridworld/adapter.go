package hierarchicalgridworld

import (
	"context"
	"fmt"

	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

// Adapter wraps an *Env to satisfy rl.Environment, translating between
// Env's Position/Action-based Reset/Step API (kept as-is so it can
// still be unit tested directly) and the generic
// Observation/Action/StepResult shapes pkg/reinforce/pkg/ppo/pkg/hierarchical
// train against.
type Adapter struct {
	env *Env
}

// NewAdapter wraps env as an rl.Environment.
func NewAdapter(env *Env) *Adapter {
	return &Adapter{env: env}
}

// ObservationSize implements rl.Environment.
func (a *Adapter) ObservationSize() int {
	return ObservationSize(a.env.GridSize)
}

// ActionSpace implements rl.Environment.
func (a *Adapter) ActionSpace() int {
	return NumActions
}

// Reset implements rl.Environment. ctx is accepted only to satisfy the
// interface; this toy environment never blocks or needs cancellation.
func (a *Adapter) Reset(ctx context.Context) (rl.Observation, error) {
	a.env.Reset()
	return a.observation()
}

// Step implements rl.Environment, translating action (an index into the
// policy's output layer) into the corresponding hierarchicalgridworld
// Action. ctx is accepted only to satisfy the interface, for the same
// reason as Reset.
func (a *Adapter) Step(ctx context.Context, action rl.Action) (rl.StepResult, error) {
	reward, done := a.env.Step(Action(action))

	observation, err := a.observation()
	if err != nil {
		return rl.StepResult{}, err
	}
	return rl.StepResult{Observation: observation, Reward: reward, Done: done}, nil
}

// observation builds the current observation vector. If the agent has
// just left the grid (Env.OutOfBounds), it returns a zero vector
// instead of erroring, mirroring pkg/snakeenv's/pkg/gridworldenv's
// Adapters: that terminal observation is never fed back into a policy
// (the caller stops on Done), it just needs to be validly shaped.
func (a *Adapter) observation() (rl.Observation, error) {
	values := make([]float32, ObservationSize(a.env.GridSize))
	if a.env.OutOfBounds() {
		return rl.Observation{Values: values}, nil
	}

	state := State{
		Agent:         a.env.Agent,
		BuildTarget:   a.env.BuildTarget,
		Resource:      a.env.Resource,
		Hazard:        a.env.Hazard,
		Mob:           a.env.Mob,
		MobActive:     a.env.MobActive,
		ResourcesHeld: a.env.ResourcesHeld,
		BuildProgress: a.env.BuildProgress,
		BuildGoal:     a.env.BuildGoal,
	}
	if err := BuildObservation(values, state, a.env.Cols, a.env.GridSize); err != nil {
		return rl.Observation{}, fmt.Errorf("hierarchicalgridworld: building observation: %w", err)
	}
	return rl.Observation{Values: values}, nil
}
