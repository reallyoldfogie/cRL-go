// Package snakeenv implements a small grid-foraging environment: a single
// point-agent moves around a square grid trying to reach a food cell.
//
// Despite the name (carried over from the original C project, which called
// it "Snake"), there is no snake body, tail, or self-collision here — only
// a single position that dies on hitting the grid boundary. See
// docs/03-policy-gradients-and-reinforce.md for how this environment is
// used to train a policy, and docs/05-porting-notes.md for why the naming
// was kept despite being a bit misleading.
//
// This is a reimplementation of the SnakeENV type and its functions from
// env.c in github.com/harshbhatt7585/cRL.
package snakeenv

// Action is one of the five moves the agent can take.
type Action int32

const (
	ActionLeft Action = iota
	ActionRight
	ActionUp
	ActionDown
	// ActionNone means "keep moving in the current direction" (the
	// agent's last non-None action, tracked as Env.POV). It's a valid,
	// frequently-useful action in its own right, not an error case.
	ActionNone

	// NumActions is the number of distinct actions, and therefore the
	// size of the policy network's output layer.
	NumActions = int(ActionNone) + 1
)

func (a Action) String() string {
	switch a {
	case ActionLeft:
		return "left"
	case ActionRight:
		return "right"
	case ActionUp:
		return "up"
	case ActionDown:
		return "down"
	case ActionNone:
		return "none"
	default:
		return "invalid"
	}
}
