package reinforce

import (
	"fmt"
	"math/rand/v2"

	"github.com/reallyoldfogie/cRL-go/pkg/policy"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

// EnvFactory builds a fresh rl.Environment for one rollout, using rng for
// whatever randomness that environment needs at construction time (e.g.
// snakeenv's food placement). Environments are not required to use rng
// at all (e.g. gridworldenv's fixed layout ignores it).
type EnvFactory func(rng *rand.Rand) (rl.Environment, error)

// collectTrajectory runs one episode (up to episodeLen steps, ending
// early if the environment reports Done) using an environment built by
// envFactory and a fresh policy.InferenceNetwork built over params, both
// private to this call. rng drives both envFactory and action sampling.
//
// Because the InferenceNetwork's weights alias params (see
// policy.NewInferenceNetwork) rather than copying them, and every other
// piece of state here (the environment, rng, the network's activation
// buffers) is freshly allocated, this function is safe to call
// concurrently from multiple goroutines against the same params, as long
// as no other goroutine is concurrently mutating params (e.g. via an SGD
// step) and envFactory's environments are themselves safe to construct
// concurrently (true for the cheap, stateless toy environments in this
// module; not a general guarantee for every rl.Environment).
func collectTrajectory(params *policy.Params, envFactory EnvFactory, episodeLen int, rng *rand.Rand) (*rl.Episode, error) {
	env, err := envFactory(rng)
	if err != nil {
		return nil, fmt.Errorf("reinforce: creating environment: %w", err)
	}

	actor, err := policy.NewActor(params)
	if err != nil {
		return nil, fmt.Errorf("reinforce: building actor: %w", err)
	}

	observation, err := env.Reset()
	if err != nil {
		return nil, fmt.Errorf("reinforce: resetting environment: %w", err)
	}

	episode := &rl.Episode{Transitions: make([]rl.Transition, 0, episodeLen)}

	for range episodeLen {
		action, err := actor.Act(observation, nil, rng)
		if err != nil {
			return nil, fmt.Errorf("reinforce: sampling action: %w", err)
		}

		result, err := env.Step(action)
		if err != nil {
			return nil, fmt.Errorf("reinforce: stepping environment: %w", err)
		}

		episode.Transitions = append(episode.Transitions, rl.Transition{
			Observation: observation,
			Action:      action,
			Reward:      result.Reward,
			Done:        result.Done,
		})

		observation = result.Observation
		if result.Done {
			break
		}
	}

	return episode, nil
}
