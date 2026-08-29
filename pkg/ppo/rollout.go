package ppo

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/mat"
	"github.com/reallyoldfogie/cRL-go/pkg/reinforce"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

// logProbEpsilon is the smallest probability value used before taking a
// log, matching pkg/mat's probEpsilon so a near-zero action probability
// never produces a -Inf/NaN log-probability.
const logProbEpsilon float32 = 1e-8

// collectTrajectory runs one episode (up to episodeLen steps, ending
// early if the environment reports Done) using an environment built by
// envFactory and a fresh actorcritic.InferenceNetwork built over params,
// both private to this call. rng drives both envFactory and action
// sampling, via reinforce.SampleAction (reused as-is: sampling a
// category from a probability vector has no algorithm-specific
// behavior of its own).
//
// Unlike pkg/reinforce's collectTrajectory, this also records, at every
// step, the sampled action's log-probability and the value head's
// estimate — both required by computeGAE and the PPO loss (see
// gae.go, loss.go) — as a Rollout alongside the plain rl.Episode.
func collectTrajectory(params *actorcritic.Params, envFactory reinforce.EnvFactory, episodeLen int, rng *rand.Rand) (*Rollout, error) {
	env, err := envFactory(rng)
	if err != nil {
		return nil, fmt.Errorf("ppo: creating environment: %w", err)
	}

	net, err := actorcritic.NewInferenceNetwork(params)
	if err != nil {
		return nil, fmt.Errorf("ppo: building inference network: %w", err)
	}

	observation, err := env.Reset()
	if err != nil {
		return nil, fmt.Errorf("ppo: resetting environment: %w", err)
	}

	episode := &rl.Episode{Transitions: make([]rl.Transition, 0, episodeLen)}
	logProbs := make([]float32, 0, episodeLen)
	values := make([]float32, 0, episodeLen)

	for range episodeLen {
		copy(net.Input.Val.Data, observation.Values)
		net.Graph.Forward()

		action := reinforce.SampleAction(net.PolicyOutput.Val, rng)
		logProbs = append(logProbs, actionLogProb(net.PolicyOutput.Val, action))
		values = append(values, net.ValueOutput.Val.Data[0])

		result, err := env.Step(action)
		if err != nil {
			return nil, fmt.Errorf("ppo: stepping environment: %w", err)
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

	return &Rollout{Episode: episode, LogProbs: logProbs, Values: values}, nil
}

// actionLogProb returns log(max(probs[action], logProbEpsilon)).
func actionLogProb(probs *mat.Matrix, action rl.Action) float32 {
	probability := probs.Data[action]
	if probability < logProbEpsilon {
		probability = logProbEpsilon
	}
	return float32(math.Log(float64(probability)))
}
