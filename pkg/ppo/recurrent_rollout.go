package ppo

import (
	"context"
	"fmt"
	"math/rand/v2"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/reinforce"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

// RecurrentRollout is Rollout (see rollout.go) plus the LSTM hidden/cell
// state that was fed into each step — HPrevs[t]/CPrevs[t] is the state
// the policy's recurrent trunk saw as input when it produced
// Episode.Transitions[t] (the zero vector at t=0, i.e. episode start).
//
// Recording only the *incoming* state at rollout time (not replaying an
// episode's full prefix at training time) is the state-representation
// strategy docs/plans/20-configurable-depth-step-memory-and-dynamic-architecture.md
// Part B chose: a training-time chunk seeds its unrolled graph's starting
// state from whichever one of these was recorded at that chunk's first
// step, and every later step within the same chunk recomputes its own
// state live during that chunk's own forward pass — so only the first
// step of each chunk actually needs its recorded state read back; every
// other entry here exists so a chunk boundary can start at any step
// index, not just ones a rollout collector happened to pre-align to a
// particular chunk length.
type RecurrentRollout struct {
	*Rollout
	HPrevs [][]float32
	CPrevs [][]float32
}

// collectRecurrentTrajectory is collectTrajectory's recurrent
// counterpart: same shape, driving a actorcritic.RecurrentInferenceNetwork
// instead of InferenceNetwork, carrying (h, c) forward step to step
// (reset to zero at Reset), and recording each step's *incoming* (h, c)
// onto the result (see RecurrentRollout's own doc comment for why
// incoming, not outgoing, state is what's recorded).
func collectRecurrentTrajectory(ctx context.Context, params *actorcritic.RecurrentParams, envFactory reinforce.EnvFactory, episodeLen int, rng *rand.Rand) (*RecurrentRollout, error) {
	env, err := envFactory(rng)
	if err != nil {
		return nil, fmt.Errorf("ppo: creating environment: %w", err)
	}
	return collectRecurrentTrajectoryFromEnv(ctx, params, env, episodeLen, rng)
}

// collectRecurrentTrajectoryFromEnv is collectRecurrentTrajectory's
// shared core, mirroring collectTrajectoryFromEnv's own split for the
// same reason (see its doc comment): it runs one episode against an
// already-constructed env, so it can back both collectRecurrentTrajectory
// and a future persistent-environment variant (not built in this pass —
// see docs/plans/20's own Scope).
func collectRecurrentTrajectoryFromEnv(ctx context.Context, params *actorcritic.RecurrentParams, env rl.Environment, episodeLen int, rng *rand.Rand) (*RecurrentRollout, error) {
	net, err := actorcritic.NewRecurrentInferenceNetwork(params)
	if err != nil {
		return nil, fmt.Errorf("ppo: building recurrent inference network: %w", err)
	}

	observation, err := env.Reset(ctx)
	if err != nil {
		return nil, fmt.Errorf("ppo: resetting environment: %w", err)
	}

	episode := &rl.Episode{Transitions: make([]rl.Transition, 0, episodeLen)}
	logProbs := make([]float32, 0, episodeLen)
	values := make([]float32, 0, episodeLen)
	hPrevs := make([][]float32, 0, episodeLen)
	cPrevs := make([][]float32, 0, episodeLen)

	// h, c start zeroed (episode start); net.HPrevIn/CPrevIn are already
	// zero-valued from NewRecurrentInferenceNetwork, so nothing to reset
	// explicitly before the first step.
	for range episodeLen {
		hPrev := append([]float32(nil), net.HPrevIn.Val.Data...)
		cPrev := append([]float32(nil), net.CPrevIn.Val.Data...)

		copy(net.Input.Val.Data, observation.Values)
		net.Graph.Forward()

		var mask []bool
		if masker, ok := env.(rl.ActionMasker); ok {
			mask = masker.ActionMask()
		}

		action, err := reinforce.SampleMaskedAction(net.PolicyOutput.Val, mask, rng)
		if err != nil {
			return nil, fmt.Errorf("ppo: sampling action: %w", err)
		}
		logProbs = append(logProbs, actionLogProb(net.PolicyOutput.Val, action))
		values = append(values, net.ValueOutput.Val.Data[0])
		hPrevs = append(hPrevs, hPrev)
		cPrevs = append(cPrevs, cPrev)

		result, err := env.Step(ctx, action)
		if err != nil {
			return nil, fmt.Errorf("ppo: stepping environment: %w", err)
		}

		episode.Transitions = append(episode.Transitions, rl.Transition{
			Observation: observation,
			Action:      action,
			Reward:      result.Reward,
			Done:        result.Done,
			Mask:        mask,
		})

		// Carry this step's output state into the next step's input.
		copy(net.HPrevIn.Val.Data, net.HOut.Val.Data)
		copy(net.CPrevIn.Val.Data, net.COut.Val.Data)

		observation = result.Observation
		if result.Done {
			break
		}
	}

	return &RecurrentRollout{
		Rollout: &Rollout{Episode: episode, LogProbs: logProbs, Values: values},
		HPrevs:  hPrevs,
		CPrevs:  cPrevs,
	}, nil
}
