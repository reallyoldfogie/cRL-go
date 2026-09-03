// Command watch loads a trained checkpoint and renders one episode of
// it acting in a chosen toy environment step by step in the terminal,
// for eyeballing whether a trained policy behaves sensibly rather than
// only reading its average-return numbers — see
// docs/plans/15-agent-and-training-visualization.md.
//
// It supports checkpoints saved by cmd/train (REINFORCE, pkg/policy),
// cmd/train-ppo (PPO, pkg/actorcritic), and cmd/train-hierarchical
// (hierarchical, pkg/hierarchical, via hierarchical.Actor).
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math/rand/v2"
	"os"
	"time"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/decisionlog"
	"github.com/reallyoldfogie/cRL-go/pkg/gridworldenv"
	"github.com/reallyoldfogie/cRL-go/pkg/hierarchical"
	"github.com/reallyoldfogie/cRL-go/pkg/hierarchicalgridworld"
	"github.com/reallyoldfogie/cRL-go/pkg/policy"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/reallyoldfogie/cRL-go/pkg/snakeenv"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}

func run(args []string) error {
	fs := flag.NewFlagSet("watch", flag.ContinueOnError)
	envName := fs.String("env", "snake", "environment to watch: snake, gridworld, or hierarchicalgridworld")
	algo := fs.String("algo", "reinforce", "which algorithm the checkpoint was trained with: reinforce, ppo, or hierarchical")
	gridSize := fs.Int("grid-size", 36, "grid size the checkpoint was trained against (must match the training run's -grid-size)")
	checkpointPath := fs.String("checkpoint", "", "path to a checkpoint saved by cmd/train, cmd/train-ppo, or cmd/train-hierarchical (required)")
	episodeLen := fs.Int("episode-len", 100, "maximum number of steps to render before stopping")
	delay := fs.Duration("delay", 300*time.Millisecond, "how long to pause between rendered steps")
	seed := fs.Uint64("seed", 1, "seed for the environment's own randomness (e.g. food/goal placement) and action sampling")
	numSubgoals := fs.Int("num-subgoals", 4, "number of subgoals the checkpoint's meta-controller chooses among (only used with -algo=hierarchical; must match the training run's -num-subgoals)")
	subgoalInterval := fs.Int("subgoal-interval", 8, "environment steps between meta-controller decisions (only used with -algo=hierarchical; must match the training run's -subgoal-interval)")
	decisionLogOut := fs.String("decision-log-out", "", "optional path to write one pkg/decisionlog.Record per step, as newline-delimited JSON, for later replay/analysis")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *checkpointPath == "" {
		return fmt.Errorf("-checkpoint is required (see -h)")
	}

	rng := rand.New(rand.NewPCG(*seed, *seed))
	env, render, err := newWatchEnv(*envName, *gridSize, rng)
	if err != nil {
		return err
	}

	// environmentID mirrors cmd/train's/cmd/train-ppo's/
	// cmd/train-hierarchical's own convention (e.g. "snake:36"), so a
	// checkpoint trained against a different -env/-grid-size is
	// rejected outright rather than loaded into a mismatched network
	// shape.
	environmentID := fmt.Sprintf("%s:%d", *envName, *gridSize)

	act, reset, err := newActFunc(*algo, *checkpointPath, environmentID, *numSubgoals, *subgoalInterval)
	if err != nil {
		return err
	}

	var decisionLog *decisionlog.FileWriter
	if *decisionLogOut != "" {
		decisionLog, err = decisionlog.NewFileWriter(*decisionLogOut)
		if err != nil {
			return err
		}
		defer decisionLog.Close()
	}

	ctx := context.Background()
	obs, err := env.Reset(ctx)
	if err != nil {
		return err
	}
	reset()

	for step := 0; step < *episodeLen; step++ {
		lines := render()
		renderStep(step, lines)

		decision, extra, extraFields, err := act(obs, rng)
		if err != nil {
			return err
		}

		result, err := env.Step(ctx, decision.Action)
		if err != nil {
			return err
		}
		printDecision(decision, extra, result.Reward)
		time.Sleep(*delay)

		if decisionLog != nil {
			if err := decisionLog.Write(decisionlog.Record{
				Step:        step,
				Observation: obs.Values,
				Decision:    decision,
				Reward:      result.Reward,
				Done:        result.Done,
				Render:      lines,
				Extra:       extraFields,
			}); err != nil {
				return err
			}
		}

		obs = result.Observation
		if result.Done {
			renderStep(step+1, render())
			fmt.Printf("Episode ended after %d step(s).\n", step+1)
			return nil
		}
	}
	fmt.Printf("Reached -episode-len=%d without the episode ending.\n", *episodeLen)
	return nil
}

// printDecision prints the chosen action, the probability it was
// sampled with (from rl.Decision.Probabilities, see
// docs/plans/16-decision-auditing-and-explainability.md), the critic's
// value estimate when the underlying Actor has one, any algorithm-
// specific extra context (e.g. hierarchical.Actor's active subgoal),
// and the reward the environment returned for taking it — a low
// chosen-action probability signals an undertrained or genuinely
// ambiguous decision point, not visible from the action/reward alone.
func printDecision(decision rl.Decision, extra string, reward float32) {
	fmt.Printf("action=%d prob=%.3f", decision.Action, decision.Probabilities[decision.Action])
	if decision.HasValue {
		fmt.Printf(" value=%.3f", decision.Value)
	}
	if extra != "" {
		fmt.Printf(" %s", extra)
	}
	fmt.Printf(" reward=%.2f\n", reward)
}

// renderStep clears the terminal and prints step's number followed by
// lines, a rendered environment snapshot (see pkg/gridrender).
func renderStep(step int, lines []string) {
	// ANSI "clear screen, move cursor to top-left" — a plain,
	// dependency-free way to redraw in place for this first, minimal
	// "watch mode" (see docs/plans/15-agent-and-training-visualization.md's
	// options for richer TUI treatments this could grow into later).
	fmt.Print("\033[H\033[2J")
	fmt.Printf("Step %d\n", step)
	for _, line := range lines {
		fmt.Println(line)
	}
}

// renderFunc returns a human-readable snapshot of an environment's
// current state, matching the Render() method every toy Env in this
// module now exposes.
type renderFunc func() []string

// newWatchEnv builds the rl.Environment and matching renderFunc for
// envName, both driving the same underlying *snakeenv.Env/*gridworldenv.Env
// instance: env.Reset/env.Step (via the returned rl.Environment) mutate
// the exact env whose Render method the returned renderFunc calls.
func newWatchEnv(envName string, gridSize int, rng *rand.Rand) (rl.Environment, renderFunc, error) {
	switch envName {
	case "snake":
		env, err := snakeenv.New(gridSize, rng)
		if err != nil {
			return nil, nil, err
		}
		return snakeenv.NewAdapter(env), env.Render, nil
	case "gridworld":
		env, err := gridworldenv.New(gridSize)
		if err != nil {
			return nil, nil, err
		}
		return gridworldenv.NewAdapter(env), env.Render, nil
	case "hierarchicalgridworld":
		env, err := hierarchicalgridworld.New(gridSize, rng)
		if err != nil {
			return nil, nil, err
		}
		return hierarchicalgridworld.NewAdapter(env), env.Render, nil
	default:
		return nil, nil, fmt.Errorf("unknown -env %q, want \"snake\", \"gridworld\", or \"hierarchicalgridworld\"", envName)
	}
}

// actFunc samples one decision from an observation, returning the
// rl.Decision that determined the environment action; an optional extra
// line of algorithm-specific context to print alongside it (empty for
// policy.Actor/actorcritic.Actor; hierarchical.Actor uses it to report
// the currently-active subgoal and whether a new meta-decision was just
// made); and the same context as structured data (nil where extra is
// empty) for decisionlog.Record.Extra, since extra's formatted string
// can't be losslessly recovered into structured data — see
// docs/plans/18-shared-decision-logging-format.md.
type actFunc func(obs rl.Observation, rng *rand.Rand) (decision rl.Decision, extra string, extraFields map[string]any, err error)

// newActFunc loads the checkpoint at checkpointPath as the algorithm
// named by algo, returning an actFunc that samples from it and a reset
// func to call once per episode, before the first actFunc call of that
// episode (a no-op for policy/actorcritic; hierarchical.Actor needs it
// to clear which subgoal is active, since it's the only stateful Actor
// here — see pkg/hierarchical/actor.go). Every case uses ActWithInfo
// rather than Act so watch mode can display why an action was chosen
// (its sampled probability, and a value estimate when available), not
// only which action was chosen — see
// docs/plans/16-decision-auditing-and-explainability.md.
func newActFunc(algo, checkpointPath, environmentID string, numSubgoals, subgoalInterval int) (actFunc, func(), error) {
	noopReset := func() {}

	switch algo {
	case "reinforce":
		params, _, err := policy.LoadFile(checkpointPath, environmentID)
		if err != nil {
			return nil, nil, fmt.Errorf("loading checkpoint: %w", err)
		}
		actor, err := policy.NewActor(params)
		if err != nil {
			return nil, nil, err
		}
		return func(obs rl.Observation, rng *rand.Rand) (rl.Decision, string, map[string]any, error) {
			decision, err := actor.ActWithInfo(obs, nil, rng)
			return decision, "", nil, err
		}, noopReset, nil
	case "ppo":
		params, _, err := actorcritic.LoadFile(checkpointPath, environmentID)
		if err != nil {
			return nil, nil, fmt.Errorf("loading checkpoint: %w", err)
		}
		actor, err := actorcritic.NewActor(params)
		if err != nil {
			return nil, nil, err
		}
		return func(obs rl.Observation, rng *rand.Rand) (rl.Decision, string, map[string]any, error) {
			decision, err := actor.ActWithInfo(obs, nil, rng)
			return decision, "", nil, err
		}, noopReset, nil
	case "hierarchical":
		meta, subs, _, err := hierarchical.LoadFile(checkpointPath, environmentID, numSubgoals)
		if err != nil {
			return nil, nil, fmt.Errorf("loading checkpoint: %w", err)
		}
		actor, err := hierarchical.NewActor(meta, subs, numSubgoals, subgoalInterval)
		if err != nil {
			return nil, nil, err
		}
		return func(obs rl.Observation, rng *rand.Rand) (rl.Decision, string, map[string]any, error) {
			decision, err := actor.ActWithInfo(obs, rng)
			if err != nil {
				return rl.Decision{}, "", nil, err
			}
			extra := fmt.Sprintf("subgoal=%d", decision.ActiveSubgoal)
			extraFields := map[string]any{"active_subgoal": int(decision.ActiveSubgoal)}
			if decision.MetaDecisionMade {
				extra += fmt.Sprintf(" (new, meta-prob=%.3f)", decision.MetaDecision.Probabilities[decision.MetaDecision.Action])
				extraFields["meta_decision_made"] = true
				extraFields["meta_decision"] = decision.MetaDecision
			}
			return decision.SubDecision, extra, extraFields, nil
		}, actor.Reset, nil
	default:
		return nil, nil, fmt.Errorf("unknown -algo %q, want \"reinforce\", \"ppo\", or \"hierarchical\"", algo)
	}
}
