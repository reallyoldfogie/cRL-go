// Command watch loads a trained checkpoint and renders one episode of
// it acting in a chosen toy environment step by step in the terminal,
// for eyeballing whether a trained policy behaves sensibly rather than
// only reading its average-return numbers — see
// docs/plans/15-agent-and-training-visualization.md.
//
// It supports checkpoints saved by cmd/train (REINFORCE, pkg/policy)
// and cmd/train-ppo (PPO, pkg/actorcritic). pkg/hierarchical isn't
// supported yet: it has no single live-inference Actor driving both the
// meta-controller and the active sub-policy the way policy.Actor/
// actorcritic.Actor drive one flat network — see that plan's open
// questions.
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
	"github.com/reallyoldfogie/cRL-go/pkg/gridworldenv"
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
	envName := fs.String("env", "snake", "environment to watch: snake or gridworld")
	algo := fs.String("algo", "reinforce", "which algorithm the checkpoint was trained with: reinforce or ppo")
	gridSize := fs.Int("grid-size", 36, "grid size the checkpoint was trained against (must match the training run's -grid-size)")
	checkpointPath := fs.String("checkpoint", "", "path to a checkpoint saved by cmd/train or cmd/train-ppo (required)")
	episodeLen := fs.Int("episode-len", 100, "maximum number of steps to render before stopping")
	delay := fs.Duration("delay", 300*time.Millisecond, "how long to pause between rendered steps")
	seed := fs.Uint64("seed", 1, "seed for the environment's own randomness (e.g. food/goal placement) and action sampling")
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

	// environmentID mirrors cmd/train's/cmd/train-ppo's own convention
	// (e.g. "snake:36"), so a checkpoint trained against a different
	// -env/-grid-size is rejected outright rather than loaded into a
	// mismatched network shape.
	environmentID := fmt.Sprintf("%s:%d", *envName, *gridSize)

	act, err := newActFunc(*algo, *checkpointPath, environmentID)
	if err != nil {
		return err
	}

	ctx := context.Background()
	obs, err := env.Reset(ctx)
	if err != nil {
		return err
	}

	for step := 0; step < *episodeLen; step++ {
		renderStep(step, render())

		action, err := act(obs, rng)
		if err != nil {
			return err
		}

		result, err := env.Step(ctx, action)
		if err != nil {
			return err
		}
		fmt.Printf("action=%d reward=%.2f\n", action, result.Reward)
		time.Sleep(*delay)

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
	default:
		return nil, nil, fmt.Errorf("unknown -env %q, want \"snake\" or \"gridworld\"", envName)
	}
}

// actFunc samples one action from an observation, matching
// policy.Actor.Act's and actorcritic.Actor.Act's shared signature
// (mask is always nil here: watch mode has no legality constraints of
// its own to enforce).
type actFunc func(obs rl.Observation, rng *rand.Rand) (rl.Action, error)

// newActFunc loads the checkpoint at checkpointPath as the algorithm
// named by algo, returning an actFunc that samples from it.
func newActFunc(algo, checkpointPath, environmentID string) (actFunc, error) {
	switch algo {
	case "reinforce":
		params, _, err := policy.LoadFile(checkpointPath, environmentID)
		if err != nil {
			return nil, fmt.Errorf("loading checkpoint: %w", err)
		}
		actor, err := policy.NewActor(params)
		if err != nil {
			return nil, err
		}
		return func(obs rl.Observation, rng *rand.Rand) (rl.Action, error) {
			return actor.Act(obs, nil, rng)
		}, nil
	case "ppo":
		params, _, err := actorcritic.LoadFile(checkpointPath, environmentID)
		if err != nil {
			return nil, fmt.Errorf("loading checkpoint: %w", err)
		}
		actor, err := actorcritic.NewActor(params)
		if err != nil {
			return nil, err
		}
		return func(obs rl.Observation, rng *rand.Rand) (rl.Action, error) {
			return actor.Act(obs, nil, rng)
		}, nil
	default:
		return nil, fmt.Errorf("unknown -algo %q, want \"reinforce\" or \"ppo\"", algo)
	}
}
