// Command train reproduces the original cRL quickstart: it trains a small
// policy network using REINFORCE against a choice of toy environments
// (see -env), printing per-epoch progress to stdout.
//
// See docs/ for an explanation of the underlying ML concepts, and
// docs/05-porting-notes.md for how this Go port differs from
// github.com/harshbhatt7585/cRL's original C implementation.
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand/v2"
	"os"

	"github.com/reallyoldfogie/cRL-go/pkg/config"
	"github.com/reallyoldfogie/cRL-go/pkg/gridworldenv"
	"github.com/reallyoldfogie/cRL-go/pkg/policy"
	"github.com/reallyoldfogie/cRL-go/pkg/reinforce"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/reallyoldfogie/cRL-go/pkg/snakeenv"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}

func run(args []string) error {
	fs := flag.NewFlagSet("train", flag.ContinueOnError)
	overrides := config.RegisterFlags(fs)
	envName := fs.String("env", "snake", "environment to train against: snake or gridworld")
	checkpointIn := fs.String("checkpoint-in", "", "path to a checkpoint (see -checkpoint-out) to resume training from, instead of a fresh policy (optional)")
	checkpointOut := fs.String("checkpoint-out", "", "path to save the trained policy weights to after training completes (optional)")
	if err := fs.Parse(args); err != nil {
		return err
	}

	settings, err := config.Load(overrides.ConfigPath)
	if err != nil {
		return err
	}
	config.Apply(&settings, fs, overrides)

	if err := settings.Validate(); err != nil {
		return err
	}

	envFactory, err := newEnvFactory(*envName, settings.GridSize)
	if err != nil {
		return err
	}

	initialParams, err := loadInitialParams(*checkpointIn)
	if err != nil {
		return err
	}

	trainer, err := reinforce.New(settings, envFactory, initialParams)
	if err != nil {
		return err
	}

	for epoch := range settings.Epochs {
		stats, err := trainer.RunEpoch(epoch)
		if err != nil {
			return err
		}

		fmt.Printf(
			"Epoch %d | Average return: %.3f | Samples: %d | Return std: %.3f\n",
			stats.Epoch, stats.AverageReturn, stats.SampleCount, stats.ReturnStd,
		)
	}

	if *checkpointOut != "" {
		if err := policy.SaveFile(*checkpointOut, trainer.Params()); err != nil {
			return fmt.Errorf("saving checkpoint: %w", err)
		}
	}
	return nil
}

// loadInitialParams loads the checkpoint at checkpointPath if one was
// given, returning nil (meaning "initialize fresh") if checkpointPath is
// empty.
func loadInitialParams(checkpointPath string) (*policy.Params, error) {
	if checkpointPath == "" {
		return nil, nil
	}

	params, err := policy.LoadFile(checkpointPath)
	if err != nil {
		return nil, fmt.Errorf("loading checkpoint: %w", err)
	}
	return params, nil
}

// newEnvFactory returns the reinforce.EnvFactory for the environment
// selected by -env, both of which are sized by gridSize.
func newEnvFactory(envName string, gridSize int) (reinforce.EnvFactory, error) {
	switch envName {
	case "snake":
		return func(rng *rand.Rand) (rl.Environment, error) {
			env, err := snakeenv.New(gridSize, rng)
			if err != nil {
				return nil, err
			}
			return snakeenv.NewAdapter(env), nil
		}, nil
	case "gridworld":
		return func(rng *rand.Rand) (rl.Environment, error) {
			env, err := gridworldenv.New(gridSize)
			if err != nil {
				return nil, err
			}
			return gridworldenv.NewAdapter(env), nil
		}, nil
	default:
		return nil, fmt.Errorf("unknown -env %q, want \"snake\" or \"gridworld\"", envName)
	}
}
