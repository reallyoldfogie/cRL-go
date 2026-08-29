// Command train-ppo trains an actor-critic policy using PPO (Proximal
// Policy Optimization) against a choice of toy environments (see -env),
// printing per-epoch progress to stdout. It is the PPO counterpart to
// cmd/train's REINFORCE trainer, kept as a separate binary rather than a
// flag on cmd/train so the two trainers' hyperparameter surfaces don't
// have to share one flag set — see
// docs/plans/04-adam-optimizer-and-minibatch-trainer.md.
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand/v2"
	"os"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/config"
	"github.com/reallyoldfogie/cRL-go/pkg/gridworldenv"
	"github.com/reallyoldfogie/cRL-go/pkg/ppo"
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
	fs := flag.NewFlagSet("train-ppo", flag.ContinueOnError)
	overrides := config.RegisterFlags(fs)
	envName := fs.String("env", "snake", "environment to train against: snake or gridworld")
	checkpointIn := fs.String("checkpoint-in", "", "path to a checkpoint (see -checkpoint-out) to resume training from, instead of a fresh policy (optional)")
	checkpointOut := fs.String("checkpoint-out", "", "path to save the trained actor-critic weights to after training completes (optional)")
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

	// environmentID identifies the environment/action-space a checkpoint
	// was trained against (e.g. "snake:36"), so a checkpoint saved for
	// one -env/-grid-size combination can't be silently loaded into an
	// incompatible one (see actorcritic.Load).
	environmentID := fmt.Sprintf("%s:%d", *envName, settings.GridSize)

	initialParams, err := loadInitialParams(*checkpointIn, environmentID)
	if err != nil {
		return err
	}

	trainer, err := ppo.New(settings, envFactory, initialParams)
	if err != nil {
		return err
	}

	for epoch := range settings.Epochs {
		stats, err := trainer.RunEpoch(epoch)
		if err != nil {
			return err
		}

		fmt.Printf(
			"Epoch %d | Average return: %.3f | Samples: %d\n",
			stats.Epoch, stats.AverageReturn, stats.SampleCount,
		)
	}

	if *checkpointOut != "" {
		if err := actorcritic.SaveFile(*checkpointOut, trainer.Params(), environmentID); err != nil {
			return fmt.Errorf("saving checkpoint: %w", err)
		}
	}
	return nil
}

// loadInitialParams loads the checkpoint at checkpointPath if one was
// given, returning nil (meaning "initialize fresh") if checkpointPath is
// empty. expectedEnvironmentID is validated against the checkpoint's own
// saved environment ID (see actorcritic.Load).
func loadInitialParams(checkpointPath, expectedEnvironmentID string) (*actorcritic.Params, error) {
	if checkpointPath == "" {
		return nil, nil
	}

	params, err := actorcritic.LoadFile(checkpointPath, expectedEnvironmentID)
	if err != nil {
		return nil, fmt.Errorf("loading checkpoint: %w", err)
	}
	return params, nil
}

// newEnvFactory returns the reinforce.EnvFactory for the environment
// selected by -env, both of which are sized by gridSize. Duplicated from
// cmd/train/main.go rather than shared: each cmd/ binary is a complete,
// independent main package, and this is a handful of lines.
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
