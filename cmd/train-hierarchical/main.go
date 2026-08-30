// Command train-hierarchical trains a two-level hierarchical PPO agent
// (a meta-controller choosing among coarse subgoals, plus one
// specialized sub-policy per subgoal) against pkg/hierarchicalgridworld,
// the only environment in this module with genuinely competing
// objectives — see docs/plans/11-hierarchical-meta-controller-and-subpolicies.md.
// It shares cmd/train-ppo's config file and PPO-hyperparameter flags,
// plus its own hierarchy-specific flags, printing per-epoch average
// return alongside the meta-controller's and each subgoal's update
// counts.
//
// Checkpointing is intentionally not supported here yet — see the
// plan's "Explicitly out of scope" section.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math/rand/v2"
	"os"

	"github.com/reallyoldfogie/cRL-go/pkg/config"
	"github.com/reallyoldfogie/cRL-go/pkg/hierarchical"
	"github.com/reallyoldfogie/cRL-go/pkg/hierarchicalgridworld"
	"github.com/reallyoldfogie/cRL-go/pkg/reinforce"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}

func run(args []string) error {
	fs := flag.NewFlagSet("train-hierarchical", flag.ContinueOnError)
	overrides := config.RegisterFlags(fs)

	// -hidden-size/-learning-rate (registered by RegisterFlags above)
	// are unused by this command: the meta-controller and sub-policies
	// each have their own hidden size/learning rate, set below.
	numSubgoals := fs.Int("num-subgoals", 4, "number of subgoals the meta-controller chooses among")
	subgoalInterval := fs.Int("subgoal-interval", 8, "environment steps between meta-controller decisions")
	metaHiddenSize := fs.Int("meta-hidden-size", 16, "meta-controller hidden layer width")
	subHiddenSize := fs.Int("sub-hidden-size", 16, "sub-policy hidden layer width")
	metaLearningRate := fs.Float64("meta-learning-rate", 0.003, "meta-controller Adam learning rate")
	subLearningRate := fs.Float64("sub-learning-rate", 0.003, "sub-policy Adam learning rate")
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

	cfg := hierarchical.Config{
		NumSubgoals:      *numSubgoals,
		SubgoalInterval:  *subgoalInterval,
		MetaHiddenSize:   *metaHiddenSize,
		SubHiddenSize:    *subHiddenSize,
		MetaLearningRate: float32(*metaLearningRate),
		SubLearningRate:  float32(*subLearningRate),
	}

	envFactory := reinforce.EnvFactory(func(rng *rand.Rand) (rl.Environment, error) {
		env, err := hierarchicalgridworld.New(settings.GridSize, rng)
		if err != nil {
			return nil, err
		}
		return hierarchicalgridworld.NewAdapter(env), nil
	})

	trainer, err := hierarchical.New(settings, cfg, envFactory)
	if err != nil {
		return err
	}

	// Wiring real cancellation (e.g. signal.NotifyContext) through this
	// command is out of scope for now; context.Background() is what
	// RunEpoch's ctx.Context parameter exists to support once a caller
	// needs it (e.g. a live environment that can block).
	ctx := context.Background()

	for epoch := range settings.Epochs {
		stats, err := trainer.RunEpoch(ctx, epoch)
		if err != nil {
			return err
		}

		fmt.Printf("Epoch %d | Average return: %.3f | Samples: %d | Meta updates: %d",
			stats.Epoch, stats.AverageReturn, stats.SampleCount, stats.MetaUpdateCount)
		for s := range cfg.NumSubgoals {
			subgoal := hierarchical.Subgoal(s)
			fmt.Printf(" | Sub[%d] updates: %d", s, stats.SubUpdateCounts[subgoal])
		}
		fmt.Println()
	}
	return nil
}
