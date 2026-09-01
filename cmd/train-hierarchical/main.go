// Command train-hierarchical trains a two-level hierarchical PPO agent
// (a meta-controller choosing among coarse subgoals, plus one
// specialized sub-policy per subgoal) against pkg/hierarchicalgridworld,
// the only environment in this module with genuinely competing
// objectives — see docs/plans/11-hierarchical-meta-controller-and-subpolicies.md.
// It shares cmd/train-ppo's config file and PPO-hyperparameter flags,
// plus its own hierarchy-specific flags, printing per-epoch average
// return alongside the meta-controller's and each subgoal's update
// counts. It also shares cmd/train-ppo's checkpoint flag conventions
// (-checkpoint-in/-checkpoint-out, -checkpoint-dir/-checkpoint-interval)
// — see checkpoints.go and docs/archive/plans/14-hierarchical-checkpointing.md.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math/rand/v2"
	"os"

	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"
	"github.com/reallyoldfogie/cRL-go/pkg/config"
	"github.com/reallyoldfogie/cRL-go/pkg/hierarchical"
	"github.com/reallyoldfogie/cRL-go/pkg/hierarchicalgridworld"
	"github.com/reallyoldfogie/cRL-go/pkg/metrics"
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
	checkpointIn := fs.String("checkpoint-in", "", "path to a checkpoint (see -checkpoint-out) to resume training from, instead of fresh meta-controller/sub-policies (optional)")
	checkpointOut := fs.String("checkpoint-out", "", "path to save the trained meta-controller and sub-policy weights to after training completes (optional)")
	checkpointDir := fs.String("checkpoint-dir", "", "directory to auto-resume the latest checkpoint from, and periodically save numbered checkpoints into (optional; independent of -checkpoint-in/-checkpoint-out)")
	checkpointInterval := fs.Int("checkpoint-interval", 50, "save a checkpoint to -checkpoint-dir every N epochs (only used if -checkpoint-dir is set)")
	metricsOut := fs.String("metrics-out", "", "path to write one CSV row of per-epoch metrics to (optional; see docs/plans/15-agent-and-training-visualization.md)")
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

	// environmentID identifies the environment this checkpoint was
	// trained against (e.g. "hierarchicalgridworld:36"), mirroring
	// cmd/train-ppo's convention (see hierarchical.Load). NumSubgoals is
	// validated separately by hierarchical.Load, since it determines
	// every sub-policy's shape independent of the environment itself.
	environmentID := fmt.Sprintf("hierarchicalgridworld:%d", settings.GridSize)

	resume, err := resumeFromCheckpointDir(*checkpointDir, environmentID, cfg.NumSubgoals)
	if err != nil {
		return err
	}

	initial := resume.Initial
	if initial != nil && *checkpointIn != "" {
		return fmt.Errorf("both -checkpoint-in and an existing checkpoint in -checkpoint-dir were found; use only one to resume from")
	}
	if initial == nil {
		initial, err = loadInitialParams(*checkpointIn, environmentID, cfg.NumSubgoals)
		if err != nil {
			return err
		}
	}

	envFactory := reinforce.EnvFactory(func(rng *rand.Rand) (rl.Environment, error) {
		env, err := hierarchicalgridworld.New(settings.GridSize, rng)
		if err != nil {
			return nil, err
		}
		return hierarchicalgridworld.NewAdapter(env), nil
	})

	trainer, err := hierarchical.New(settings, cfg, envFactory, initial)
	if err != nil {
		return err
	}

	var metricsWriter *metrics.CSVWriter
	if *metricsOut != "" {
		header := []string{"epoch", "average_return", "sample_count", "meta_update_count"}
		for s := range cfg.NumSubgoals {
			header = append(header, fmt.Sprintf("sub_%d_updates", s))
		}
		metricsWriter, err = metrics.NewCSVWriter(*metricsOut, header)
		if err != nil {
			return err
		}
	}

	// Wiring real cancellation (e.g. signal.NotifyContext) through this
	// command is out of scope for now; context.Background() is what
	// RunEpoch's ctx.Context parameter exists to support once a caller
	// needs it (e.g. a live environment that can block).
	ctx := context.Background()

	bestReturn := resume.BestReturn
	totalUpdates := resume.TotalUpdates

	for epoch := resume.StartEpoch; epoch < settings.Epochs; epoch++ {
		stats, err := trainer.RunEpoch(ctx, epoch)
		if err != nil {
			return err
		}

		epochUpdates := stats.MetaUpdateCount
		for _, count := range stats.SubUpdateCounts {
			epochUpdates += count
		}
		totalUpdates += epochUpdates
		if stats.AverageReturn > bestReturn {
			bestReturn = stats.AverageReturn
		}

		fmt.Printf("Epoch %d | Average return: %.3f | Samples: %d | Meta updates: %d",
			stats.Epoch, stats.AverageReturn, stats.SampleCount, stats.MetaUpdateCount)
		for s := range cfg.NumSubgoals {
			subgoal := hierarchical.Subgoal(s)
			fmt.Printf(" | Sub[%d] updates: %d", s, stats.SubUpdateCounts[subgoal])
		}
		fmt.Println()

		if metricsWriter != nil {
			row := []any{stats.Epoch, stats.AverageReturn, stats.SampleCount, stats.MetaUpdateCount}
			for s := range cfg.NumSubgoals {
				row = append(row, stats.SubUpdateCounts[hierarchical.Subgoal(s)])
			}
			if err := metricsWriter.WriteRow(row...); err != nil {
				return err
			}
		}

		interval := *checkpointInterval
		if *checkpointDir != "" && interval > 0 && (epoch+1)%interval == 0 {
			if err := saveCheckpointToDir(*checkpointDir, trainer, environmentID, epoch, bestReturn, totalUpdates); err != nil {
				return fmt.Errorf("saving periodic checkpoint: %w", err)
			}
		}
	}

	if metricsWriter != nil {
		if err := metricsWriter.Close(); err != nil {
			return err
		}
	}

	if *checkpointDir != "" && settings.Epochs > resume.StartEpoch {
		if err := saveCheckpointToDir(*checkpointDir, trainer, environmentID, settings.Epochs-1, bestReturn, totalUpdates); err != nil {
			return fmt.Errorf("saving final checkpoint: %w", err)
		}
	}

	if *checkpointOut != "" {
		metadata := checkpoint.Metadata{Epoch: settings.Epochs - 1, BestReturn: bestReturn, TotalUpdates: totalUpdates}
		if err := trainer.SaveFile(*checkpointOut, environmentID, metadata); err != nil {
			return fmt.Errorf("saving checkpoint: %w", err)
		}
	}
	return nil
}

// loadInitialParams loads the checkpoint at checkpointPath if one was
// given, returning nil (meaning "initialize fresh") if checkpointPath is
// empty. Its metadata is discarded: -checkpoint-in is the "manual
// single-file" resume mode, which has always restarted epoch numbering
// from 0; -checkpoint-dir (see checkpoints.go) is the mode that
// continues it.
func loadInitialParams(checkpointPath, expectedEnvironmentID string, expectedNumSubgoals int) (*hierarchical.InitialParams, error) {
	if checkpointPath == "" {
		return nil, nil
	}

	meta, subs, _, err := hierarchical.LoadFile(checkpointPath, expectedEnvironmentID, expectedNumSubgoals)
	if err != nil {
		return nil, fmt.Errorf("loading checkpoint: %w", err)
	}
	return &hierarchical.InitialParams{Meta: meta, Subs: subs}, nil
}
