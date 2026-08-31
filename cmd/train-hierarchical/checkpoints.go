package main

import (
	"fmt"
	"math"
	"os"

	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"
	"github.com/reallyoldfogie/cRL-go/pkg/hierarchical"
)

// checkpointPrefix distinguishes cmd/train-hierarchical's
// -checkpoint-dir checkpoints from cmd/train's/cmd/train-ppo's (see
// checkpoint.Path/checkpoint.Latest).
const checkpointPrefix = "hierarchical"

// resumeState is what resumeFromCheckpointDir found (or didn't) in a
// -checkpoint-dir: the loaded meta-controller/sub-policy Initial params
// (nil meaning "start fresh") and the run-progress counters to continue
// from.
type resumeState struct {
	Initial      *hierarchical.InitialParams
	StartEpoch   int
	BestReturn   float32
	TotalUpdates int
}

// resumeFromCheckpointDir looks for the latest checkpoint.Path-named
// checkpoint in dir and loads it if found, returning a resumeState with
// Initial nil (meaning "start fresh") if dir is empty or contains no
// matching checkpoint yet — both are normal, expected states (e.g. the
// very first run against a new -checkpoint-dir), not errors. It only
// returns an error if dir contains a checkpoint that fails to load
// (corrupt file, or an environment/subgoal-count mismatch).
func resumeFromCheckpointDir(dir, environmentID string, numSubgoals int) (resumeState, error) {
	fresh := resumeState{BestReturn: float32(math.Inf(-1))}
	if dir == "" {
		return fresh, nil
	}

	latestPath, err := checkpoint.Latest(dir, checkpointPrefix)
	if err != nil {
		return fresh, nil
	}

	meta, subs, metadata, err := hierarchical.LoadFile(latestPath, environmentID, numSubgoals)
	if err != nil {
		return resumeState{}, fmt.Errorf("resuming from %s: %w", latestPath, err)
	}

	return resumeState{
		Initial:      &hierarchical.InitialParams{Meta: meta, Subs: subs},
		StartEpoch:   metadata.Epoch + 1,
		BestReturn:   metadata.BestReturn,
		TotalUpdates: metadata.TotalUpdates,
	}, nil
}

// saveCheckpointToDir saves trainer's meta-controller and every
// sub-policy's params (tagged with environmentID and the given
// run-progress counters) to dir under checkpoint.Path's naming
// convention for epoch, creating dir first if it doesn't exist yet.
func saveCheckpointToDir(dir string, trainer *hierarchical.Trainer, environmentID string, epoch int, bestReturn float32, totalUpdates int) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("creating checkpoint directory %s: %w", dir, err)
	}

	metadata := checkpoint.Metadata{Epoch: epoch, BestReturn: bestReturn, TotalUpdates: totalUpdates}
	return trainer.SaveFile(checkpoint.Path(dir, checkpointPrefix, epoch), environmentID, metadata)
}
