package main

import (
	"fmt"
	"math"
	"os"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"
)

// checkpointPrefix distinguishes cmd/train-ppo's -checkpoint-dir
// checkpoints from cmd/train's (see checkpoint.Path/checkpoint.Latest).
const checkpointPrefix = "ppo"

// resumeState is what resumeFromCheckpointDir found (or didn't) in a
// -checkpoint-dir: the loaded Params (nil if starting fresh) and the
// run-progress counters to continue from.
type resumeState struct {
	Params       *actorcritic.Params
	StartEpoch   int
	BestReturn   float32
	TotalUpdates int
}

// resumeFromCheckpointDir looks for the latest checkpoint.Path-named
// checkpoint in dir and loads it if found, returning a resumeState with
// Params nil (meaning "start fresh") if dir is empty or contains no
// matching checkpoint yet — both are normal, expected states (e.g. the
// very first run against a new -checkpoint-dir), not errors. It only
// returns an error if dir contains a checkpoint that fails to load
// (corrupt file, or an architecture/environment mismatch).
func resumeFromCheckpointDir(dir, environmentID string) (resumeState, error) {
	fresh := resumeState{BestReturn: float32(math.Inf(-1))}
	if dir == "" {
		return fresh, nil
	}

	latestPath, err := checkpoint.Latest(dir, checkpointPrefix)
	if err != nil {
		return fresh, nil
	}

	params, metadata, err := actorcritic.LoadFile(latestPath, environmentID)
	if err != nil {
		return resumeState{}, fmt.Errorf("resuming from %s: %w", latestPath, err)
	}

	return resumeState{
		Params:       params,
		StartEpoch:   metadata.Epoch + 1,
		BestReturn:   metadata.BestReturn,
		TotalUpdates: metadata.TotalUpdates,
	}, nil
}

// saveCheckpointToDir saves params (tagged with environmentID and the
// given run-progress counters) to dir under checkpoint.Path's naming
// convention for epoch, creating dir first if it doesn't exist yet.
func saveCheckpointToDir(dir string, params *actorcritic.Params, environmentID string, epoch int, bestReturn float32, totalUpdates int) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("creating checkpoint directory %s: %w", dir, err)
	}

	metadata := checkpoint.Metadata{Epoch: epoch, BestReturn: bestReturn, TotalUpdates: totalUpdates}
	return actorcritic.SaveFile(checkpoint.Path(dir, checkpointPrefix, epoch), params, environmentID, metadata)
}
