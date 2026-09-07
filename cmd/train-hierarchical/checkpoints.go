package main

import (
	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"
	"github.com/reallyoldfogie/cRL-go/pkg/hierarchical"
)

// checkpointPrefix distinguishes cmd/train-hierarchical's
// -checkpoint-dir checkpoints from cmd/train's/cmd/train-ppo's (see
// checkpoint.Path/checkpoint.Latest).
const checkpointPrefix = "hierarchical"

// resumeState is what resumeFromCheckpointDir found (or didn't) in a
// -checkpoint-dir: the loaded meta-controller/sub-policy Params (nil if
// starting fresh) and the run-progress counters to continue from.
type resumeState = checkpoint.Resumed[*hierarchical.InitialParams]

// resumeFromCheckpointDir looks for the latest checkpoint in dir and
// loads it if found; see checkpoint.Resume for the fresh-start/error
// contract. hierarchical.LoadFile's extra numSubgoals argument and its
// two-value (meta, subs) return, rather than one params value, are why
// this needs a closure instead of passing hierarchical.LoadFile
// directly, unlike cmd/train's and cmd/train-ppo's equivalents.
func resumeFromCheckpointDir(dir, environmentID string, numSubgoals int) (resumeState, error) {
	return checkpoint.Resume(dir, checkpointPrefix, func(path string) (*hierarchical.InitialParams, checkpoint.Metadata, error) {
		meta, subs, metadata, err := hierarchical.LoadFile(path, environmentID, numSubgoals)
		if err != nil {
			return nil, checkpoint.Metadata{}, err
		}
		return &hierarchical.InitialParams{Meta: meta, Subs: subs}, metadata, nil
	})
}

// saveCheckpointToDir saves trainer's meta-controller and every
// sub-policy's params (tagged with environmentID and the given
// run-progress counters) to dir under checkpoint.Path's naming
// convention for epoch, creating dir first if it doesn't exist yet.
func saveCheckpointToDir(dir string, trainer *hierarchical.Trainer, environmentID string, epoch int, bestReturn float32, totalUpdates int) error {
	metadata := checkpoint.Metadata{Epoch: epoch, BestReturn: bestReturn, TotalUpdates: totalUpdates}
	return checkpoint.Save(dir, checkpointPrefix, epoch, func(path string) error {
		return trainer.SaveFile(path, environmentID, metadata)
	})
}
