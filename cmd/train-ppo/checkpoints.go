package main

import (
	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"
)

// checkpointPrefix distinguishes cmd/train-ppo's -checkpoint-dir
// checkpoints from cmd/train's (see checkpoint.Path/checkpoint.Latest).
const checkpointPrefix = "ppo"

// resumeState is what resumeFromCheckpointDir found (or didn't) in a
// -checkpoint-dir: the loaded Params (nil if starting fresh) and the
// run-progress counters to continue from.
type resumeState = checkpoint.Resumed[*actorcritic.Params]

// resumeFromCheckpointDir looks for the latest checkpoint in dir and
// loads it if found; see checkpoint.Resume for the fresh-start/error
// contract.
func resumeFromCheckpointDir(dir, environmentID string) (resumeState, error) {
	return checkpoint.Resume(dir, checkpointPrefix, func(path string) (*actorcritic.Params, checkpoint.Metadata, error) {
		return actorcritic.LoadFile(path, environmentID)
	})
}

// saveCheckpointToDir saves params (tagged with environmentID and the
// given run-progress counters) to dir under checkpoint.Path's naming
// convention for epoch, creating dir first if it doesn't exist yet.
func saveCheckpointToDir(dir string, params *actorcritic.Params, environmentID string, epoch int, bestReturn float32, totalUpdates int) error {
	metadata := checkpoint.Metadata{Epoch: epoch, BestReturn: bestReturn, TotalUpdates: totalUpdates}
	return checkpoint.Save(dir, checkpointPrefix, epoch, func(path string) error {
		return actorcritic.SaveFile(path, params, environmentID, metadata)
	})
}
