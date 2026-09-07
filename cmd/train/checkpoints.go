package main

import (
	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"
	"github.com/reallyoldfogie/cRL-go/pkg/policy"
)

// checkpointPrefix distinguishes cmd/train's -checkpoint-dir checkpoints
// from cmd/train-ppo's (see checkpoint.Path/checkpoint.Latest).
const checkpointPrefix = "policy"

// resumeState is what resumeFromCheckpointDir found (or didn't) in a
// -checkpoint-dir: the loaded Params (nil if starting fresh) and the
// run-progress counters to continue from.
type resumeState = checkpoint.Resumed[*policy.Params]

// resumeFromCheckpointDir looks for the latest checkpoint in dir and
// loads it if found; see checkpoint.Resume for the fresh-start/error
// contract.
func resumeFromCheckpointDir(dir, environmentID string) (resumeState, error) {
	return checkpoint.Resume(dir, checkpointPrefix, func(path string) (*policy.Params, checkpoint.Metadata, error) {
		return policy.LoadFile(path, environmentID)
	})
}

// saveCheckpointToDir saves params (tagged with environmentID and the
// given run-progress counters) to dir under checkpoint.Path's naming
// convention for epoch, creating dir first if it doesn't exist yet.
func saveCheckpointToDir(dir string, params *policy.Params, environmentID string, epoch int, bestReturn float32, totalUpdates int) error {
	metadata := checkpoint.Metadata{Epoch: epoch, BestReturn: bestReturn, TotalUpdates: totalUpdates}
	return checkpoint.Save(dir, checkpointPrefix, epoch, func(path string) error {
		return policy.SaveFile(path, params, environmentID, metadata)
	})
}
