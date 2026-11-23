package main

import (
	"fmt"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"
	"github.com/reallyoldfogie/cRL-go/pkg/ppo"
)

// checkpointPrefix distinguishes cmd/train-ppo's -checkpoint-dir
// checkpoints from cmd/train's (see checkpoint.Path/checkpoint.Latest).
// policyCheckpointPrefix/valueCheckpointPrefix further distinguish the
// policy-branch and value-branch files a single epoch's checkpoint now
// saves as — see ppo.ActorCriticParams' doc comment for why Trainer
// trains these as two independent *actorcritic.Params rather than one.
//
// This is a breaking change to this command's on-disk checkpoint
// format: a pre-existing single "ppo-epoch-*.json" file (the old
// shared-trunk shape) no longer matches either prefix, so
// resumeFromCheckpointDir treats a -checkpoint-dir containing only such
// files as empty (a fresh start), exactly like checkpoint.Resume treats
// any other directory with no matching checkpoint — not an error. There
// is no automated migration from the old shared-trunk shape to the new
// split one: splitting one trained trunk back into two independent
// branches has no well-defined, non-destructive answer (contrast
// actorcritic's own migrateSchemaV1, which is a lossless reshaping of
// the same data). Given this command's trainer is not the one currently
// used for real training runs (see pkg/ppo/recurrent_trainer.go), this
// was judged an acceptable one-time break rather than added migration
// complexity for a path with no checkpoints worth preserving.
const (
	checkpointPrefix       = "ppo"
	policyCheckpointPrefix = checkpointPrefix + "-policy"
	valueCheckpointPrefix  = checkpointPrefix + "-value"
)

// resumeState is what resumeFromCheckpointDir found (or didn't) in a
// -checkpoint-dir: the loaded ActorCriticParams (nil if starting fresh)
// and the run-progress counters to continue from.
type resumeState = checkpoint.Resumed[*ppo.ActorCriticParams]

// resumeFromCheckpointDir looks for the latest policy/value checkpoint
// pair in dir and loads both if found; see checkpoint.Resume for the
// fresh-start/error contract. The two files are saved together, always
// at the same epoch (see saveCheckpointToDir), so finding one without
// the other indicates a corrupt or hand-edited checkpoint directory
// rather than a normal fresh-start state.
func resumeFromCheckpointDir(dir, environmentID string) (resumeState, error) {
	policyResumed, err := checkpoint.Resume(dir, policyCheckpointPrefix, func(path string) (*actorcritic.Params, checkpoint.Metadata, error) {
		return actorcritic.LoadFile(path, environmentID)
	})
	if err != nil {
		return resumeState{}, err
	}
	if policyResumed.Params == nil {
		return resumeState{BestReturn: policyResumed.BestReturn}, nil
	}

	valueResumed, err := checkpoint.Resume(dir, valueCheckpointPrefix, func(path string) (*actorcritic.Params, checkpoint.Metadata, error) {
		return actorcritic.LoadFile(path, environmentID)
	})
	if err != nil {
		return resumeState{}, err
	}
	if valueResumed.Params == nil {
		return resumeState{}, fmt.Errorf("found a %q checkpoint in %s without a matching %q checkpoint", policyCheckpointPrefix, dir, valueCheckpointPrefix)
	}

	return resumeState{
		Params:       &ppo.ActorCriticParams{Policy: policyResumed.Params, Value: valueResumed.Params},
		StartEpoch:   policyResumed.StartEpoch,
		BestReturn:   policyResumed.BestReturn,
		TotalUpdates: policyResumed.TotalUpdates,
	}, nil
}

// saveCheckpointToDir saves params' Policy and Value branches (each
// tagged with environmentID and the given run-progress counters) to dir
// under checkpoint.Path's naming convention for epoch, creating dir
// first if it doesn't exist yet.
func saveCheckpointToDir(dir string, params *ppo.ActorCriticParams, environmentID string, epoch int, bestReturn float32, totalUpdates int) error {
	metadata := checkpoint.Metadata{Epoch: epoch, BestReturn: bestReturn, TotalUpdates: totalUpdates}
	if err := checkpoint.Save(dir, policyCheckpointPrefix, epoch, func(path string) error {
		return actorcritic.SaveFile(path, params.Policy, environmentID, metadata)
	}); err != nil {
		return err
	}
	return checkpoint.Save(dir, valueCheckpointPrefix, epoch, func(path string) error {
		return actorcritic.SaveFile(path, params.Value, environmentID, metadata)
	})
}
