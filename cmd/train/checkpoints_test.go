package main

import (
	"math"
	"math/rand/v2"
	"path/filepath"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"
	"github.com/reallyoldfogie/cRL-go/pkg/policy"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResumeFromCheckpointDirStartsFreshWithNoDirSet(t *testing.T) {
	resume, err := resumeFromCheckpointDir("", "snake:4")
	require.NoError(t, err)

	assert.Nil(t, resume.Params)
	assert.Equal(t, 0, resume.StartEpoch)
	assert.Equal(t, 0, resume.TotalUpdates)
	assert.True(t, math.IsInf(float64(resume.BestReturn), -1), "a fresh resumeState's BestReturn must be -Inf so any real return improves it")
}

func TestResumeFromCheckpointDirStartsFreshWithEmptyDir(t *testing.T) {
	resume, err := resumeFromCheckpointDir(t.TempDir(), "snake:4")
	require.NoError(t, err)
	assert.Nil(t, resume.Params)
	assert.Equal(t, 0, resume.StartEpoch)
}

// TestResumeFromCheckpointDirContinuesExactEpochAndMetadata is the
// integration-style check docs/plans/05-checkpoint-tooling-and-autoresume.md
// asks for: saving a checkpoint mid-training and then resuming from it
// must continue from that checkpoint's exact epoch/metadata, not
// restart the counters from zero.
func TestResumeFromCheckpointDirContinuesExactEpochAndMetadata(t *testing.T) {
	dir := t.TempDir()
	environmentID := "snake:4"

	rng := rand.New(rand.NewPCG(1, 2))
	params := policy.NewParams(rng, 8, 4, 3)

	savedMetadata := checkpoint.Metadata{Epoch: 41, BestReturn: 12.5, TotalUpdates: 328}
	require.NoError(t, saveCheckpointToDir(dir, params, environmentID, savedMetadata.Epoch, savedMetadata.BestReturn, savedMetadata.TotalUpdates))

	resume, err := resumeFromCheckpointDir(dir, environmentID)
	require.NoError(t, err)

	require.NotNil(t, resume.Params)
	assert.Equal(t, params.W0.Data, resume.Params.W0.Data, "resuming must load the exact saved weights")
	assert.Equal(t, savedMetadata.Epoch+1, resume.StartEpoch, "resuming must continue from the epoch *after* the one the checkpoint recorded")
	assert.Equal(t, savedMetadata.BestReturn, resume.BestReturn)
	assert.Equal(t, savedMetadata.TotalUpdates, resume.TotalUpdates)
}

func TestResumeFromCheckpointDirPicksLatestAcrossMultipleSaves(t *testing.T) {
	dir := t.TempDir()
	environmentID := "snake:4"
	rng := rand.New(rand.NewPCG(3, 4))
	params := policy.NewParams(rng, 8, 4, 3)

	require.NoError(t, saveCheckpointToDir(dir, params, environmentID, 10, 1.0, 100))
	require.NoError(t, saveCheckpointToDir(dir, params, environmentID, 50, 5.0, 500))
	require.NoError(t, saveCheckpointToDir(dir, params, environmentID, 30, 3.0, 300))

	resume, err := resumeFromCheckpointDir(dir, environmentID)
	require.NoError(t, err)
	assert.Equal(t, 51, resume.StartEpoch)
	assert.Equal(t, float32(5.0), resume.BestReturn)
	assert.Equal(t, 500, resume.TotalUpdates)
}

func TestResumeFromCheckpointDirRejectsMismatchedEnvironment(t *testing.T) {
	dir := t.TempDir()
	rng := rand.New(rand.NewPCG(5, 6))
	params := policy.NewParams(rng, 8, 4, 3)

	require.NoError(t, saveCheckpointToDir(dir, params, "snake:4", 1, 0, 1))

	_, err := resumeFromCheckpointDir(dir, "gridworld:4")
	assert.Error(t, err)
}

func TestSaveCheckpointToDirCreatesDirectory(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "does-not-exist-yet")
	rng := rand.New(rand.NewPCG(7, 8))
	params := policy.NewParams(rng, 4, 4, 2)

	require.NoError(t, saveCheckpointToDir(dir, params, "snake:4", 0, 0, 1))

	_, err := checkpoint.Latest(dir, checkpointPrefix)
	assert.NoError(t, err)
}
