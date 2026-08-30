package main

import (
	"math/rand/v2"
	"path/filepath"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"
	"github.com/reallyoldfogie/cRL-go/pkg/policy"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestReadCheckpointInfoWorksAgainstAPolicyCheckpoint(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "policy-epoch-000000005.json")

	rng := rand.New(rand.NewPCG(1, 2))
	params := policy.NewParams(rng, 8, 4, 3)
	metadata := checkpoint.Metadata{Epoch: 5, BestReturn: 10.5, TotalUpdates: 6}
	require.NoError(t, policy.SaveFile(path, params, "snake:8", metadata))

	info, err := readCheckpointInfo(path)
	require.NoError(t, err)
	assert.Equal(t, "snake:8", info.EnvironmentID)
	assert.Equal(t, 8, info.InputSize)
	assert.Equal(t, 4, info.HiddenSize)
	assert.Equal(t, 3, info.OutputSize)
	assert.Equal(t, metadata, info.Metadata)
}

func TestReadCheckpointInfoWorksAgainstAnActorCriticCheckpoint(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "ppo-epoch-000000009.json")

	rng := rand.New(rand.NewPCG(3, 4))
	params := actorcritic.NewParams(rng, 6, 5, 2)
	metadata := checkpoint.Metadata{Epoch: 9, BestReturn: -1.5, TotalUpdates: 40}
	require.NoError(t, actorcritic.SaveFile(path, params, "gridworld:6", metadata))

	info, err := readCheckpointInfo(path)
	require.NoError(t, err)
	assert.Equal(t, "gridworld:6", info.EnvironmentID)
	assert.Equal(t, 6, info.InputSize)
	assert.Equal(t, 5, info.HiddenSize)
	assert.Equal(t, 2, info.OutputSize)
	assert.Equal(t, metadata, info.Metadata)
}

func TestReadCheckpointInfoRejectsMissingFile(t *testing.T) {
	_, err := readCheckpointInfo(filepath.Join(t.TempDir(), "does-not-exist.json"))
	assert.Error(t, err)
}

func TestRunListSucceedsAgainstADirectoryOfCheckpoints(t *testing.T) {
	dir := t.TempDir()
	rng := rand.New(rand.NewPCG(5, 6))
	params := policy.NewParams(rng, 4, 4, 2)

	require.NoError(t, policy.SaveFile(checkpoint.Path(dir, "policy", 1), params, "snake:4", checkpoint.Metadata{Epoch: 1}))
	require.NoError(t, policy.SaveFile(checkpoint.Path(dir, "policy", 2), params, "snake:4", checkpoint.Metadata{Epoch: 2}))

	assert.NoError(t, runList([]string{dir}))
}

func TestRunInfoAndCompareSucceedAgainstRealCheckpoints(t *testing.T) {
	dir := t.TempDir()
	rng := rand.New(rand.NewPCG(7, 8))
	paramsA := policy.NewParams(rng, 4, 4, 2)
	paramsB := policy.NewParams(rng, 4, 4, 2)

	pathA := checkpoint.Path(dir, "policy", 1)
	pathB := checkpoint.Path(dir, "policy", 2)
	require.NoError(t, policy.SaveFile(pathA, paramsA, "snake:4", checkpoint.Metadata{Epoch: 1, BestReturn: 1.0}))
	require.NoError(t, policy.SaveFile(pathB, paramsB, "snake:4", checkpoint.Metadata{Epoch: 2, BestReturn: 2.0}))

	assert.NoError(t, runInfo([]string{pathA}))
	assert.NoError(t, runCompare([]string{pathA, pathB}))
}

func TestRunRejectsUnknownSubcommand(t *testing.T) {
	assert.Error(t, run([]string{"frobnicate"}))
}

func TestRunRejectsNoArgs(t *testing.T) {
	assert.Error(t, run(nil))
}
