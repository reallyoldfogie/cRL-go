package checkpoint

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fakeCheckpoint is a minimal on-disk format exercising Resume/Save
// independent of any real pkg/policy/pkg/actorcritic/pkg/hierarchical
// format, so these tests cover the generic control flow itself.
type fakeCheckpoint struct {
	EnvironmentID string   `json:"environment_id"`
	Metadata      Metadata `json:"metadata"`
	Value         string   `json:"value"`
}

func fakeSave(path, environmentID, value string, metadata Metadata) error {
	data, err := json.Marshal(fakeCheckpoint{EnvironmentID: environmentID, Metadata: metadata, Value: value})
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func fakeLoad(expectedEnvironmentID string) func(path string) (string, Metadata, error) {
	return func(path string) (string, Metadata, error) {
		data, err := os.ReadFile(path)
		if err != nil {
			return "", Metadata{}, err
		}
		var loaded fakeCheckpoint
		if err := json.Unmarshal(data, &loaded); err != nil {
			return "", Metadata{}, err
		}
		if loaded.EnvironmentID != expectedEnvironmentID {
			return "", Metadata{}, fmt.Errorf("saved for environment %q, want %q", loaded.EnvironmentID, expectedEnvironmentID)
		}
		return loaded.Value, loaded.Metadata, nil
	}
}

func TestResumeStartsFreshWithNoDirSet(t *testing.T) {
	resumed, err := Resume("", "fake", fakeLoad("env"))
	require.NoError(t, err)

	assert.Equal(t, "", resumed.Params, "zero value of T means start fresh")
	assert.Equal(t, 0, resumed.StartEpoch)
	assert.Equal(t, 0, resumed.TotalUpdates)
	assert.True(t, math.IsInf(float64(resumed.BestReturn), -1), "a fresh Resumed's BestReturn must be -Inf so any real return improves it")
}

func TestResumeStartsFreshWithEmptyDir(t *testing.T) {
	resumed, err := Resume(t.TempDir(), "fake", fakeLoad("env"))
	require.NoError(t, err)
	assert.Equal(t, "", resumed.Params)
	assert.Equal(t, 0, resumed.StartEpoch)
}

func TestResumeContinuesExactEpochAndMetadata(t *testing.T) {
	dir := t.TempDir()
	savedMetadata := Metadata{Epoch: 41, BestReturn: 12.5, TotalUpdates: 328}

	require.NoError(t, Save(dir, "fake", savedMetadata.Epoch, func(path string) error {
		return fakeSave(path, "env", "trained-weights", savedMetadata)
	}))

	resumed, err := Resume(dir, "fake", fakeLoad("env"))
	require.NoError(t, err)

	assert.Equal(t, "trained-weights", resumed.Params)
	assert.Equal(t, savedMetadata.Epoch+1, resumed.StartEpoch)
	assert.Equal(t, savedMetadata.BestReturn, resumed.BestReturn)
	assert.Equal(t, savedMetadata.TotalUpdates, resumed.TotalUpdates)
}

func TestResumePicksLatestAcrossMultipleSaves(t *testing.T) {
	dir := t.TempDir()
	save := func(epoch int, bestReturn float32, totalUpdates int) error {
		return Save(dir, "fake", epoch, func(path string) error {
			return fakeSave(path, "env", "v", Metadata{Epoch: epoch, BestReturn: bestReturn, TotalUpdates: totalUpdates})
		})
	}
	require.NoError(t, save(10, 1.0, 100))
	require.NoError(t, save(50, 5.0, 500))
	require.NoError(t, save(30, 3.0, 300))

	resumed, err := Resume(dir, "fake", fakeLoad("env"))
	require.NoError(t, err)
	assert.Equal(t, 51, resumed.StartEpoch)
	assert.Equal(t, float32(5.0), resumed.BestReturn)
	assert.Equal(t, 500, resumed.TotalUpdates)
}

func TestResumePropagatesLoadError(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, Save(dir, "fake", 1, func(path string) error {
		return fakeSave(path, "env-a", "v", Metadata{Epoch: 1})
	}))

	_, err := Resume(dir, "fake", fakeLoad("env-b"))
	assert.Error(t, err)
}

func TestSaveCreatesDirectory(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "does-not-exist-yet")

	require.NoError(t, Save(dir, "fake", 0, func(path string) error {
		return fakeSave(path, "env", "v", Metadata{Epoch: 0})
	}))

	_, err := Latest(dir, "fake")
	assert.NoError(t, err)
}
