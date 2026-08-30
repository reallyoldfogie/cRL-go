package actorcritic

import (
	"bytes"
	"math/rand/v2"
	"path/filepath"
	"strings"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSaveLoadRoundTripPreservesWeights(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	original := NewParams(rng, 12, 8, 5)

	var buf bytes.Buffer
	require.NoError(t, original.Save(&buf, "snake:36", checkpoint.Metadata{}))

	loaded, _, err := Load(&buf, "snake:36")
	require.NoError(t, err)

	assert.Equal(t, original.InputSize(), loaded.InputSize())
	assert.Equal(t, original.HiddenSize(), loaded.HiddenSize())
	assert.Equal(t, original.OutputSize(), loaded.OutputSize())
	assert.Equal(t, original.W0.Data, loaded.W0.Data)
	assert.Equal(t, original.B0.Data, loaded.B0.Data)
	assert.Equal(t, original.W1.Data, loaded.W1.Data)
	assert.Equal(t, original.B1.Data, loaded.B1.Data)
	assert.Equal(t, original.Wpi.Data, loaded.Wpi.Data)
	assert.Equal(t, original.Bpi.Data, loaded.Bpi.Data)
	assert.Equal(t, original.Wv.Data, loaded.Wv.Data)
	assert.Equal(t, original.Bv.Data, loaded.Bv.Data)
}

func TestSaveLoadRoundTripPreservesMetadata(t *testing.T) {
	rng := rand.New(rand.NewPCG(31, 37))
	original := NewParams(rng, 6, 4, 3)
	metadata := checkpoint.Metadata{Epoch: 12, BestReturn: 3.5, TotalUpdates: 480}

	var buf bytes.Buffer
	require.NoError(t, original.Save(&buf, "snake:36", metadata))

	_, loadedMetadata, err := Load(&buf, "snake:36")
	require.NoError(t, err)
	assert.Equal(t, metadata, loadedMetadata)
}

func TestSaveFileLoadFileRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	original := NewParams(rng, 6, 4, 3)

	path := filepath.Join(t.TempDir(), "checkpoint.json")
	require.NoError(t, SaveFile(path, original, "gridworld:36", checkpoint.Metadata{Epoch: 9}))

	loaded, metadata, err := LoadFile(path, "gridworld:36")
	require.NoError(t, err)
	assert.Equal(t, original.W0.Data, loaded.W0.Data)
	assert.Equal(t, original.Wv.Data, loaded.Wv.Data)
	assert.Equal(t, 9, metadata.Epoch)
}

func TestLoadFileRejectsMissingFile(t *testing.T) {
	_, _, err := LoadFile(filepath.Join(t.TempDir(), "does-not-exist.json"), "snake:36")
	assert.Error(t, err)
}

func TestLoadRejectsInvalidJSON(t *testing.T) {
	_, _, err := Load(strings.NewReader("not json"), "snake:36")
	assert.Error(t, err)
}

func TestLoadRejectsMismatchedEnvironmentID(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	original := NewParams(rng, 6, 4, 3)

	var buf bytes.Buffer
	require.NoError(t, original.Save(&buf, "snake:36", checkpoint.Metadata{}))

	_, _, err := Load(&buf, "gridworld:36")
	assert.Error(t, err)
}

func TestLoadRejectsUnsupportedSchemaVersion(t *testing.T) {
	wrongVersion := `{"schema_version":99,"environment_id":"snake:12","input_size":2,"hidden_size":2,"output_size":1,` +
		`"w0":[0,0,0,0],"b0":[0,0],"w1":[0,0,0,0],"b1":[0,0],"wpi":[0,0],"bpi":[0],"wv":[0,0],"bv":[0]}`
	_, _, err := Load(strings.NewReader(wrongVersion), "snake:12")
	assert.Error(t, err)
}

func TestLoadRejectsSizeMismatchedData(t *testing.T) {
	// A checkpoint claiming a 12-wide input layer but only shipping 3
	// W0 values must be rejected rather than silently truncated or
	// zero-padded.
	corrupt := `{"schema_version":1,"environment_id":"snake:12","input_size":12,"hidden_size":8,"output_size":5,` +
		`"w0":[1,2,3],"b0":[],"w1":[],"b1":[],"wpi":[],"bpi":[],"wv":[],"bv":[]}`
	_, _, err := Load(strings.NewReader(corrupt), "snake:12")
	assert.Error(t, err)
}

func TestSaveFileRejectsUnwritableDirectory(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	params := NewParams(rng, 4, 4, 2)

	err := SaveFile(filepath.Join(string(filepath.Separator), "does-not-exist-dir", "checkpoint.json"), params, "snake:4", checkpoint.Metadata{})
	assert.Error(t, err)
}
