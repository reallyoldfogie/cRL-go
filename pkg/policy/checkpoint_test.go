package policy

import (
	"bytes"
	"math/rand/v2"
	"os"
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
	assert.Equal(t, original.W2.Data, loaded.W2.Data)
	assert.Equal(t, original.B2.Data, loaded.B2.Data)
}

func TestSaveLoadRoundTripPreservesMetadata(t *testing.T) {
	rng := rand.New(rand.NewPCG(21, 23))
	original := NewParams(rng, 6, 4, 3)
	metadata := checkpoint.Metadata{Epoch: 41, BestReturn: 12.5, TotalUpdates: 4100}

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
	require.NoError(t, SaveFile(path, original, "gridworld:36", checkpoint.Metadata{Epoch: 7}))

	loaded, metadata, err := LoadFile(path, "gridworld:36")
	require.NoError(t, err)
	assert.Equal(t, original.W0.Data, loaded.W0.Data)
	assert.Equal(t, original.W2.Data, loaded.W2.Data)
	assert.Equal(t, 7, metadata.Epoch)
}

func TestLoadFileRejectsMissingFile(t *testing.T) {
	_, _, err := LoadFile(filepath.Join(t.TempDir(), "does-not-exist.json"), "snake:36")
	assert.Error(t, err)
}

func TestLoadRejectsInvalidJSON(t *testing.T) {
	_, _, err := Load(strings.NewReader("not json"), "snake:36")
	assert.Error(t, err)
}

func TestLoadRejectsSizeMismatchedData(t *testing.T) {
	// A checkpoint claiming a 12-wide input layer but only shipping 3
	// W0 values must be rejected rather than silently truncated or
	// zero-padded.
	corrupt := `{"input_size":12,"hidden_size":8,"output_size":5,"w0":[1,2,3],"b0":[],"w1":[],"b1":[],"w2":[],"b2":[]}`
	_, _, err := Load(strings.NewReader(corrupt), "snake:12")
	assert.Error(t, err)
}

func TestSaveFileRejectsUnwritableDirectory(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	params := NewParams(rng, 4, 4, 2)

	err := SaveFile(filepath.Join(string(os.PathSeparator), "does-not-exist-dir", "checkpoint.json"), params, "snake:4", checkpoint.Metadata{})
	assert.Error(t, err)
}

func TestLoadRejectsMismatchedEnvironmentID(t *testing.T) {
	rng := rand.New(rand.NewPCG(13, 17))
	original := NewParams(rng, 6, 4, 3)

	var buf bytes.Buffer
	require.NoError(t, original.Save(&buf, "snake:36", checkpoint.Metadata{}))

	_, _, err := Load(&buf, "gridworld:36")
	assert.Error(t, err)
}

func TestLoadRejectsUnsupportedFutureSchemaVersion(t *testing.T) {
	fromTheFuture := `{"schema_version":99,"environment_id":"snake:36","input_size":2,"hidden_size":2,"output_size":1,` +
		`"w0":[0,0,0,0],"b0":[0,0],"w1":[0,0,0,0],"b1":[0,0],"w2":[0,0],"b2":[0]}`
	_, _, err := Load(strings.NewReader(fromTheFuture), "snake:36")
	assert.Error(t, err)
}

func TestLoadSkipsEnvironmentCheckForLegacyCheckpoint(t *testing.T) {
	// A checkpoint saved before schema_version/environment_id existed
	// has neither field in its JSON at all (as opposed to an empty
	// string for environment_id, which is a checkpoint saved by *this*
	// version of Save with an empty environmentID). It must still load
	// successfully regardless of what environment the caller expects.
	legacy := `{"input_size":2,"hidden_size":2,"output_size":1,"w0":[0,0,0,0],"b0":[0,0],"w1":[0,0,0,0],"b1":[0,0],"w2":[0,0],"b2":[0]}`

	loaded, _, err := Load(strings.NewReader(legacy), "whatever-the-caller-expects")
	require.NoError(t, err)
	assert.Equal(t, 2, loaded.InputSize())
	assert.Equal(t, 2, loaded.HiddenSize())
	assert.Equal(t, 1, loaded.OutputSize())
}
