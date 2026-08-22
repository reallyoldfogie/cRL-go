package policy

import (
	"bytes"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSaveLoadRoundTripPreservesWeights(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	original := NewParams(rng, 12, 8, 5)

	var buf bytes.Buffer
	require.NoError(t, original.Save(&buf))

	loaded, err := Load(&buf)
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

func TestSaveFileLoadFileRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	original := NewParams(rng, 6, 4, 3)

	path := filepath.Join(t.TempDir(), "checkpoint.json")
	require.NoError(t, SaveFile(path, original))

	loaded, err := LoadFile(path)
	require.NoError(t, err)
	assert.Equal(t, original.W0.Data, loaded.W0.Data)
	assert.Equal(t, original.W2.Data, loaded.W2.Data)
}

func TestLoadFileRejectsMissingFile(t *testing.T) {
	_, err := LoadFile(filepath.Join(t.TempDir(), "does-not-exist.json"))
	assert.Error(t, err)
}

func TestLoadRejectsInvalidJSON(t *testing.T) {
	_, err := Load(strings.NewReader("not json"))
	assert.Error(t, err)
}

func TestLoadRejectsSizeMismatchedData(t *testing.T) {
	// A checkpoint claiming a 12-wide input layer but only shipping 3
	// W0 values must be rejected rather than silently truncated or
	// zero-padded.
	corrupt := `{"input_size":12,"hidden_size":8,"output_size":5,"w0":[1,2,3],"b0":[],"w1":[],"b1":[],"w2":[],"b2":[]}`
	_, err := Load(strings.NewReader(corrupt))
	assert.Error(t, err)
}

func TestSaveFileRejectsUnwritableDirectory(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	params := NewParams(rng, 4, 4, 2)

	err := SaveFile(filepath.Join(string(os.PathSeparator), "does-not-exist-dir", "checkpoint.json"), params)
	assert.Error(t, err)
}
