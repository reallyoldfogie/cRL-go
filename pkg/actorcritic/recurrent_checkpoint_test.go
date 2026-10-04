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

func TestRecurrentSaveLoadRoundTripPreservesWeights(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	original := NewRecurrentParams(rng, 12, 8, 5, 2)

	var buf bytes.Buffer
	require.NoError(t, original.Save(&buf, "snake:36", checkpoint.Metadata{}))

	loaded, _, err := RecurrentLoad(&buf, "snake:36")
	require.NoError(t, err)

	assert.Equal(t, original.InputSize(), loaded.InputSize())
	assert.Equal(t, original.HiddenSize(), loaded.HiddenSize())
	assert.Equal(t, original.OutputSize(), loaded.OutputSize())
	require.Equal(t, original.NumHiddenLayers(), loaded.NumHiddenLayers())
	for i := range original.Trunk.Hidden {
		assert.Equal(t, original.Trunk.Hidden[i].W.Data, loaded.Trunk.Hidden[i].W.Data)
		assert.Equal(t, original.Trunk.Hidden[i].B.Data, loaded.Trunk.Hidden[i].B.Data)
	}
	assert.Equal(t, original.Trunk.Wpi.Data, loaded.Trunk.Wpi.Data)
	assert.Equal(t, original.Trunk.Wv.Data, loaded.Trunk.Wv.Data)

	assert.Equal(t, original.LSTM.WxF.Data, loaded.LSTM.WxF.Data)
	assert.Equal(t, original.LSTM.WhF.Data, loaded.LSTM.WhF.Data)
	assert.Equal(t, original.LSTM.BF.Data, loaded.LSTM.BF.Data)
	assert.Equal(t, original.LSTM.WxI.Data, loaded.LSTM.WxI.Data)
	assert.Equal(t, original.LSTM.WxO.Data, loaded.LSTM.WxO.Data)
	assert.Equal(t, original.LSTM.WxG.Data, loaded.LSTM.WxG.Data)
}

func TestRecurrentSaveLoadRoundTripPreservesMetadata(t *testing.T) {
	rng := rand.New(rand.NewPCG(31, 37))
	original := NewRecurrentParams(rng, 6, 4, 3, 1)
	metadata := checkpoint.Metadata{Epoch: 12, BestReturn: 3.5, TotalUpdates: 480}

	var buf bytes.Buffer
	require.NoError(t, original.Save(&buf, "snake:36", metadata))

	_, loadedMetadata, err := RecurrentLoad(&buf, "snake:36")
	require.NoError(t, err)
	assert.Equal(t, metadata, loadedMetadata)
}

func TestRecurrentSaveFileLoadFileRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	original := NewRecurrentParams(rng, 6, 4, 3, 1)

	path := filepath.Join(t.TempDir(), "checkpoint.json")
	require.NoError(t, original.SaveFile(path, "gridworld:36", checkpoint.Metadata{Epoch: 9}))

	loaded, metadata, err := RecurrentLoadFile(path, "gridworld:36")
	require.NoError(t, err)
	assert.Equal(t, original.LSTM.WxF.Data, loaded.LSTM.WxF.Data)
	assert.Equal(t, 9, metadata.Epoch)
}

func TestRecurrentLoadRejectsMismatchedEnvironmentID(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	original := NewRecurrentParams(rng, 6, 4, 3, 1)

	var buf bytes.Buffer
	require.NoError(t, original.Save(&buf, "snake:36", checkpoint.Metadata{}))

	_, _, err := RecurrentLoad(&buf, "gridworld:36")
	assert.Error(t, err)
}

func TestRecurrentLoadRejectsUnsupportedSchemaVersion(t *testing.T) {
	wrongVersion := `{"schema_version":99,"environment_id":"snake:12","input_size":2,"hidden_size":2,"output_size":1,"hidden":[],"wpi":[],"bpi":[],"wv":[],"bv":[],"lstm":{}}`
	_, _, err := RecurrentLoad(strings.NewReader(wrongVersion), "snake:12")
	assert.Error(t, err)
}

func TestRecurrentLoadRejectsSizeMismatchedData(t *testing.T) {
	corrupt := `{"schema_version":1,"environment_id":"snake:12","input_size":12,"hidden_size":8,"output_size":5,"hidden":[],"wpi":[],"bpi":[],"wv":[],"bv":[],"lstm":{"wxf":[1,2,3]}}`
	_, _, err := RecurrentLoad(strings.NewReader(corrupt), "snake:12")
	assert.Error(t, err)
}
