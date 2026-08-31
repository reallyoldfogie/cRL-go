package hierarchical

import (
	"bytes"
	"path/filepath"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const testEnvironmentID = "hierarchicalgridworld:16"

func TestTrainerSaveLoadRoundTripPreservesWeights(t *testing.T) {
	settings := smallTestSettings()
	cfg := smallTestConfig()

	trainer, err := New(settings, cfg, hierarchicalGridWorldFactory(settings.GridSize), nil)
	require.NoError(t, err)

	var buf bytes.Buffer
	require.NoError(t, trainer.Save(&buf, testEnvironmentID, checkpoint.Metadata{}))

	meta, subs, _, err := Load(&buf, testEnvironmentID, cfg.NumSubgoals)
	require.NoError(t, err)

	wantMeta, wantSubs := trainer.Params()
	assert.Equal(t, wantMeta.W0.Data, meta.W0.Data)
	assert.Equal(t, wantMeta.Wpi.Data, meta.Wpi.Data)
	assert.Equal(t, wantMeta.Wv.Data, meta.Wv.Data)

	require.Len(t, subs, cfg.NumSubgoals)
	for i := range cfg.NumSubgoals {
		subgoal := Subgoal(i)
		assert.Equal(t, wantSubs[subgoal].W0.Data, subs[subgoal].W0.Data)
		assert.Equal(t, wantSubs[subgoal].Wpi.Data, subs[subgoal].Wpi.Data)
		assert.Equal(t, wantSubs[subgoal].Wv.Data, subs[subgoal].Wv.Data)
	}
}

func TestTrainerSaveLoadRoundTripPreservesMetadata(t *testing.T) {
	settings := smallTestSettings()
	cfg := smallTestConfig()

	trainer, err := New(settings, cfg, hierarchicalGridWorldFactory(settings.GridSize), nil)
	require.NoError(t, err)

	metadata := checkpoint.Metadata{Epoch: 12, BestReturn: 3.5, TotalUpdates: 480}

	var buf bytes.Buffer
	require.NoError(t, trainer.Save(&buf, testEnvironmentID, metadata))

	_, _, loadedMetadata, err := Load(&buf, testEnvironmentID, cfg.NumSubgoals)
	require.NoError(t, err)
	assert.Equal(t, metadata, loadedMetadata)
}

func TestTrainerSaveFileLoadFileRoundTrip(t *testing.T) {
	settings := smallTestSettings()
	cfg := smallTestConfig()

	trainer, err := New(settings, cfg, hierarchicalGridWorldFactory(settings.GridSize), nil)
	require.NoError(t, err)

	path := filepath.Join(t.TempDir(), "checkpoint.json")
	require.NoError(t, trainer.SaveFile(path, testEnvironmentID, checkpoint.Metadata{Epoch: 9}))

	meta, subs, metadata, err := LoadFile(path, testEnvironmentID, cfg.NumSubgoals)
	require.NoError(t, err)
	assert.NotNil(t, meta)
	assert.Len(t, subs, cfg.NumSubgoals)
	assert.Equal(t, 9, metadata.Epoch)
}

func TestLoadFileRejectsMissingFile(t *testing.T) {
	_, _, _, err := LoadFile(filepath.Join(t.TempDir(), "does-not-exist.json"), testEnvironmentID, 3)
	assert.Error(t, err)
}

func TestLoadRejectsInvalidJSON(t *testing.T) {
	_, _, _, err := Load(bytes.NewReader([]byte("not json")), testEnvironmentID, 3)
	assert.Error(t, err)
}

func TestLoadRejectsMismatchedEnvironmentID(t *testing.T) {
	settings := smallTestSettings()
	cfg := smallTestConfig()

	trainer, err := New(settings, cfg, hierarchicalGridWorldFactory(settings.GridSize), nil)
	require.NoError(t, err)

	var buf bytes.Buffer
	require.NoError(t, trainer.Save(&buf, testEnvironmentID, checkpoint.Metadata{}))

	_, _, _, err = Load(&buf, "hierarchicalgridworld:9", cfg.NumSubgoals)
	assert.Error(t, err)
}

func TestLoadRejectsMismatchedNumSubgoals(t *testing.T) {
	settings := smallTestSettings()
	cfg := smallTestConfig()

	trainer, err := New(settings, cfg, hierarchicalGridWorldFactory(settings.GridSize), nil)
	require.NoError(t, err)

	var buf bytes.Buffer
	require.NoError(t, trainer.Save(&buf, testEnvironmentID, checkpoint.Metadata{}))

	_, _, _, err = Load(&buf, testEnvironmentID, cfg.NumSubgoals+1)
	assert.Error(t, err)
}

func TestLoadRejectsUnsupportedSchemaVersion(t *testing.T) {
	wrongVersion := `{"schema_version":99,"environment_id":"hierarchicalgridworld:16","num_subgoals":0,` +
		`"metadata":{"epoch":0,"best_return":0,"total_updates":0},` +
		`"meta":{"input_size":1,"hidden_size":1,"output_size":1,"w0":[0],"b0":[0],"w1":[0],"b1":[0],"wpi":[0],"bpi":[0],"wv":[0],"bv":[0]},"subs":[]}`
	_, _, _, err := Load(bytes.NewReader([]byte(wrongVersion)), testEnvironmentID, 0)
	assert.Error(t, err)
}

func TestNewResumesFromInitialParams(t *testing.T) {
	settings := smallTestSettings()
	cfg := smallTestConfig()

	original, err := New(settings, cfg, hierarchicalGridWorldFactory(settings.GridSize), nil)
	require.NoError(t, err)

	meta, subs := original.Params()
	initial := &InitialParams{Meta: meta, Subs: subs}

	resumed, err := New(settings, cfg, hierarchicalGridWorldFactory(settings.GridSize), initial)
	require.NoError(t, err)

	resumedMeta, resumedSubs := resumed.Params()
	assert.Same(t, meta, resumedMeta)
	for i := range cfg.NumSubgoals {
		assert.Same(t, subs[Subgoal(i)], resumedSubs[Subgoal(i)])
	}
}

func TestNewRejectsIncompleteInitialParams(t *testing.T) {
	settings := smallTestSettings()
	cfg := smallTestConfig()

	original, err := New(settings, cfg, hierarchicalGridWorldFactory(settings.GridSize), nil)
	require.NoError(t, err)

	meta, subs := original.Params()

	// Deliberately omit subgoal 0 to build an incomplete bundle.
	incompleteSubs := make(map[Subgoal]*actorcritic.Params, cfg.NumSubgoals-1)
	for i := 1; i < cfg.NumSubgoals; i++ {
		incompleteSubs[Subgoal(i)] = subs[Subgoal(i)]
	}
	initial := &InitialParams{Meta: meta, Subs: incompleteSubs}

	_, err = New(settings, cfg, hierarchicalGridWorldFactory(settings.GridSize), initial)
	assert.Error(t, err)
}

func TestNewRejectsMismatchedInitialParamsShape(t *testing.T) {
	settings := smallTestSettings()
	cfg := smallTestConfig()

	mismatchedCfg := smallTestConfig()
	mismatchedCfg.MetaHiddenSize = cfg.MetaHiddenSize + 4

	original, err := New(settings, mismatchedCfg, hierarchicalGridWorldFactory(settings.GridSize), nil)
	require.NoError(t, err)

	meta, subs := original.Params()
	initial := &InitialParams{Meta: meta, Subs: subs}

	_, err = New(settings, cfg, hierarchicalGridWorldFactory(settings.GridSize), initial)
	assert.Error(t, err)
}
