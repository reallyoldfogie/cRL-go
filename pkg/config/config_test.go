package config

import (
	"flag"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoadFallsBackToDefaultsWhenFileMissing(t *testing.T) {
	settings, err := Load(filepath.Join(t.TempDir(), "does-not-exist.json"))
	require.NoError(t, err)
	assert.Equal(t, Default(), settings)
}

func TestLoadOverridesOnlyFieldsPresentInFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.json")
	require.NoError(t, writeFile(path, `{"epochs": 10, "learning_rate": 0.01}`))

	settings, err := Load(path)
	require.NoError(t, err)

	assert.Equal(t, 10, settings.Epochs)
	assert.Equal(t, float32(0.01), settings.LearningRate)
	// Untouched fields keep their defaults.
	assert.Equal(t, Default().RolloutSize, settings.RolloutSize)
	assert.Equal(t, Default().GridSize, settings.GridSize)
}

func TestLoadRejectsInvalidJSON(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.json")
	require.NoError(t, writeFile(path, `not json`))

	_, err := Load(path)
	assert.Error(t, err)
}

func TestValidateRejectsNonSquareGridSize(t *testing.T) {
	settings := Default()
	settings.GridSize = 10
	assert.Error(t, settings.Validate())
}

func TestValidateAcceptsDefaults(t *testing.T) {
	assert.NoError(t, Default().Validate())
}

func TestApplyOnlyOverridesExplicitlyPassedFlags(t *testing.T) {
	settings := Default()

	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	overrides := RegisterFlags(fs)
	require.NoError(t, fs.Parse([]string{"-epochs=5", "-seed=0"}))

	Apply(&settings, fs, overrides)

	assert.Equal(t, 5, settings.Epochs, "explicitly passed flag should override the config value")
	assert.Equal(t, uint64(0), settings.Seed, "an explicitly passed zero value must still be honored")
	// Every other field must retain its default (i.e. was not clobbered
	// by the zero-valued, but unset, flags for those fields).
	assert.Equal(t, Default().RolloutSize, settings.RolloutSize)
	assert.Equal(t, Default().LearningRate, settings.LearningRate)
	assert.Equal(t, Default().GridSize, settings.GridSize)
	assert.Equal(t, Default().Workers, settings.Workers)
}

func TestApplyLeavesSettingsUnchangedWhenNoFlagsPassed(t *testing.T) {
	settings := Default()

	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	overrides := RegisterFlags(fs)
	require.NoError(t, fs.Parse(nil))

	Apply(&settings, fs, overrides)

	assert.Equal(t, Default(), settings)
}

func writeFile(path, contents string) error {
	return os.WriteFile(path, []byte(contents), 0o644)
}
