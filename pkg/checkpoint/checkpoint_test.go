package checkpoint

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPathIsZeroPaddedAndDeterministic(t *testing.T) {
	assert.Equal(t, filepath.Join("dir", "policy-epoch-000000042.json"), Path("dir", "policy", 42))
	assert.Equal(t, Path("dir", "policy", 42), Path("dir", "policy", 42), "Path must be a pure function of its arguments")
}

func TestLatestRejectsMissingDirectory(t *testing.T) {
	_, err := Latest(filepath.Join(t.TempDir(), "does-not-exist"), "policy")
	assert.Error(t, err)
}

func TestLatestReportsClearErrorWithNoCheckpointsPresent(t *testing.T) {
	dir := t.TempDir()
	_, err := Latest(dir, "policy")
	assert.Error(t, err)
}

// TestLatestPicksHighestEpochNotListingOrder writes checkpoints in an
// order that does not match epoch order (both by creation time and by
// a naive lexical sort of unrelated file names mixed in), then confirms
// Latest still returns the actual highest-epoch one, exercising the
// requirement that this be determined by parsing epoch numbers rather
// than trusting directory-listing or lexical order.
func TestLatestPicksHighestEpochNotListingOrder(t *testing.T) {
	dir := t.TempDir()

	// Written out of epoch order, and interleaved with a
	// higher-prefix, non-checkpoint file name and a checkpoint for a
	// different prefix, both of which must be ignored.
	require.NoError(t, os.WriteFile(Path(dir, "policy", 5), []byte("{}"), 0o644))
	require.NoError(t, os.WriteFile(Path(dir, "policy", 100), []byte("{}"), 0o644))
	require.NoError(t, os.WriteFile(Path(dir, "policy", 42), []byte("{}"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "zzz-not-a-checkpoint.json"), []byte("{}"), 0o644))
	require.NoError(t, os.WriteFile(Path(dir, "ppo", 999), []byte("{}"), 0o644)) // different prefix

	latest, err := Latest(dir, "policy")
	require.NoError(t, err)
	assert.Equal(t, Path(dir, "policy", 100), latest)
}

func TestLatestDistinguishesPrefixes(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, os.WriteFile(Path(dir, "policy", 10), []byte("{}"), 0o644))
	require.NoError(t, os.WriteFile(Path(dir, "ppo", 20), []byte("{}"), 0o644))

	latestPolicy, err := Latest(dir, "policy")
	require.NoError(t, err)
	assert.Equal(t, Path(dir, "policy", 10), latestPolicy)

	latestPPO, err := Latest(dir, "ppo")
	require.NoError(t, err)
	assert.Equal(t, Path(dir, "ppo", 20), latestPPO)
}
