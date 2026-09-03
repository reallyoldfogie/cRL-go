package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRunWritesEmbeddedViewerVerbatim(t *testing.T) {
	outPath := filepath.Join(t.TempDir(), "replay.html")
	require.NoError(t, run([]string{"-out=" + outPath}))

	got, err := os.ReadFile(outPath)
	require.NoError(t, err)
	assert.Equal(t, replayHTML, got)
}

func TestRunDefaultsToReplayHTMLInCurrentDirectory(t *testing.T) {
	dir := t.TempDir()
	oldwd, err := os.Getwd()
	require.NoError(t, err)
	require.NoError(t, os.Chdir(dir))
	t.Cleanup(func() { _ = os.Chdir(oldwd) })

	require.NoError(t, run(nil))

	_, err = os.Stat(filepath.Join(dir, "replay.html"))
	require.NoError(t, err)
}

func TestEmbeddedViewerIsSelfContained(t *testing.T) {
	content := string(replayHTML)
	assert.NotContains(t, content, "http://", "the viewer must not fetch anything over the network")
	assert.NotContains(t, content, "https://", "the viewer must not fetch anything over the network")
	assert.Contains(t, content, "decision-log-out")
	assert.Contains(t, content, "<script>")
}
