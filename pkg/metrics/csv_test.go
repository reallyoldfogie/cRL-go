package metrics

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCSVWriterWritesHeaderAndRows(t *testing.T) {
	path := filepath.Join(t.TempDir(), "metrics.csv")

	writer, err := NewCSVWriter(path, []string{"epoch", "average_return"})
	require.NoError(t, err)
	require.NoError(t, writer.WriteRow(0, 1.5))
	require.NoError(t, writer.WriteRow(1, 2.25))
	require.NoError(t, writer.Close())

	data, err := os.ReadFile(path)
	require.NoError(t, err)
	assert.Equal(t, "epoch,average_return\n0,1.5\n1,2.25\n", string(data))
}

func TestNewCSVWriterRejectsUnwritableDirectory(t *testing.T) {
	_, err := NewCSVWriter(filepath.Join(string(filepath.Separator), "does-not-exist-dir", "metrics.csv"), []string{"epoch"})
	assert.Error(t, err)
}
