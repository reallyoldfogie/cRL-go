package decisionlog

import (
	"bytes"
	"os"
	"strings"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWriterReadAllRoundTrip(t *testing.T) {
	records := []Record{
		{
			Step:        0,
			Observation: []float32{0.1, 0.2},
			Decision: rl.Decision{
				Action:           1,
				Probabilities:    []float32{0.3, 0.7},
				RawProbabilities: []float32{0.3, 0.7},
				Value:            0.5,
				HasValue:         true,
			},
			Reward: 1,
			Done:   false,
			Render: []string{"..", ".@"},
			Extra:  map[string]any{"active_subgoal": float64(2)},
		},
		{
			Step:        1,
			Observation: []float32{0.3, 0.4},
			Decision: rl.Decision{
				Action:           0,
				Probabilities:    []float32{0.6, 0.4},
				RawProbabilities: []float32{0.6, 0.4},
			},
			Reward: 2,
			Done:   true,
		},
	}

	var buf bytes.Buffer
	writer := NewWriter(&buf)
	for _, rec := range records {
		require.NoError(t, writer.Write(rec))
	}

	got, err := ReadAll(&buf)
	require.NoError(t, err)
	assert.Equal(t, records, got)
}

func TestReadAllOnEmptyReaderReturnsEmptyNonNilSlice(t *testing.T) {
	got, err := ReadAll(strings.NewReader(""))
	require.NoError(t, err)
	assert.NotNil(t, got)
	assert.Empty(t, got)
}

func TestReadAllRejectsMalformedRecord(t *testing.T) {
	_, err := ReadAll(strings.NewReader(`{"step": 0}` + "\n" + `not json` + "\n"))
	require.Error(t, err)
}

func TestFileWriterCreatesTruncatesAndReadsBack(t *testing.T) {
	path := t.TempDir() + "/decisions.jsonl"

	writer, err := NewFileWriter(path)
	require.NoError(t, err)
	require.NoError(t, writer.Write(Record{Step: 0, Decision: rl.Decision{Action: 1}}))
	require.NoError(t, writer.Write(Record{Step: 1, Decision: rl.Decision{Action: 2}}))
	require.NoError(t, writer.Close())

	file, err := os.Open(path)
	require.NoError(t, err)
	defer file.Close()

	got, err := ReadAll(file)
	require.NoError(t, err)
	require.Len(t, got, 2)
	assert.Equal(t, rl.Action(1), got[0].Decision.Action)
	assert.Equal(t, rl.Action(2), got[1].Decision.Action)

	// Re-creating a FileWriter at the same path must truncate, not
	// append, mirroring pkg/metrics.CSVWriter's own create-or-truncate
	// contract.
	writer2, err := NewFileWriter(path)
	require.NoError(t, err)
	require.NoError(t, writer2.Write(Record{Step: 0, Decision: rl.Decision{Action: 9}}))
	require.NoError(t, writer2.Close())

	file2, err := os.Open(path)
	require.NoError(t, err)
	defer file2.Close()

	got2, err := ReadAll(file2)
	require.NoError(t, err)
	require.Len(t, got2, 1)
	assert.Equal(t, rl.Action(9), got2[0].Decision.Action)
}
