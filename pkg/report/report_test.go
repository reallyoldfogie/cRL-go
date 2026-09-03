package report

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const sampleCSV = `epoch,average_return,sample_count
0,-8.5,89
1,-6.2,91
2,-3.1,90
`

func TestParseCSV(t *testing.T) {
	rpt, err := ParseCSV(strings.NewReader(sampleCSV))
	require.NoError(t, err)

	assert.Equal(t, []int{0, 1, 2}, rpt.Epochs)
	assert.Equal(t, []string{"average_return", "sample_count"}, rpt.Columns)
	assert.Equal(t, []float64{-8.5, -6.2, -3.1}, rpt.Series["average_return"])
	assert.Equal(t, []float64{89, 91, 90}, rpt.Series["sample_count"])
}

func TestParseCSVRejectsMissingEpochHeader(t *testing.T) {
	_, err := ParseCSV(strings.NewReader("foo,bar\n1,2\n"))
	require.Error(t, err)
}

func TestParseCSVRejectsEmptyInput(t *testing.T) {
	_, err := ParseCSV(strings.NewReader(""))
	require.Error(t, err)
}

func TestParseCSVRejectsNonNumericColumn(t *testing.T) {
	_, err := ParseCSV(strings.NewReader("epoch,average_return\n0,not-a-number\n"))
	require.Error(t, err)
}

func TestParseCSVHeaderOnlyProducesNoRows(t *testing.T) {
	rpt, err := ParseCSV(strings.NewReader("epoch,average_return\n"))
	require.NoError(t, err)
	assert.Empty(t, rpt.Epochs)
	assert.Equal(t, []string{"average_return"}, rpt.Columns)
	assert.Empty(t, rpt.Series["average_return"])
}

func TestWriteHTMLContainsEveryColumnAndTitle(t *testing.T) {
	rpt, err := ParseCSV(strings.NewReader(sampleCSV))
	require.NoError(t, err)

	var buf strings.Builder
	require.NoError(t, WriteHTML(&buf, rpt, "My Training Run"))

	out := buf.String()
	assert.Contains(t, out, "My Training Run")
	assert.Contains(t, out, "average_return")
	assert.Contains(t, out, "sample_count")
	assert.Contains(t, out, "<svg")
	assert.Contains(t, out, "<polyline")
	assert.Equal(t, 2, strings.Count(out, "<polyline"), "one chart per metric column")
}

func TestWriteHTMLEscapesUntrustedText(t *testing.T) {
	rpt := Report{
		Epochs:  []int{0},
		Columns: []string{"<script>evil</script>"},
		Series:  map[string][]float64{"<script>evil</script>": {1}},
	}

	var buf strings.Builder
	require.NoError(t, WriteHTML(&buf, rpt, "<script>alert(1)</script>"))

	out := buf.String()
	assert.NotContains(t, out, "<script>evil</script>")
	assert.NotContains(t, out, "<script>alert(1)</script>")
}

func TestWriteHTMLHandlesFlatSeriesWithoutDivideByZero(t *testing.T) {
	rpt := Report{
		Epochs:  []int{0, 1, 2},
		Columns: []string{"constant"},
		Series:  map[string][]float64{"constant": {5, 5, 5}},
	}

	var buf strings.Builder
	require.NoError(t, WriteHTML(&buf, rpt, "flat"))
	assert.Contains(t, buf.String(), "<polyline")
}

func TestWriteHTMLOnEmptyReportStillProducesValidPage(t *testing.T) {
	var buf strings.Builder
	require.NoError(t, WriteHTML(&buf, Report{}, "empty"))
	assert.Contains(t, buf.String(), "No data.")
}
