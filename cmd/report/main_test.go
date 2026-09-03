package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRunWritesHTMLReportFromMetricsCSV(t *testing.T) {
	dir := t.TempDir()
	metricsPath := filepath.Join(dir, "metrics.csv")
	require.NoError(t, os.WriteFile(metricsPath, []byte("epoch,average_return,sample_count\n0,-8.5,89\n1,-3.1,90\n"), 0o644))

	outPath := filepath.Join(dir, "report.html")
	require.NoError(t, run([]string{
		"-metrics-in=" + metricsPath,
		"-out=" + outPath,
		"-title=Test Run",
	}))

	out, err := os.ReadFile(outPath)
	require.NoError(t, err)
	assert.Contains(t, string(out), "Test Run")
	assert.Contains(t, string(out), "average_return")
	assert.Contains(t, string(out), "<svg")
}

func TestRunRequiresMetricsIn(t *testing.T) {
	err := run(nil)
	require.Error(t, err)
}

func TestRunRejectsMissingMetricsFile(t *testing.T) {
	err := run([]string{"-metrics-in=/nonexistent/metrics.csv"})
	require.Error(t, err)
}
