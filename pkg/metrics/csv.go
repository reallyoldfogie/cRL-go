// Package metrics provides a minimal per-epoch CSV writer shared by
// cmd/train, cmd/train-ppo, and cmd/train-hierarchical's optional
// -metrics-out flag, so a training run's progress can be plotted or
// analyzed with any external tool instead of only read from stdout —
// see docs/plans/15-agent-and-training-visualization.md.
package metrics

import (
	"encoding/csv"
	"fmt"
	"os"
)

// CSVWriter writes one row per training epoch to a CSV file, starting
// with a header row written once at construction.
type CSVWriter struct {
	file   *os.File
	writer *csv.Writer
}

// NewCSVWriter creates (or truncates) the file at path and writes
// header as its first row.
func NewCSVWriter(path string, header []string) (*CSVWriter, error) {
	file, err := os.Create(path)
	if err != nil {
		return nil, fmt.Errorf("metrics: creating %s: %w", path, err)
	}

	writer := csv.NewWriter(file)
	if err := writer.Write(header); err != nil {
		_ = file.Close()
		return nil, fmt.Errorf("metrics: writing header to %s: %w", path, err)
	}

	return &CSVWriter{file: file, writer: writer}, nil
}

// WriteRow writes one row of values, converting each to its string
// form via fmt.Sprint.
func (w *CSVWriter) WriteRow(values ...any) error {
	row := make([]string, len(values))
	for i, value := range values {
		row[i] = fmt.Sprint(value)
	}
	if err := w.writer.Write(row); err != nil {
		return fmt.Errorf("metrics: writing row: %w", err)
	}
	return nil
}

// Close flushes any buffered rows and closes the underlying file.
func (w *CSVWriter) Close() error {
	w.writer.Flush()
	if err := w.writer.Error(); err != nil {
		_ = w.file.Close()
		return fmt.Errorf("metrics: flushing: %w", err)
	}
	if err := w.file.Close(); err != nil {
		return fmt.Errorf("metrics: closing: %w", err)
	}
	return nil
}
