// Package decisionlog defines a shared, newline-delimited-JSON format
// for recording one environment step's full decision context — not
// just the outcome, but the rl.Decision that produced it — so a
// completed run can be replayed or analyzed later instead of only
// observed live. This is the shared format docs/plans/16-decision-
// auditing-and-explainability.md's option 3 and
// docs/plans/15-agent-and-training-visualization.md's option (b).2 both
// called for rather than inventing two incompatible ones; see
// docs/plans/18-shared-decision-logging-format.md.
package decisionlog

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

// Record is one environment step's recorded decision context.
type Record struct {
	Step        int         `json:"step"`
	Observation []float32   `json:"observation"`
	Decision    rl.Decision `json:"decision"`
	Reward      float32     `json:"reward"`
	Done        bool        `json:"done"`
	// Render is an optional human-readable snapshot (e.g. from an
	// environment's Render() method) of the state Observation
	// describes, letting a replay viewer redraw a step directly rather
	// than re-deriving grid state from a raw feature vector. Nil for a
	// caller with no such snapshot available.
	Render []string `json:"render,omitempty"`
	// Extra carries whatever context is specific to the algorithm that
	// produced Decision but has no fixed field here, since this
	// package stays environment- and algorithm-agnostic by design. For
	// example, pkg/hierarchical's caller populates this with the
	// active subgoal and, on a meta-decision step, the meta-
	// controller's own rl.Decision; pkg/policy/pkg/actorcritic callers
	// leave it nil.
	Extra map[string]any `json:"extra,omitempty"`
}

// Writer appends one JSON-encoded Record per line to an underlying
// io.Writer (newline-delimited JSON), via json.Encoder's own
// one-value-per-Encode-call framing.
type Writer struct {
	enc *json.Encoder
}

// NewWriter builds a Writer over w.
func NewWriter(w io.Writer) *Writer {
	return &Writer{enc: json.NewEncoder(w)}
}

// Write encodes rec as one line.
func (w *Writer) Write(rec Record) error {
	if err := w.enc.Encode(rec); err != nil {
		return fmt.Errorf("decisionlog: writing record: %w", err)
	}
	return nil
}

// FileWriter is a Writer plus ownership of the file it writes to,
// mirroring pkg/metrics.CSVWriter's create-or-truncate-then-Close
// shape.
type FileWriter struct {
	file   *os.File
	writer *Writer
}

// NewFileWriter creates (or truncates) the file at path for writing.
func NewFileWriter(path string) (*FileWriter, error) {
	file, err := os.Create(path)
	if err != nil {
		return nil, fmt.Errorf("decisionlog: creating %s: %w", path, err)
	}
	return &FileWriter{file: file, writer: NewWriter(file)}, nil
}

// Write encodes rec as one line.
func (w *FileWriter) Write(rec Record) error {
	return w.writer.Write(rec)
}

// Close closes the underlying file.
func (w *FileWriter) Close() error {
	if err := w.file.Close(); err != nil {
		return fmt.Errorf("decisionlog: closing: %w", err)
	}
	return nil
}

// ReadAll decodes every Record from r (a full newline-delimited-JSON
// stream) into a slice, for a replay viewer or offline analysis to load
// a completed run's log in one call. Returns an empty, non-nil slice
// (never an error) if r contains no records.
func ReadAll(r io.Reader) ([]Record, error) {
	records := make([]Record, 0)

	dec := json.NewDecoder(r)
	for {
		var rec Record
		err := dec.Decode(&rec)
		if errors.Is(err, io.EOF) {
			return records, nil
		}
		if err != nil {
			return nil, fmt.Errorf("decisionlog: reading record: %w", err)
		}
		records = append(records, rec)
	}
}
