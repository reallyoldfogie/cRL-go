package policy

import (
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/reallyoldfogie/cRL-go/pkg/mat"
)

// checkpointData is the on-disk JSON representation of a Params: its
// three layer sizes, plus the flattened weight/bias data for every
// matrix. The sizes are included (rather than inferred from data length
// alone) so Load can give a precise error when a checkpoint doesn't
// match the architecture it's being restored into.
type checkpointData struct {
	InputSize  int `json:"input_size"`
	HiddenSize int `json:"hidden_size"`
	OutputSize int `json:"output_size"`

	W0 []float32 `json:"w0"`
	B0 []float32 `json:"b0"`
	W1 []float32 `json:"w1"`
	B1 []float32 `json:"b1"`
	W2 []float32 `json:"w2"`
	B2 []float32 `json:"b2"`
}

// Save writes p's learned weights and biases to w as JSON, so a future
// process can resume training (or run inference) from the same point via
// Load, instead of always starting from a fresh Xavier/Glorot
// initialization.
func (p *Params) Save(w io.Writer) error {
	data := checkpointData{
		InputSize:  p.InputSize(),
		HiddenSize: p.HiddenSize(),
		OutputSize: p.OutputSize(),
		W0:         p.W0.Data,
		B0:         p.B0.Data,
		W1:         p.W1.Data,
		B1:         p.B1.Data,
		W2:         p.W2.Data,
		B2:         p.B2.Data,
	}

	if err := json.NewEncoder(w).Encode(data); err != nil {
		return fmt.Errorf("policy: saving checkpoint: %w", err)
	}
	return nil
}

// checkpointField pairs a saved flat data slice with the matrix it
// should be copied into, for the shape-validated copy loop in Load.
type checkpointField struct {
	name string
	dst  *mat.Matrix
	src  []float32
}

// Load reads a Params previously written by Save from r, reconstructing
// each matrix at the shape implied by the saved layer sizes and
// rejecting the checkpoint if any saved matrix's data doesn't match that
// shape (e.g. a truncated or hand-edited file).
func Load(r io.Reader) (*Params, error) {
	var data checkpointData
	if err := json.NewDecoder(r).Decode(&data); err != nil {
		return nil, fmt.Errorf("policy: loading checkpoint: %w", err)
	}

	params := &Params{
		W0: mat.New(data.HiddenSize, data.InputSize),
		B0: mat.New(data.HiddenSize, 1),
		W1: mat.New(data.HiddenSize, data.HiddenSize),
		B1: mat.New(data.HiddenSize, 1),
		W2: mat.New(data.OutputSize, data.HiddenSize),
		B2: mat.New(data.OutputSize, 1),
	}

	fields := []checkpointField{
		{name: "w0", dst: params.W0, src: data.W0},
		{name: "b0", dst: params.B0, src: data.B0},
		{name: "w1", dst: params.W1, src: data.W1},
		{name: "b1", dst: params.B1, src: data.B1},
		{name: "w2", dst: params.W2, src: data.W2},
		{name: "b2", dst: params.B2, src: data.B2},
	}
	for _, field := range fields {
		if len(field.src) != len(field.dst.Data) {
			return nil, fmt.Errorf(
				"policy: loading checkpoint: %s has %d values, want %d",
				field.name, len(field.src), len(field.dst.Data),
			)
		}
		copy(field.dst.Data, field.src)
	}

	return params, nil
}

// SaveFile is a convenience wrapper around Save that writes to the file
// at path, creating or truncating it as needed.
func SaveFile(path string, p *Params) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("policy: saving checkpoint: %w", err)
	}

	if err := p.Save(file); err != nil {
		_ = file.Close()
		return err
	}

	if err := file.Close(); err != nil {
		return fmt.Errorf("policy: saving checkpoint: %w", err)
	}
	return nil
}

// LoadFile is a convenience wrapper around Load that reads from the file
// at path.
func LoadFile(path string) (*Params, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("policy: loading checkpoint: %w", err)
	}
	defer file.Close()

	return Load(file)
}
