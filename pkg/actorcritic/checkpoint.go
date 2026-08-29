package actorcritic

import (
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/reallyoldfogie/cRL-go/pkg/mat"
)

// CheckpointSchemaVersion identifies the on-disk shape of a saved
// checkpoint. Unlike pkg/policy's identically-named type, this package
// has no pre-versioning checkpoints to stay backward compatible with
// (this network didn't exist before checkpoint versioning did), so Load
// always requires and validates both SchemaVersion and EnvironmentID.
type CheckpointSchemaVersion int

// checkpointSchemaVersionCurrent is the only schema version this
// package has ever written.
const checkpointSchemaVersionCurrent CheckpointSchemaVersion = 1

// checkpointData is the on-disk JSON representation of a Params: its
// schema version, the environment/action-space identifier it was
// trained against, its three layer sizes, plus the flattened weight/bias
// data for every matrix (including the value head).
type checkpointData struct {
	SchemaVersion CheckpointSchemaVersion `json:"schema_version"`
	EnvironmentID string                  `json:"environment_id"`

	InputSize  int `json:"input_size"`
	HiddenSize int `json:"hidden_size"`
	OutputSize int `json:"output_size"`

	W0  []float32 `json:"w0"`
	B0  []float32 `json:"b0"`
	W1  []float32 `json:"w1"`
	B1  []float32 `json:"b1"`
	Wpi []float32 `json:"wpi"`
	Bpi []float32 `json:"bpi"`
	Wv  []float32 `json:"wv"`
	Bv  []float32 `json:"bv"`
}

// Save writes p's learned weights and biases to w as JSON, tagged with
// environmentID (e.g. "snake:36"), so a future Load call can reject
// restoring this checkpoint into an incompatible environment.
func (p *Params) Save(w io.Writer, environmentID string) error {
	data := checkpointData{
		SchemaVersion: checkpointSchemaVersionCurrent,
		EnvironmentID: environmentID,
		InputSize:     p.InputSize(),
		HiddenSize:    p.HiddenSize(),
		OutputSize:    p.OutputSize(),
		W0:            p.W0.Data,
		B0:            p.B0.Data,
		W1:            p.W1.Data,
		B1:            p.B1.Data,
		Wpi:           p.Wpi.Data,
		Bpi:           p.Bpi.Data,
		Wv:            p.Wv.Data,
		Bv:            p.Bv.Data,
	}

	if err := json.NewEncoder(w).Encode(data); err != nil {
		return fmt.Errorf("actorcritic: saving checkpoint: %w", err)
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
// shape (e.g. a truncated or hand-edited file), or if its EnvironmentID
// doesn't match expectedEnvironmentID.
func Load(r io.Reader, expectedEnvironmentID string) (*Params, error) {
	var data checkpointData
	if err := json.NewDecoder(r).Decode(&data); err != nil {
		return nil, fmt.Errorf("actorcritic: loading checkpoint: %w", err)
	}
	if data.SchemaVersion != checkpointSchemaVersionCurrent {
		return nil, fmt.Errorf(
			"actorcritic: loading checkpoint: unsupported schema version %d",
			data.SchemaVersion,
		)
	}

	if data.EnvironmentID != expectedEnvironmentID {
		return nil, fmt.Errorf(
			"actorcritic: loading checkpoint: saved for environment %q, want %q",
			data.EnvironmentID, expectedEnvironmentID,
		)
	}

	valueHeadSize := 1
	params := &Params{
		W0:  mat.New(data.HiddenSize, data.InputSize),
		B0:  mat.New(data.HiddenSize, 1),
		W1:  mat.New(data.HiddenSize, data.HiddenSize),
		B1:  mat.New(data.HiddenSize, 1),
		Wpi: mat.New(data.OutputSize, data.HiddenSize),
		Bpi: mat.New(data.OutputSize, 1),
		Wv:  mat.New(valueHeadSize, data.HiddenSize),
		Bv:  mat.New(valueHeadSize, 1),
	}

	fields := []checkpointField{
		{name: "w0", dst: params.W0, src: data.W0},
		{name: "b0", dst: params.B0, src: data.B0},
		{name: "w1", dst: params.W1, src: data.W1},
		{name: "b1", dst: params.B1, src: data.B1},
		{name: "wpi", dst: params.Wpi, src: data.Wpi},
		{name: "bpi", dst: params.Bpi, src: data.Bpi},
		{name: "wv", dst: params.Wv, src: data.Wv},
		{name: "bv", dst: params.Bv, src: data.Bv},
	}
	for _, field := range fields {
		if len(field.src) != len(field.dst.Data) {
			return nil, fmt.Errorf(
				"actorcritic: loading checkpoint: %s has %d values, want %d",
				field.name, len(field.src), len(field.dst.Data),
			)
		}
		copy(field.dst.Data, field.src)
	}

	return params, nil
}

// SaveFile is a convenience wrapper around Save that writes to the file
// at path, creating or truncating it as needed.
func SaveFile(path string, p *Params, environmentID string) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("actorcritic: saving checkpoint: %w", err)
	}

	if err := p.Save(file, environmentID); err != nil {
		_ = file.Close()
		return err
	}

	if err := file.Close(); err != nil {
		return fmt.Errorf("actorcritic: saving checkpoint: %w", err)
	}
	return nil
}

// LoadFile is a convenience wrapper around Load that reads from the file
// at path.
func LoadFile(path string, expectedEnvironmentID string) (*Params, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("actorcritic: loading checkpoint: %w", err)
	}
	defer file.Close()

	return Load(file, expectedEnvironmentID)
}
