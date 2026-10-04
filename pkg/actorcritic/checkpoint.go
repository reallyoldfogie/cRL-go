package actorcritic

import (
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"
	"github.com/reallyoldfogie/cRL-go/pkg/mat"
)

// CheckpointSchemaVersion identifies the on-disk shape of a saved
// checkpoint. Unlike pkg/policy's identically-named type, this package
// has no pre-versioning checkpoints to stay backward compatible with
// (this network didn't exist before checkpoint versioning did), so Load
// always requires and validates both SchemaVersion and EnvironmentID.
type CheckpointSchemaVersion int

// checkpointSchemaVersionCurrent is the schema version Save writes.
// Version 1 (exactly two hidden layers, named w0/b0/w1/b1 rather than a
// "hidden" list) is still accepted by Load — see migrateSchemaV1 — now
// that hidden-layer depth is configurable
// (docs/plans/20-configurable-depth-step-memory-and-dynamic-architecture.md,
// Part A); Save never writes version 1 again.
const checkpointSchemaVersionCurrent CheckpointSchemaVersion = 2

// hiddenLayerData is the on-disk JSON representation of one hidden
// Layer's flattened weight/bias data.
type hiddenLayerData struct {
	W []float32 `json:"w"`
	B []float32 `json:"b"`
}

// checkpointData is the on-disk JSON representation of a Params: its
// schema version, the environment/action-space identifier it was
// trained against, its run-progress Metadata, its layer sizes, plus the
// flattened weight/bias data for every hidden layer and both heads
// (including the value head).
//
// W0/B0/W1/B1 are schema-version-1-only fields: Load reads them when
// SchemaVersion is 1 (see migrateSchemaV1) and Save never writes them.
// They're kept here, rather than in a separate type, so decoding a v1
// file needs no second pass over the raw JSON.
type checkpointData struct {
	SchemaVersion CheckpointSchemaVersion `json:"schema_version"`
	EnvironmentID string                  `json:"environment_id"`
	Metadata      checkpoint.Metadata     `json:"metadata"`

	InputSize  int `json:"input_size"`
	HiddenSize int `json:"hidden_size"`
	OutputSize int `json:"output_size"`

	Hidden []hiddenLayerData `json:"hidden,omitempty"`

	Wpi []float32 `json:"wpi"`
	Bpi []float32 `json:"bpi"`
	Wv  []float32 `json:"wv"`
	Bv  []float32 `json:"bv"`

	W0 []float32 `json:"w0,omitempty"`
	B0 []float32 `json:"b0,omitempty"`
	W1 []float32 `json:"w1,omitempty"`
	B1 []float32 `json:"b1,omitempty"`
}

// migrateSchemaV1 rewrites a decoded schema-1 checkpointData's W0/B0,
// W1/B1 fields into the current Hidden-list shape: a schema-1 file is
// always exactly two hidden layers, the first HiddenSize x InputSize
// (W0/B0) and the second HiddenSize x HiddenSize (W1/B1) — exactly what
// NewParams(..., numHiddenLayers=2) still produces — so this mapping is
// lossless, not an approximation.
func migrateSchemaV1(data checkpointData) checkpointData {
	data.SchemaVersion = checkpointSchemaVersionCurrent
	data.Hidden = []hiddenLayerData{
		{W: data.W0, B: data.B0},
		{W: data.W1, B: data.B1},
	}
	data.W0, data.B0, data.W1, data.B1 = nil, nil, nil, nil
	return data
}

// Save writes p's learned weights and biases to w as JSON, tagged with
// environmentID (e.g. "snake:36") and metadata (epoch, best return,
// total update count), so a future Load call can reject restoring this
// checkpoint into an incompatible environment and a resuming trainer
// can continue its counters from where this checkpoint left off.
func (p *Params) Save(w io.Writer, environmentID string, metadata checkpoint.Metadata) error {
	hidden := make([]hiddenLayerData, len(p.Hidden))
	for i, layer := range p.Hidden {
		hidden[i] = hiddenLayerData{W: layer.W.Data, B: layer.B.Data}
	}

	data := checkpointData{
		SchemaVersion: checkpointSchemaVersionCurrent,
		EnvironmentID: environmentID,
		Metadata:      metadata,
		InputSize:     p.InputSize(),
		HiddenSize:    p.HiddenSize(),
		OutputSize:    p.OutputSize(),
		Hidden:        hidden,
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
// should be copied into, for the shape-validated copy loop in
// paramsFromCheckpointData.
type checkpointField struct {
	name string
	dst  *mat.Matrix
	src  []float32
}

// copyField copies f.src into f.dst, rejecting the checkpoint if the
// saved data's length doesn't match the matrix's shape (e.g. a
// truncated or hand-edited file).
func copyField(f checkpointField) error {
	if len(f.src) != len(f.dst.Data) {
		return fmt.Errorf(
			"actorcritic: loading checkpoint: %s has %d values, want %d",
			f.name, len(f.src), len(f.dst.Data),
		)
	}
	copy(f.dst.Data, f.src)
	return nil
}

// paramsFromCheckpointData reconstructs a Params from data (already
// migrated to the current schema if it was schema 1), validating every
// saved matrix's data against the shape implied by data's saved layer
// sizes.
func paramsFromCheckpointData(data checkpointData) (*Params, error) {
	if len(data.Hidden) == 0 {
		return nil, fmt.Errorf("actorcritic: loading checkpoint: no hidden layers")
	}

	hidden := make([]Layer, len(data.Hidden))
	for i, layerData := range data.Hidden {
		fanIn := data.HiddenSize
		if i == 0 {
			fanIn = data.InputSize
		}
		w := mat.New(data.HiddenSize, fanIn)
		b := mat.New(data.HiddenSize, 1)
		if err := copyField(checkpointField{name: fmt.Sprintf("hidden[%d].w", i), dst: w, src: layerData.W}); err != nil {
			return nil, err
		}
		if err := copyField(checkpointField{name: fmt.Sprintf("hidden[%d].b", i), dst: b, src: layerData.B}); err != nil {
			return nil, err
		}
		hidden[i] = Layer{W: w, B: b}
	}

	valueHeadSize := 1
	params := &Params{
		Hidden: hidden,
		Wpi:    mat.New(data.OutputSize, data.HiddenSize),
		Bpi:    mat.New(data.OutputSize, 1),
		Wv:     mat.New(valueHeadSize, data.HiddenSize),
		Bv:     mat.New(valueHeadSize, 1),
	}

	for _, field := range []checkpointField{
		{name: "wpi", dst: params.Wpi, src: data.Wpi},
		{name: "bpi", dst: params.Bpi, src: data.Bpi},
		{name: "wv", dst: params.Wv, src: data.Wv},
		{name: "bv", dst: params.Bv, src: data.Bv},
	} {
		if err := copyField(field); err != nil {
			return nil, err
		}
	}

	return params, nil
}

// Load reads a Params previously written by Save from r, reconstructing
// each matrix at the shape implied by the saved layer sizes and
// rejecting the checkpoint if any saved matrix's data doesn't match that
// shape (e.g. a truncated or hand-edited file), or if its EnvironmentID
// doesn't match expectedEnvironmentID. A schema-1 file (exactly two
// hidden layers, saved before hidden-layer depth was configurable) is
// migrated transparently — see migrateSchemaV1 — rather than rejected.
// The returned checkpoint.Metadata is the checkpoint's saved
// run-progress information.
func Load(r io.Reader, expectedEnvironmentID string) (*Params, checkpoint.Metadata, error) {
	var data checkpointData
	if err := json.NewDecoder(r).Decode(&data); err != nil {
		return nil, checkpoint.Metadata{}, fmt.Errorf("actorcritic: loading checkpoint: %w", err)
	}

	switch data.SchemaVersion {
	case 1:
		data = migrateSchemaV1(data)
	case checkpointSchemaVersionCurrent:
		// already current
	default:
		return nil, checkpoint.Metadata{}, fmt.Errorf(
			"actorcritic: loading checkpoint: unsupported schema version %d",
			data.SchemaVersion,
		)
	}

	if data.EnvironmentID != expectedEnvironmentID {
		return nil, checkpoint.Metadata{}, fmt.Errorf(
			"actorcritic: loading checkpoint: saved for environment %q, want %q",
			data.EnvironmentID, expectedEnvironmentID,
		)
	}

	params, err := paramsFromCheckpointData(data)
	if err != nil {
		return nil, checkpoint.Metadata{}, err
	}

	return params, data.Metadata, nil
}

// SaveFile is a convenience wrapper around Save that writes to the file
// at path, creating or truncating it as needed.
func SaveFile(path string, p *Params, environmentID string, metadata checkpoint.Metadata) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("actorcritic: saving checkpoint: %w", err)
	}

	if err := p.Save(file, environmentID, metadata); err != nil {
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
func LoadFile(path string, expectedEnvironmentID string) (*Params, checkpoint.Metadata, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, checkpoint.Metadata{}, fmt.Errorf("actorcritic: loading checkpoint: %w", err)
	}
	defer file.Close()

	return Load(file, expectedEnvironmentID)
}
