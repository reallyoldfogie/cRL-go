package hierarchical

import (
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"
	"github.com/reallyoldfogie/cRL-go/pkg/mat"
)

// CheckpointSchemaVersion identifies the on-disk shape of a saved
// hierarchical checkpoint. Like pkg/actorcritic's identically-named
// type, this package has no pre-versioning checkpoints to stay
// backward compatible with, so Load always requires and validates it.
type CheckpointSchemaVersion int

// checkpointSchemaVersionCurrent is the schema version Save writes.
// Version 1 (exactly two hidden layers per network, named
// w0/b0/w1/b1 rather than a "hidden" list) is still accepted by Load —
// see migrateNetworkSchemaV1 — now that pkg/actorcritic's hidden-layer
// depth is configurable
// (docs/plans/20-configurable-depth-step-memory-and-dynamic-architecture.md,
// Part A); Save never writes version 1 again.
const checkpointSchemaVersionCurrent CheckpointSchemaVersion = 2

// hiddenLayerData mirrors pkg/actorcritic's identically-named type
// (duplicated rather than shared, for the same reason this package's
// other checkpoint types already are — see this file's other doc
// comments): the on-disk JSON representation of one hidden Layer's
// flattened weight/bias data.
type hiddenLayerData struct {
	W []float32 `json:"w"`
	B []float32 `json:"b"`
}

// networkCheckpointData is the on-disk JSON representation of one
// actorcritic.Params' layer sizes and weight/bias data, reusing the
// same field layout as pkg/actorcritic.checkpointData. It carries no
// EnvironmentID/Metadata of its own — checkpointData below carries
// those once for the whole N+1-network generation.
//
// W0/B0/W1/B1 are schema-version-1-only fields, read when
// SchemaVersion is 1 (see migrateNetworkSchemaV1); Save never writes
// them.
type networkCheckpointData struct {
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

// migrateNetworkSchemaV1 mirrors pkg/actorcritic's migrateSchemaV1:
// a schema-1 network is always exactly two hidden layers, so mapping
// its W0/B0, W1/B1 fields onto a 2-element Hidden list is lossless.
func migrateNetworkSchemaV1(data networkCheckpointData) networkCheckpointData {
	data.Hidden = []hiddenLayerData{
		{W: data.W0, B: data.B0},
		{W: data.W1, B: data.B1},
	}
	data.W0, data.B0, data.W1, data.B1 = nil, nil, nil, nil
	return data
}

// checkpointData is the on-disk JSON representation of a Trainer's
// full N+1-network state: the meta-controller's and every sub-policy's
// weights, saved as one atomic document rather than N+1 independent
// files, so a generation can never be resumed from a mix of checkpoints
// saved at different epochs. See
// docs/archive/plans/14-hierarchical-checkpointing.md for why this
// shape was chosen over per-network files. Subs is ordered by
// ascending Subgoal (0, 1, ..., NumSubgoals-1), so Load can reconstruct
// the map without needing a subgoal index embedded in each element.
type checkpointData struct {
	SchemaVersion CheckpointSchemaVersion `json:"schema_version"`
	EnvironmentID string                  `json:"environment_id"`
	// NumSubgoals is validated on Load in addition to EnvironmentID:
	// a checkpoint's per-subgoal network shapes depend on it, so
	// resuming with a different -num-subgoals than a checkpoint was
	// trained with must be rejected outright rather than silently
	// producing mismatched shapes.
	NumSubgoals int                 `json:"num_subgoals"`
	Metadata    checkpoint.Metadata `json:"metadata"`

	Meta networkCheckpointData   `json:"meta"`
	Subs []networkCheckpointData `json:"subs"`
}

// toNetworkData copies p's layer sizes and weight/bias data into the
// on-disk representation used by Save.
func toNetworkData(p *actorcritic.Params) networkCheckpointData {
	hidden := make([]hiddenLayerData, len(p.Hidden))
	for i, layer := range p.Hidden {
		hidden[i] = hiddenLayerData{W: layer.W.Data, B: layer.B.Data}
	}

	return networkCheckpointData{
		InputSize:  p.InputSize(),
		HiddenSize: p.HiddenSize(),
		OutputSize: p.OutputSize(),
		Hidden:     hidden,
		Wpi:        p.Wpi.Data,
		Bpi:        p.Bpi.Data,
		Wv:         p.Wv.Data,
		Bv:         p.Bv.Data,
	}
}

// networkField pairs a saved flat data slice with the matrix it should
// be copied into, for the shape-validated copy loop in fromNetworkData,
// mirroring pkg/actorcritic.checkpointField.
type networkField struct {
	name string
	dst  *mat.Matrix
	src  []float32
}

// fromNetworkData reconstructs an actorcritic.Params from data (already
// migrated to the current schema if it was schema 1 — see
// migrateNetworkSchemaV1), rejecting it if any saved matrix's data
// doesn't match the shape implied by data's saved layer sizes (e.g. a
// truncated or hand-edited file). networkName (e.g. "meta-controller"
// or "sub-policy 2") identifies which network a shape-mismatch error
// refers to.
func fromNetworkData(data networkCheckpointData, networkName string) (*actorcritic.Params, error) {
	if len(data.Hidden) == 0 {
		return nil, fmt.Errorf("hierarchical: loading checkpoint: %s has no hidden layers", networkName)
	}

	hidden := make([]actorcritic.Layer, len(data.Hidden))
	for i, layerData := range data.Hidden {
		fanIn := data.HiddenSize
		if i == 0 {
			fanIn = data.InputSize
		}
		w := mat.New(data.HiddenSize, fanIn)
		b := mat.New(data.HiddenSize, 1)
		if err := copyNetworkField(networkField{name: fmt.Sprintf("hidden[%d].w", i), dst: w, src: layerData.W}, networkName); err != nil {
			return nil, err
		}
		if err := copyNetworkField(networkField{name: fmt.Sprintf("hidden[%d].b", i), dst: b, src: layerData.B}, networkName); err != nil {
			return nil, err
		}
		hidden[i] = actorcritic.Layer{W: w, B: b}
	}

	valueHeadSize := 1
	params := &actorcritic.Params{
		Hidden: hidden,
		Wpi:    mat.New(data.OutputSize, data.HiddenSize),
		Bpi:    mat.New(data.OutputSize, 1),
		Wv:     mat.New(valueHeadSize, data.HiddenSize),
		Bv:     mat.New(valueHeadSize, 1),
	}

	for _, field := range []networkField{
		{name: "wpi", dst: params.Wpi, src: data.Wpi},
		{name: "bpi", dst: params.Bpi, src: data.Bpi},
		{name: "wv", dst: params.Wv, src: data.Wv},
		{name: "bv", dst: params.Bv, src: data.Bv},
	} {
		if err := copyNetworkField(field, networkName); err != nil {
			return nil, err
		}
	}
	return params, nil
}

// copyNetworkField copies f.src into f.dst, rejecting the checkpoint if
// the saved data's length doesn't match the matrix's shape.
func copyNetworkField(f networkField, networkName string) error {
	if len(f.src) != len(f.dst.Data) {
		return fmt.Errorf(
			"hierarchical: loading checkpoint: %s %s has %d values, want %d",
			networkName, f.name, len(f.src), len(f.dst.Data),
		)
	}
	copy(f.dst.Data, f.src)
	return nil
}

// Save writes tr's meta-controller's and every sub-policy's weights to
// w as one atomic JSON document, tagged with environmentID and
// metadata, so a future Load call can reject restoring this checkpoint
// into an incompatible environment or subgoal count, and a resuming
// trainer can continue its counters from where this checkpoint left
// off. See Load for the corresponding read path.
func (tr *Trainer) Save(w io.Writer, environmentID string, metadata checkpoint.Metadata) error {
	subs := make([]networkCheckpointData, tr.cfg.NumSubgoals)
	for i := range tr.cfg.NumSubgoals {
		subs[i] = toNetworkData(tr.subParams[Subgoal(i)])
	}

	data := checkpointData{
		SchemaVersion: checkpointSchemaVersionCurrent,
		EnvironmentID: environmentID,
		NumSubgoals:   tr.cfg.NumSubgoals,
		Metadata:      metadata,
		Meta:          toNetworkData(tr.metaParams),
		Subs:          subs,
	}

	if err := json.NewEncoder(w).Encode(data); err != nil {
		return fmt.Errorf("hierarchical: saving checkpoint: %w", err)
	}
	return nil
}

// Load reads a checkpoint previously written by Trainer.Save from r,
// returning the reconstructed meta-controller and sub-policy Params
// (ready to pass to New via InitialParams) plus the checkpoint's saved
// run-progress Metadata. It rejects a checkpoint whose EnvironmentID
// doesn't match expectedEnvironmentID or whose saved subgoal count
// doesn't match expectedNumSubgoals.
func Load(r io.Reader, expectedEnvironmentID string, expectedNumSubgoals int) (*actorcritic.Params, map[Subgoal]*actorcritic.Params, checkpoint.Metadata, error) {
	var data checkpointData
	if err := json.NewDecoder(r).Decode(&data); err != nil {
		return nil, nil, checkpoint.Metadata{}, fmt.Errorf("hierarchical: loading checkpoint: %w", err)
	}

	switch data.SchemaVersion {
	case 1:
		data.Meta = migrateNetworkSchemaV1(data.Meta)
		for i, sub := range data.Subs {
			data.Subs[i] = migrateNetworkSchemaV1(sub)
		}
	case checkpointSchemaVersionCurrent:
		// already current
	default:
		return nil, nil, checkpoint.Metadata{}, fmt.Errorf(
			"hierarchical: loading checkpoint: unsupported schema version %d",
			data.SchemaVersion,
		)
	}

	if data.EnvironmentID != expectedEnvironmentID {
		return nil, nil, checkpoint.Metadata{}, fmt.Errorf(
			"hierarchical: loading checkpoint: saved for environment %q, want %q",
			data.EnvironmentID, expectedEnvironmentID,
		)
	}
	if data.NumSubgoals != expectedNumSubgoals {
		return nil, nil, checkpoint.Metadata{}, fmt.Errorf(
			"hierarchical: loading checkpoint: saved with %d subgoals, want %d",
			data.NumSubgoals, expectedNumSubgoals,
		)
	}
	if len(data.Subs) != data.NumSubgoals {
		return nil, nil, checkpoint.Metadata{}, fmt.Errorf(
			"hierarchical: loading checkpoint: has %d sub-policies, want %d",
			len(data.Subs), data.NumSubgoals,
		)
	}

	metaParams, err := fromNetworkData(data.Meta, "meta-controller")
	if err != nil {
		return nil, nil, checkpoint.Metadata{}, err
	}

	subParams := make(map[Subgoal]*actorcritic.Params, data.NumSubgoals)
	for i, subData := range data.Subs {
		params, err := fromNetworkData(subData, fmt.Sprintf("sub-policy %d", i))
		if err != nil {
			return nil, nil, checkpoint.Metadata{}, err
		}
		subParams[Subgoal(i)] = params
	}

	return metaParams, subParams, data.Metadata, nil
}

// SaveFile is a convenience wrapper around Save that writes to the file
// at path, creating or truncating it as needed.
func (tr *Trainer) SaveFile(path string, environmentID string, metadata checkpoint.Metadata) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("hierarchical: saving checkpoint: %w", err)
	}

	if err := tr.Save(file, environmentID, metadata); err != nil {
		_ = file.Close()
		return err
	}

	if err := file.Close(); err != nil {
		return fmt.Errorf("hierarchical: saving checkpoint: %w", err)
	}
	return nil
}

// LoadFile is a convenience wrapper around Load that reads from the
// file at path.
func LoadFile(path string, expectedEnvironmentID string, expectedNumSubgoals int) (*actorcritic.Params, map[Subgoal]*actorcritic.Params, checkpoint.Metadata, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, checkpoint.Metadata{}, fmt.Errorf("hierarchical: loading checkpoint: %w", err)
	}
	defer file.Close()

	return Load(file, expectedEnvironmentID, expectedNumSubgoals)
}
