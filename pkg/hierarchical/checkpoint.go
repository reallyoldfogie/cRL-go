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

// checkpointSchemaVersionCurrent is the only schema version this
// package has ever written.
const checkpointSchemaVersionCurrent CheckpointSchemaVersion = 1

// networkCheckpointData is the on-disk JSON representation of one
// actorcritic.Params' layer sizes and weight/bias data, reusing the
// same field layout as pkg/actorcritic.checkpointData's weight fields.
// It carries no EnvironmentID/Metadata of its own — checkpointData
// below carries those once for the whole N+1-network generation.
type networkCheckpointData struct {
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
	return networkCheckpointData{
		InputSize:  p.InputSize(),
		HiddenSize: p.HiddenSize(),
		OutputSize: p.OutputSize(),
		W0:         p.W0.Data,
		B0:         p.B0.Data,
		W1:         p.W1.Data,
		B1:         p.B1.Data,
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

// fromNetworkData reconstructs an actorcritic.Params from data,
// rejecting it if any saved matrix's data doesn't match the shape
// implied by data's saved layer sizes (e.g. a truncated or hand-edited
// file). networkName (e.g. "meta-controller" or "sub-policy 2")
// identifies which network a shape-mismatch error refers to.
func fromNetworkData(data networkCheckpointData, networkName string) (*actorcritic.Params, error) {
	valueHeadSize := 1
	params := &actorcritic.Params{
		W0:  mat.New(data.HiddenSize, data.InputSize),
		B0:  mat.New(data.HiddenSize, 1),
		W1:  mat.New(data.HiddenSize, data.HiddenSize),
		B1:  mat.New(data.HiddenSize, 1),
		Wpi: mat.New(data.OutputSize, data.HiddenSize),
		Bpi: mat.New(data.OutputSize, 1),
		Wv:  mat.New(valueHeadSize, data.HiddenSize),
		Bv:  mat.New(valueHeadSize, 1),
	}

	fields := []networkField{
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
				"hierarchical: loading checkpoint: %s %s has %d values, want %d",
				networkName, field.name, len(field.src), len(field.dst.Data),
			)
		}
		copy(field.dst.Data, field.src)
	}
	return params, nil
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
	if data.SchemaVersion != checkpointSchemaVersionCurrent {
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
