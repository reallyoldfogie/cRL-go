package actorcritic

import (
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"
	"github.com/reallyoldfogie/cRL-go/pkg/mat"
)

// RecurrentCheckpointSchemaVersion identifies the on-disk shape of a
// saved RecurrentParams checkpoint. This is a wholly separate format
// from Params' own CheckpointSchemaVersion (different file shape, not a
// newer version of the same one), since nothing of this shape existed
// before RecurrentParams did, so there is no migration to support.
type RecurrentCheckpointSchemaVersion int

// recurrentCheckpointSchemaVersionCurrent is the only schema version
// this file has ever written.
const recurrentCheckpointSchemaVersionCurrent RecurrentCheckpointSchemaVersion = 1

// lstmLayerData is the on-disk JSON representation of an LSTMLayer's
// flattened weight/bias data.
type lstmLayerData struct {
	WxF []float32 `json:"wxf"`
	WhF []float32 `json:"whf"`
	BF  []float32 `json:"bf"`
	WxI []float32 `json:"wxi"`
	WhI []float32 `json:"whi"`
	BI  []float32 `json:"bi"`
	WxO []float32 `json:"wxo"`
	WhO []float32 `json:"who"`
	BO  []float32 `json:"bo"`
	WxG []float32 `json:"wxg"`
	WhG []float32 `json:"whg"`
	BG  []float32 `json:"bg"`
}

// recurrentCheckpointData is the on-disk JSON representation of a
// RecurrentParams: its schema version, the environment/action-space
// identifier it was trained against, its run-progress Metadata, the
// Trunk's layer sizes and flattened weight/bias data (the same shape as
// Params' own checkpointData), and the LSTM's flattened weight/bias
// data.
type recurrentCheckpointData struct {
	SchemaVersion RecurrentCheckpointSchemaVersion `json:"schema_version"`
	EnvironmentID string                           `json:"environment_id"`
	Metadata      checkpoint.Metadata              `json:"metadata"`

	InputSize  int `json:"input_size"`
	HiddenSize int `json:"hidden_size"`
	OutputSize int `json:"output_size"`

	Hidden []hiddenLayerData `json:"hidden"`

	Wpi []float32 `json:"wpi"`
	Bpi []float32 `json:"bpi"`
	Wv  []float32 `json:"wv"`
	Bv  []float32 `json:"bv"`

	LSTM lstmLayerData `json:"lstm"`
}

// Save writes p's learned weights and biases (both Trunk's and LSTM's)
// to w as JSON, tagged with environmentID and metadata, mirroring
// Params.Save's own doc comment.
func (p *RecurrentParams) Save(w io.Writer, environmentID string, metadata checkpoint.Metadata) error {
	hidden := make([]hiddenLayerData, len(p.Trunk.Hidden))
	for i, layer := range p.Trunk.Hidden {
		hidden[i] = hiddenLayerData{W: layer.W.Data, B: layer.B.Data}
	}

	data := recurrentCheckpointData{
		SchemaVersion: recurrentCheckpointSchemaVersionCurrent,
		EnvironmentID: environmentID,
		Metadata:      metadata,
		InputSize:     p.InputSize(),
		HiddenSize:    p.HiddenSize(),
		OutputSize:    p.OutputSize(),
		Hidden:        hidden,
		Wpi:           p.Trunk.Wpi.Data,
		Bpi:           p.Trunk.Bpi.Data,
		Wv:            p.Trunk.Wv.Data,
		Bv:            p.Trunk.Bv.Data,
		LSTM: lstmLayerData{
			WxF: p.LSTM.WxF.Data, WhF: p.LSTM.WhF.Data, BF: p.LSTM.BF.Data,
			WxI: p.LSTM.WxI.Data, WhI: p.LSTM.WhI.Data, BI: p.LSTM.BI.Data,
			WxO: p.LSTM.WxO.Data, WhO: p.LSTM.WhO.Data, BO: p.LSTM.BO.Data,
			WxG: p.LSTM.WxG.Data, WhG: p.LSTM.WhG.Data, BG: p.LSTM.BG.Data,
		},
	}

	if err := json.NewEncoder(w).Encode(data); err != nil {
		return fmt.Errorf("actorcritic: saving recurrent checkpoint: %w", err)
	}
	return nil
}

// RecurrentLoad reads a RecurrentParams previously written by Save from
// r, reconstructing every matrix at the shape implied by the saved layer
// sizes and rejecting the checkpoint if any saved matrix's data doesn't
// match that shape, or if its EnvironmentID doesn't match
// expectedEnvironmentID. The returned checkpoint.Metadata is the
// checkpoint's saved run-progress information.
func RecurrentLoad(r io.Reader, expectedEnvironmentID string) (*RecurrentParams, checkpoint.Metadata, error) {
	var data recurrentCheckpointData
	if err := json.NewDecoder(r).Decode(&data); err != nil {
		return nil, checkpoint.Metadata{}, fmt.Errorf("actorcritic: loading recurrent checkpoint: %w", err)
	}
	if data.SchemaVersion != recurrentCheckpointSchemaVersionCurrent {
		return nil, checkpoint.Metadata{}, fmt.Errorf(
			"actorcritic: loading recurrent checkpoint: unsupported schema version %d",
			data.SchemaVersion,
		)
	}
	if data.EnvironmentID != expectedEnvironmentID {
		return nil, checkpoint.Metadata{}, fmt.Errorf(
			"actorcritic: loading recurrent checkpoint: saved for environment %q, want %q",
			data.EnvironmentID, expectedEnvironmentID,
		)
	}

	trunkParams, err := paramsFromCheckpointData(checkpointData{
		InputSize:  data.InputSize,
		HiddenSize: data.HiddenSize,
		OutputSize: data.OutputSize,
		Hidden:     data.Hidden,
		Wpi:        data.Wpi,
		Bpi:        data.Bpi,
		Wv:         data.Wv,
		Bv:         data.Bv,
	})
	if err != nil {
		return nil, checkpoint.Metadata{}, err
	}

	lstm, err := lstmLayerFromCheckpointData(data.LSTM, data.HiddenSize)
	if err != nil {
		return nil, checkpoint.Metadata{}, err
	}

	return &RecurrentParams{Trunk: trunkParams, LSTM: lstm}, data.Metadata, nil
}

// lstmLayerFromCheckpointData reconstructs an LSTMLayer from data, every
// gate's weights sized hiddenSize x hiddenSize (Wh*) or hiddenSize x
// hiddenSize (Wx*, since the LSTM's input is always the trunk's
// HiddenSize-wide output — see RecurrentParams' own doc comment),
// rejecting the checkpoint if any saved matrix's data doesn't match that
// shape.
func lstmLayerFromCheckpointData(data lstmLayerData, hiddenSize int) (*LSTMLayer, error) {
	l := &LSTMLayer{
		WxF: mat.New(hiddenSize, hiddenSize), WhF: mat.New(hiddenSize, hiddenSize), BF: mat.New(hiddenSize, 1),
		WxI: mat.New(hiddenSize, hiddenSize), WhI: mat.New(hiddenSize, hiddenSize), BI: mat.New(hiddenSize, 1),
		WxO: mat.New(hiddenSize, hiddenSize), WhO: mat.New(hiddenSize, hiddenSize), BO: mat.New(hiddenSize, 1),
		WxG: mat.New(hiddenSize, hiddenSize), WhG: mat.New(hiddenSize, hiddenSize), BG: mat.New(hiddenSize, 1),
	}

	for _, field := range []checkpointField{
		{name: "lstm.wxf", dst: l.WxF, src: data.WxF}, {name: "lstm.whf", dst: l.WhF, src: data.WhF}, {name: "lstm.bf", dst: l.BF, src: data.BF},
		{name: "lstm.wxi", dst: l.WxI, src: data.WxI}, {name: "lstm.whi", dst: l.WhI, src: data.WhI}, {name: "lstm.bi", dst: l.BI, src: data.BI},
		{name: "lstm.wxo", dst: l.WxO, src: data.WxO}, {name: "lstm.who", dst: l.WhO, src: data.WhO}, {name: "lstm.bo", dst: l.BO, src: data.BO},
		{name: "lstm.wxg", dst: l.WxG, src: data.WxG}, {name: "lstm.whg", dst: l.WhG, src: data.WhG}, {name: "lstm.bg", dst: l.BG, src: data.BG},
	} {
		if err := copyField(field); err != nil {
			return nil, err
		}
	}
	return l, nil
}

// SaveFile is a convenience wrapper around Save that writes to the file
// at path, creating or truncating it as needed.
func (p *RecurrentParams) SaveFile(path string, environmentID string, metadata checkpoint.Metadata) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("actorcritic: saving recurrent checkpoint: %w", err)
	}

	if err := p.Save(file, environmentID, metadata); err != nil {
		_ = file.Close()
		return err
	}

	if err := file.Close(); err != nil {
		return fmt.Errorf("actorcritic: saving recurrent checkpoint: %w", err)
	}
	return nil
}

// RecurrentLoadFile is a convenience wrapper around RecurrentLoad that
// reads from the file at path.
func RecurrentLoadFile(path string, expectedEnvironmentID string) (*RecurrentParams, checkpoint.Metadata, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, checkpoint.Metadata{}, fmt.Errorf("actorcritic: loading recurrent checkpoint: %w", err)
	}
	defer file.Close()

	return RecurrentLoad(file, expectedEnvironmentID)
}
