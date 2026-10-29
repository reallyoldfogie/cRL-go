package actorcritic

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"regexp"
	"strconv"

	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"
	"github.com/reallyoldfogie/cRL-go/pkg/mat"
)

// ExpandParams returns a new Params that computes the same function p
// does on any input whose first p.InputSize() features match (any
// beyond that are new and start contributing nothing), reshaped to
// newInputSize inputs and newOutputSize outputs: p's own learned
// weights are kept exactly as they are, at their original indices, and
// every new input column / new output row is zero-filled. p itself is
// unmodified.
//
// Unlike Net2WiderNet (which widens the HIDDEN hiding layers, every
// existing unit cloned so none of newHiddenSize-p.HiddenSize()'s added
// capacity starts at zero), this widens the INPUT and OUTPUT
// boundaries, where zero really is the only choice that preserves
// function exactly: a zero input column means a new observation
// feature is multiplied by zero and so contributes nothing to any
// hidden unit's pre-activation, regardless of what value that feature
// actually holds this step. A zero output row means a new action's
// policy logit starts at 0 — not a problem in practice, since the
// caller only exists because something upstream already decided the
// new action should start masked out of sampling until its own task
// gets explicitly turned on, at which point training moves this row's
// weights away from zero like any other.
//
// Exists for exactly the situation Net2WiderNet/Net2DeeperNet's own doc
// comments describe for their own axis: a checkpoint's observation/
// action-space shape is otherwise fixed for its whole life (Load/
// RecurrentLoad reconstruct whatever shape the file itself stored,
// rejecting anything else), but mc-agent's own rlenv package has grown
// its observation vector and action vocabulary by simply appending new
// features/actions at the end on three separate occasions already (see
// mc-agent's rlenv/observation.go and rlenv/action.go, and
// mc-rsi-trainer's own EnvironmentID convention,
// "mc-agent-rlenv:actions=N:obs=M[:norm=1]") - each one otherwise forces
// every in-flight checkpoint to restart cold the moment the dependency
// bump lands, discarding however much training it already has. See
// LoadExpanding/RecurrentLoadExpanding for where this gets applied
// automatically on checkpoint resume, rather than requiring an operator
// to remember to run a conversion step by hand first.
//
// newInputSize/newOutputSize must each be at least p's own current
// InputSize()/OutputSize() - shrinking which weights a checkpoint keeps
// is not a thing this function will guess how to do safely, so it's
// rejected outright rather than attempted.
func ExpandParams(p *Params, newInputSize, newOutputSize int) (*Params, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	oldInputSize, oldOutputSize := p.InputSize(), p.OutputSize()
	if newInputSize < oldInputSize {
		return nil, fmt.Errorf("actorcritic: ExpandParams: newInputSize %d is smaller than the current input size %d - shrinking is not supported", newInputSize, oldInputSize)
	}
	if newOutputSize < oldOutputSize {
		return nil, fmt.Errorf("actorcritic: ExpandParams: newOutputSize %d is smaller than the current output size %d - shrinking is not supported", newOutputSize, oldOutputSize)
	}

	hidden := make([]Layer, len(p.Hidden))
	for i, layer := range p.Hidden {
		w := cloneMatrix(layer.W)
		if i == 0 && newInputSize > oldInputSize {
			w = expandCols(w, newInputSize)
		}
		hidden[i] = Layer{W: w, B: cloneMatrix(layer.B)}
	}

	wpi, bpi := cloneMatrix(p.Wpi), cloneMatrix(p.Bpi)
	if newOutputSize > oldOutputSize {
		wpi, bpi = expandRows(wpi, newOutputSize), expandRows(bpi, newOutputSize)
	}

	return &Params{
		Hidden: hidden,
		Wpi:    wpi, Bpi: bpi,
		Wv: cloneMatrix(p.Wv), Bv: cloneMatrix(p.Bv),
	}, nil
}

// ExpandRecurrentParams is ExpandParams for a RecurrentParams: both the
// policy and value trunks' input boundaries are expanded (they consume
// the same observation), but the value trunk's OWN OutputSize (its
// Wpi/Bpi - a full policy-shaped head NewParams always allocates even
// for a trunk only ever read through its separate Wv/Bv scalar value
// head, and so simply unused dead weight here) is left exactly as it
// was: there's nothing semantically meaningful to expand it to, and
// nothing reads it either way. Neither LSTM layer is touched at all: it
// sits strictly between the trunks' hidden output and their heads (see
// RecurrentParams' own doc comment's diagram), so its gate matrices are
// HiddenSize x HiddenSize throughout, never sized against the
// observation or action space in the first place.
func ExpandRecurrentParams(rp *RecurrentParams, newInputSize, newOutputSize int) (*RecurrentParams, error) {
	rp.mu.RLock()
	defer rp.mu.RUnlock()

	policy, err := ExpandParams(rp.policyTrunk(), newInputSize, newOutputSize)
	if err != nil {
		return nil, fmt.Errorf("actorcritic: ExpandRecurrentParams: policy trunk: %w", err)
	}
	value, err := ExpandParams(rp.valueTrunk(), newInputSize, rp.valueTrunk().OutputSize())
	if err != nil {
		return nil, fmt.Errorf("actorcritic: ExpandRecurrentParams: value trunk: %w", err)
	}
	policyLSTM := rp.policyLSTM().clone()
	valueLSTM := rp.valueLSTM().clone()

	return &RecurrentParams{
		Trunk: policy, LSTM: policyLSTM,
		PolicyTrunk: policy, ValueTrunk: value,
		PolicyLSTM: policyLSTM, ValueLSTM: valueLSTM,
	}, nil
}

// expandCols returns a new matrix with m.Rows rows and newCols columns
// (newCols > m.Cols): the first m.Cols columns are m's own data,
// unchanged; every column beyond that is zero (mat.New's own
// zero-value allocation - never written to here).
func expandCols(m *mat.Matrix, newCols int) *mat.Matrix {
	out := mat.New(m.Rows, newCols)
	for r := 0; r < m.Rows; r++ {
		copy(out.Data[r*newCols:r*newCols+m.Cols], m.Data[r*m.Cols:(r+1)*m.Cols])
	}
	return out
}

// expandRows returns a new matrix with newRows rows (newRows > m.Rows)
// and m.Cols columns: the first m.Rows rows are m's own data,
// unchanged; every row beyond that is zero.
func expandRows(m *mat.Matrix, newRows int) *mat.Matrix {
	out := mat.New(newRows, m.Cols)
	copy(out.Data, m.Data)
	return out
}

// environmentIDPattern matches the "actions=N:obs=M" fragment every
// EnvironmentID this package has ever seen embeds somewhere in its own,
// otherwise caller-owned, string (see Save's own doc comment - "e.g.
// snake:36" - EnvironmentID is opaque to this package in general; this
// one convention is the sole exception, and only LoadExpanding/
// RecurrentLoadExpanding below ever look for it).
var environmentIDPattern = regexp.MustCompile(`actions=(\d+):obs=(\d+)`)

// parseActionObsID extracts the actions=/obs= counts embedded in id,
// and template: id with that fragment replaced by a fixed placeholder,
// so two IDs that differ only in those two numbers produce equal
// templates. ok is false if id doesn't contain the fragment at all.
func parseActionObsID(id string) (actions, obs int, template string, ok bool) {
	loc := environmentIDPattern.FindStringSubmatchIndex(id)
	if loc == nil {
		return 0, 0, "", false
	}
	actions, errA := strconv.Atoi(id[loc[2]:loc[3]])
	obs, errB := strconv.Atoi(id[loc[4]:loc[5]])
	if errA != nil || errB != nil {
		return 0, 0, "", false
	}
	template = id[:loc[0]] + "actions=#:obs=#" + id[loc[1]:]
	return actions, obs, template, true
}

// ExpandTarget reports whether a checkpoint saved under storedID can be
// loaded into the shape expectedID describes by expanding rather than
// retraining from scratch: both IDs must embed the actions=/obs=
// fragment, agree on everything else in the string, and expectedID's
// own actions/obs counts must each be at least storedID's own (strictly
// greater in at least one - an exact match never reaches here, since
// Load/RecurrentLoad already succeed on their own in that case).
// Anything else - a shrink, or a mismatch anywhere outside that one
// fragment - returns ok=false, so the caller falls back to Load's own
// unmodified rejection rather than silently reshaping a checkpoint that
// was never compatible with this one to begin with.
func ExpandTarget(storedID, expectedID string) (newActions, newObs int, ok bool) {
	storedActions, storedObs, storedTemplate, ok1 := parseActionObsID(storedID)
	wantActions, wantObs, wantTemplate, ok2 := parseActionObsID(expectedID)
	if !ok1 || !ok2 || storedTemplate != wantTemplate {
		return 0, 0, false
	}
	if wantActions < storedActions || wantObs < storedObs {
		return 0, 0, false
	}
	if wantActions == storedActions && wantObs == storedObs {
		return 0, 0, false
	}
	return wantActions, wantObs, true
}

// LoadExpanding is Load, but additionally tolerates a checkpoint saved
// for a smaller actions=/obs= shape than expectedEnvironmentID's own
// (everything else about the two IDs must match exactly - see
// ExpandTarget) by expanding it via ExpandParams instead of rejecting
// it outright. This is the ordinary situation right after mc-agent
// grows its action space or observation vector and a given checkpoint
// lineage hasn't been retrained since: the expanded Params this returns
// already matches expectedEnvironmentID's own shape, so the very next
// Save call (which every caller already derives from the CURRENT code's
// own ObservationSize()/ActionSpace(), not from whatever a loaded
// checkpoint happened to say) naturally writes expectedEnvironmentID -
// this only ever triggers once per checkpoint, not on every resume
// after that. A stored ID that's incompatible in any other way (a
// shrink, or any difference outside the actions=/obs= fragment) returns
// exactly the error Load itself would have returned.
func LoadExpanding(r io.Reader, expectedEnvironmentID string) (*Params, checkpoint.Metadata, error) {
	var data checkpointData
	raw, err := io.ReadAll(r)
	if err != nil {
		return nil, checkpoint.Metadata{}, fmt.Errorf("actorcritic: loading checkpoint: %w", err)
	}
	if err := json.Unmarshal(raw, &data); err != nil {
		return nil, checkpoint.Metadata{}, fmt.Errorf("actorcritic: loading checkpoint: %w", err)
	}

	if data.EnvironmentID == expectedEnvironmentID {
		return Load(bytes.NewReader(raw), expectedEnvironmentID)
	}

	newActions, newObs, ok := ExpandTarget(data.EnvironmentID, expectedEnvironmentID)
	if !ok {
		// Not expandable - let Load produce its own, already-correct
		// rejection rather than duplicating that error message here.
		return Load(bytes.NewReader(raw), expectedEnvironmentID)
	}

	params, metadata, err := Load(bytes.NewReader(raw), data.EnvironmentID)
	if err != nil {
		return nil, checkpoint.Metadata{}, err
	}
	expanded, err := ExpandParams(params, newObs, newActions)
	if err != nil {
		return nil, checkpoint.Metadata{}, fmt.Errorf("actorcritic: loading checkpoint: expanding from %q to %q: %w", data.EnvironmentID, expectedEnvironmentID, err)
	}
	return expanded, metadata, nil
}

// LoadFileExpanding is a convenience wrapper around LoadExpanding that
// reads from the file at path, mirroring LoadFile's own relationship to
// Load.
func LoadFileExpanding(path string, expectedEnvironmentID string) (*Params, checkpoint.Metadata, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, checkpoint.Metadata{}, fmt.Errorf("actorcritic: loading checkpoint: %w", err)
	}
	defer file.Close()
	return LoadExpanding(file, expectedEnvironmentID)
}

// RecurrentLoadExpanding is RecurrentLoad, but additionally tolerates a
// checkpoint saved for a smaller actions=/obs= shape than
// expectedEnvironmentID's own, the same way LoadExpanding does for a
// plain Params checkpoint - see its own doc comment for the full
// reasoning, which applies identically here.
func RecurrentLoadExpanding(r io.Reader, expectedEnvironmentID string) (*RecurrentParams, checkpoint.Metadata, error) {
	var data recurrentCheckpointData
	raw, err := io.ReadAll(r)
	if err != nil {
		return nil, checkpoint.Metadata{}, fmt.Errorf("actorcritic: loading recurrent checkpoint: %w", err)
	}
	if err := json.Unmarshal(raw, &data); err != nil {
		return nil, checkpoint.Metadata{}, fmt.Errorf("actorcritic: loading recurrent checkpoint: %w", err)
	}

	if data.EnvironmentID == expectedEnvironmentID {
		return RecurrentLoad(bytes.NewReader(raw), expectedEnvironmentID)
	}

	newActions, newObs, ok := ExpandTarget(data.EnvironmentID, expectedEnvironmentID)
	if !ok {
		return RecurrentLoad(bytes.NewReader(raw), expectedEnvironmentID)
	}

	params, metadata, err := RecurrentLoad(bytes.NewReader(raw), data.EnvironmentID)
	if err != nil {
		return nil, checkpoint.Metadata{}, err
	}
	expanded, err := ExpandRecurrentParams(params, newObs, newActions)
	if err != nil {
		return nil, checkpoint.Metadata{}, fmt.Errorf("actorcritic: loading recurrent checkpoint: expanding from %q to %q: %w", data.EnvironmentID, expectedEnvironmentID, err)
	}
	return expanded, metadata, nil
}

// RecurrentLoadFileExpanding is a convenience wrapper around
// RecurrentLoadExpanding that reads from the file at path, mirroring
// RecurrentLoadFile's own relationship to RecurrentLoad.
func RecurrentLoadFileExpanding(path string, expectedEnvironmentID string) (*RecurrentParams, checkpoint.Metadata, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, checkpoint.Metadata{}, fmt.Errorf("actorcritic: loading recurrent checkpoint: %w", err)
	}
	defer file.Close()
	return RecurrentLoadExpanding(file, expectedEnvironmentID)
}
