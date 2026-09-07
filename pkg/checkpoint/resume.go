package checkpoint

import (
	"fmt"
	"math"
	"os"
)

// Resumed is what Resume found (or didn't) in a checkpoint directory:
// the loaded params (the zero value of T, e.g. nil for a pointer type,
// if starting fresh) and the run-progress counters to continue from.
type Resumed[T any] struct {
	Params       T
	StartEpoch   int
	BestReturn   float32
	TotalUpdates int
}

// Resume looks for the latest Path-named checkpoint matching prefix in
// dir and loads it via load if found, returning a Resumed with the zero
// value of T (meaning "start fresh") if dir is empty or contains no
// matching checkpoint yet — both are normal, expected states (e.g. the
// very first run against a new checkpoint directory), not errors. It
// only returns an error if dir contains a checkpoint that load rejects
// (corrupt file, or an architecture/environment mismatch).
//
// load is one of pkg/policy's, pkg/actorcritic's, or pkg/hierarchical's
// own LoadFile functions (or a small closure adapting one, for a format
// like pkg/hierarchical's whose LoadFile takes extra arguments and
// returns more than one params value) that takes the checkpoint path
// and returns the loaded params and its saved Metadata.
func Resume[T any](dir, prefix string, load func(path string) (T, Metadata, error)) (Resumed[T], error) {
	fresh := Resumed[T]{BestReturn: float32(math.Inf(-1))}
	if dir == "" {
		return fresh, nil
	}

	latestPath, err := Latest(dir, prefix)
	if err != nil {
		return fresh, nil
	}

	params, metadata, err := load(latestPath)
	if err != nil {
		return Resumed[T]{}, fmt.Errorf("resuming from %s: %w", latestPath, err)
	}

	return Resumed[T]{
		Params:       params,
		StartEpoch:   metadata.Epoch + 1,
		BestReturn:   metadata.BestReturn,
		TotalUpdates: metadata.TotalUpdates,
	}, nil
}

// Save creates dir if it doesn't exist yet, then calls save with the
// Path-conventional file name for epoch under dir, tagged with prefix.
//
// save is one of pkg/policy's, pkg/actorcritic's, or
// pkg/hierarchical's own SaveFile functions/methods (or a small closure
// adapting one) that writes the checkpoint to the given path.
func Save(dir, prefix string, epoch int, save func(path string) error) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("creating checkpoint directory %s: %w", dir, err)
	}
	return save(Path(dir, prefix, epoch))
}
