// Package checkpoint provides the naming, discovery, and run-progress
// metadata helpers shared by pkg/policy's and pkg/actorcritic's
// checkpoint formats: a conventional file name for a checkpoint saved
// at a given epoch, finding the most recent one in a directory, and the
// run-progress fields (independent of either format's own
// schema-version/architecture fields) every checkpoint embeds.
//
// This package intentionally knows nothing about either checkpoint
// format's actual weight data; it only deals with file names and the
// small piece of metadata both formats embed identically.
package checkpoint

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// Metadata is run-progress information every checkpoint format embeds
// alongside its own schema/architecture fields (see pkg/policy's and
// pkg/actorcritic's own checkpointData types): how far training had
// gotten and the best result seen so far, so resuming from a checkpoint
// can continue those counters instead of restarting them from zero.
type Metadata struct {
	// Epoch is the last training epoch completed before this checkpoint
	// was saved.
	Epoch int `json:"epoch"`
	// BestReturn is the best EpochStats.AverageReturn observed by the
	// run that produced this checkpoint, up to and including Epoch.
	BestReturn float32 `json:"best_return"`
	// TotalUpdates is the total number of gradient-update steps applied
	// by the run that produced this checkpoint, up to and including
	// Epoch.
	TotalUpdates int `json:"total_updates"`
}

// checkpointFileSuffix is the file extension every checkpoint produced
// by Path (and therefore recognized by Latest) uses.
const checkpointFileSuffix = ".json"

// Path returns the conventional file name for a checkpoint saved at the
// given epoch, under dir, tagged with prefix (e.g. "policy" or "ppo",
// distinguishing which trainer/format produced it): dir/prefix-epoch-
// <N>.json, zero-padded so lexical and numeric ordering agree up to 9
// digits of epoch.
func Path(dir, prefix string, epoch int) string {
	return filepath.Join(dir, fmt.Sprintf("%s-epoch-%09d%s", prefix, epoch, checkpointFileSuffix))
}

// checkpointFilePrefix returns the file-name prefix (excluding the
// epoch number and suffix) Path uses for prefix, so Latest can recognize
// which files in a directory are checkpoints it produced.
func checkpointFilePrefix(prefix string) string {
	return prefix + "-epoch-"
}

// Latest returns the path to the highest-epoch checkpoint matching
// prefix in dir, determined by parsing each matching file name's epoch
// number, not by directory-listing order (which os.ReadDir does not
// guarantee to reflect epoch order) or by lexical sort alone (which
// only agrees with numeric order among file names Path itself
// produced). It returns an error if dir doesn't exist or contains no
// matching checkpoint, so callers can fall back to fresh initialization
// on either case (see pkg/policy.LoadFile/pkg/actorcritic.LoadFile's
// error-returning convention for a missing file).
func Latest(dir, prefix string) (string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", fmt.Errorf("checkpoint: listing %s: %w", dir, err)
	}

	filePrefix := checkpointFilePrefix(prefix)
	latestEpoch := -1
	latestPath := ""

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		name := entry.Name()
		if !strings.HasPrefix(name, filePrefix) || !strings.HasSuffix(name, checkpointFileSuffix) {
			continue
		}

		epochText := strings.TrimSuffix(strings.TrimPrefix(name, filePrefix), checkpointFileSuffix)
		epoch, err := strconv.Atoi(epochText)
		if err != nil {
			continue // not one of Path's checkpoint files after all
		}

		if epoch > latestEpoch {
			latestEpoch = epoch
			latestPath = filepath.Join(dir, name)
		}
	}

	if latestPath == "" {
		return "", fmt.Errorf("checkpoint: no %q checkpoints found in %s", prefix, dir)
	}
	return latestPath, nil
}
