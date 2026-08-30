// Command checkpoint-tool inspects checkpoint files saved by cmd/train
// (pkg/policy) or cmd/train-ppo (pkg/actorcritic), without needing to
// know which one produced a given file: checkpointData in both packages
// is unexported, so this tool decodes only the fields common to both
// formats' on-disk JSON (schema version, environment ID, layer sizes,
// and run-progress metadata — see pkg/checkpoint.Metadata), ignoring
// the format-specific weight arrays entirely.
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"
)

// checkpointInfo is the subset of either checkpoint format's on-disk
// JSON this tool reads. A checkpoint saved before schema_version/
// environment_id/metadata existed simply decodes those fields to their
// zero values.
type checkpointInfo struct {
	SchemaVersion int                 `json:"schema_version"`
	EnvironmentID string              `json:"environment_id"`
	InputSize     int                 `json:"input_size"`
	HiddenSize    int                 `json:"hidden_size"`
	OutputSize    int                 `json:"output_size"`
	Metadata      checkpoint.Metadata `json:"metadata"`
}

func main() {
	if err := run(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: checkpoint-tool <list|info|compare> ...")
	}

	switch args[0] {
	case "list":
		return runList(args[1:])
	case "info":
		return runInfo(args[1:])
	case "compare":
		return runCompare(args[1:])
	default:
		return fmt.Errorf("unknown subcommand %q, want \"list\", \"info\", or \"compare\"", args[0])
	}
}

// runList prints one summary line per checkpoint file (any file whose
// name ends in .json) found directly under dir, sorted by file name
// (which, for checkpoint.Path-named files, is also epoch order — see
// that function's zero-padding).
func runList(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: checkpoint-tool list <dir>")
	}
	dir := args[0]

	entries, err := os.ReadDir(dir)
	if err != nil {
		return fmt.Errorf("listing %s: %w", dir, err)
	}

	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}

		path := filepath.Join(dir, entry.Name())
		info, err := readCheckpointInfo(path)
		if err != nil {
			fmt.Printf("%s\t(unreadable: %v)\n", entry.Name(), err)
			continue
		}

		fmt.Printf(
			"%s\tenv=%s\tepoch=%d\tbest_return=%.3f\ttotal_updates=%d\n",
			entry.Name(), info.EnvironmentID, info.Metadata.Epoch, info.Metadata.BestReturn, info.Metadata.TotalUpdates,
		)
	}
	return nil
}

// runInfo prints every field runList summarizes, plus the layer sizes
// and schema version, for a single checkpoint file.
func runInfo(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: checkpoint-tool info <checkpoint>")
	}

	info, err := readCheckpointInfo(args[0])
	if err != nil {
		return err
	}
	printCheckpointInfo(info)
	return nil
}

func printCheckpointInfo(info checkpointInfo) {
	fmt.Printf("environment_id:  %s\n", info.EnvironmentID)
	fmt.Printf("schema_version:  %d\n", info.SchemaVersion)
	fmt.Printf("input_size:      %d\n", info.InputSize)
	fmt.Printf("hidden_size:     %d\n", info.HiddenSize)
	fmt.Printf("output_size:     %d\n", info.OutputSize)
	fmt.Printf("epoch:           %d\n", info.Metadata.Epoch)
	fmt.Printf("best_return:     %.3f\n", info.Metadata.BestReturn)
	fmt.Printf("total_updates:   %d\n", info.Metadata.TotalUpdates)
}

// runCompare prints a's and b's fields side by side.
func runCompare(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: checkpoint-tool compare <a> <b>")
	}

	infoA, err := readCheckpointInfo(args[0])
	if err != nil {
		return err
	}
	infoB, err := readCheckpointInfo(args[1])
	if err != nil {
		return err
	}

	fmt.Printf("%-16s %-24s %-24s\n", "field", args[0], args[1])
	fmt.Printf("%-16s %-24s %-24s\n", "environment_id", infoA.EnvironmentID, infoB.EnvironmentID)
	fmt.Printf("%-16s %-24d %-24d\n", "schema_version", infoA.SchemaVersion, infoB.SchemaVersion)
	fmt.Printf("%-16s %-24d %-24d\n", "epoch", infoA.Metadata.Epoch, infoB.Metadata.Epoch)
	fmt.Printf("%-16s %-24.3f %-24.3f\n", "best_return", infoA.Metadata.BestReturn, infoB.Metadata.BestReturn)
	fmt.Printf("%-16s %-24d %-24d\n", "total_updates", infoA.Metadata.TotalUpdates, infoB.Metadata.TotalUpdates)
	return nil
}

func readCheckpointInfo(path string) (checkpointInfo, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return checkpointInfo{}, fmt.Errorf("reading %s: %w", path, err)
	}

	var info checkpointInfo
	if err := json.Unmarshal(data, &info); err != nil {
		return checkpointInfo{}, fmt.Errorf("parsing %s: %w", path, err)
	}
	return info, nil
}
