// Command replay writes a static, dependency-free HTML+JS viewer for a
// pkg/decisionlog file (see cmd/watch's -decision-log-out). Open the
// written file directly in any browser — no server needed, it reads a
// log you pick via its file input entirely client-side — and step
// through the run frame by frame: play/pause, scrub, or step one at a
// time, seeing the rendered grid, the full action-probability
// distribution, and any algorithm-specific Extra alongside each step.
// See docs/plans/15-agent-and-training-visualization.md, option (b).2.
package main

import (
	_ "embed"
	"flag"
	"fmt"
	"log"
	"os"
)

//go:embed replay.html
var replayHTML []byte

func main() {
	if err := run(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}

func run(args []string) error {
	fs := flag.NewFlagSet("replay", flag.ContinueOnError)
	out := fs.String("out", "replay.html", "path to write the static replay viewer")
	if err := fs.Parse(args); err != nil {
		return err
	}

	if err := os.WriteFile(*out, replayHTML, 0o644); err != nil {
		return fmt.Errorf("writing %s: %w", *out, err)
	}
	fmt.Printf("Wrote %s. Open it in a browser and load a -decision-log-out file to replay it.\n", *out)
	return nil
}
