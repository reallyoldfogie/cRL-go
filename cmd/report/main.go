// Command report reads a CSV written by any cmd/train*'s -metrics-out
// flag and writes a static, dependency-free HTML page charting every
// numeric column against epoch, for a shareable, at-a-glance view of a
// training run instead of only reading raw printed numbers — see
// docs/plans/15-agent-and-training-visualization.md, option (a).3.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/reallyoldfogie/cRL-go/pkg/report"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}

func run(args []string) error {
	fs := flag.NewFlagSet("report", flag.ContinueOnError)
	metricsIn := fs.String("metrics-in", "", "path to a CSV written by cmd/train*'s -metrics-out flag (required)")
	out := fs.String("out", "report.html", "path to write the generated HTML report")
	title := fs.String("title", "Training metrics", "title shown at the top of the generated report")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *metricsIn == "" {
		return fmt.Errorf("-metrics-in is required (see -h)")
	}

	in, err := os.Open(*metricsIn)
	if err != nil {
		return fmt.Errorf("opening %s: %w", *metricsIn, err)
	}
	defer in.Close()

	rpt, err := report.ParseCSV(in)
	if err != nil {
		return err
	}

	outFile, err := os.Create(*out)
	if err != nil {
		return fmt.Errorf("creating %s: %w", *out, err)
	}
	defer outFile.Close()

	if err := report.WriteHTML(outFile, rpt, *title); err != nil {
		return err
	}

	fmt.Printf("Wrote %s (%d epoch(s), %d metric column(s)).\n", *out, len(rpt.Epochs), len(rpt.Columns))
	return nil
}
