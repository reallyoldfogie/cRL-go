// Package report turns a CSV written by any cmd/train*'s -metrics-out
// flag (pkg/metrics.CSVWriter) into a static, dependency-free HTML page
// charting every numeric column against epoch — a shareable,
// at-a-glance view of a training run instead of only scrolling raw
// printed text or a spreadsheet. See
// docs/plans/15-agent-and-training-visualization.md, option (a).3.
//
// Every cmd/train* binary writes a different set of metric columns
// (e.g. cmd/train's return_std vs. cmd/train-ppo's update_count vs.
// cmd/train-hierarchical's meta_update_count/sub_N_updates), so this
// package charts whatever columns are present rather than assuming any
// particular one exists, staying algorithm-agnostic the same way every
// other shared package in this module does.
package report

import (
	"encoding/csv"
	"fmt"
	"html"
	"io"
	"math"
	"strconv"
	"strings"
)

// Report is a parsed metrics CSV: one value per numeric column, per
// epoch.
type Report struct {
	Epochs []int
	// Columns preserves the CSV header's column order, excluding the
	// leading "epoch" column.
	Columns []string
	// Series maps a Columns entry to its per-epoch values, index-
	// aligned with Epochs.
	Series map[string][]float64
}

// ParseCSV parses a CSV written by pkg/metrics.CSVWriter: a header row
// whose first column is "epoch", followed by one or more numeric metric
// columns, then one data row per epoch.
func ParseCSV(r io.Reader) (Report, error) {
	rows, err := csv.NewReader(r).ReadAll()
	if err != nil {
		return Report{}, fmt.Errorf("report: reading CSV: %w", err)
	}
	if len(rows) == 0 {
		return Report{}, fmt.Errorf("report: CSV has no header row")
	}

	header := rows[0]
	if len(header) < 2 || header[0] != "epoch" {
		return Report{}, fmt.Errorf(`report: CSV header must start with "epoch" followed by at least one metric column, got %v`, header)
	}
	columns := header[1:]

	series := make(map[string][]float64, len(columns))
	for _, col := range columns {
		series[col] = make([]float64, 0, len(rows)-1)
	}

	epochs := make([]int, 0, len(rows)-1)
	for i, row := range rows[1:] {
		if len(row) != len(header) {
			return Report{}, fmt.Errorf("report: row %d has %d column(s), want %d", i+1, len(row), len(header))
		}

		epoch, err := strconv.Atoi(row[0])
		if err != nil {
			return Report{}, fmt.Errorf("report: row %d: parsing epoch %q: %w", i+1, row[0], err)
		}
		epochs = append(epochs, epoch)

		for j, col := range columns {
			value, err := strconv.ParseFloat(row[j+1], 64)
			if err != nil {
				return Report{}, fmt.Errorf("report: row %d: parsing column %q value %q: %w", i+1, col, row[j+1], err)
			}
			series[col] = append(series[col], value)
		}
	}

	return Report{Epochs: epochs, Columns: columns, Series: series}, nil
}

const (
	chartWidth   = 760
	chartHeight  = 220
	chartPadding = 40
)

// WriteHTML writes rpt as a single self-contained HTML page (inline
// CSS, no external stylesheet/script, no CDN dependency, so it opens
// correctly straight from disk with no network access) titled title,
// with one hand-drawn SVG line chart per column.
func WriteHTML(w io.Writer, rpt Report, title string) error {
	var b strings.Builder
	b.WriteString("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\n")
	fmt.Fprintf(&b, "<title>%s</title>\n", html.EscapeString(title))
	fmt.Fprintf(&b, `<style>
body{font-family:system-ui,sans-serif;background:#fafafa;color:#222;margin:2rem}
h1{font-size:1.4rem}
.chart{background:#fff;border:1px solid #ddd;border-radius:6px;padding:1rem;margin-bottom:1.5rem;max-width:%dpx}
.chart h2{font-size:1rem;margin:0 0 .5rem;font-family:monospace}
polyline{fill:none;stroke:#2563eb;stroke-width:2}
.axis{fill:#666;font-size:11px;font-family:monospace}
</style>
`, chartWidth)
	fmt.Fprintf(&b, "</head><body>\n<h1>%s</h1>\n", html.EscapeString(title))

	if len(rpt.Epochs) == 0 || len(rpt.Columns) == 0 {
		b.WriteString("<p>No data.</p>\n")
	}
	for _, col := range rpt.Columns {
		b.WriteString("<div class=\"chart\">\n")
		fmt.Fprintf(&b, "<h2>%s</h2>\n", html.EscapeString(col))
		b.WriteString(lineChartSVG(rpt.Epochs, rpt.Series[col]))
		b.WriteString("</div>\n")
	}
	b.WriteString("</body></html>\n")

	if _, err := io.WriteString(w, b.String()); err != nil {
		return fmt.Errorf("report: writing HTML: %w", err)
	}
	return nil
}

// lineChartSVG renders one column's values against epochs as a single
// SVG <polyline>, with min/max value labels and first/last epoch
// labels — no charting library, just plain coordinate math, consistent
// with this module's stdlib-only, minimal-dependency approach
// elsewhere (see e.g. pkg/autograd/pkg/mat's own doc comments).
func lineChartSVG(epochs []int, values []float64) string {
	if len(values) == 0 {
		return "<p>No data.</p>\n"
	}

	minValue, maxValue := values[0], values[0]
	for _, v := range values {
		minValue = math.Min(minValue, v)
		maxValue = math.Max(maxValue, v)
	}
	if minValue == maxValue {
		// A flat series would otherwise divide by zero below; widen
		// the range symmetrically so it renders as a flat line at
		// mid-height instead of failing.
		minValue--
		maxValue++
	}

	const innerWidth = chartWidth - 2*chartPadding
	const innerHeight = chartHeight - 2*chartPadding

	var points strings.Builder
	for i, v := range values {
		x := float64(chartPadding)
		if len(values) > 1 {
			x += innerWidth * float64(i) / float64(len(values)-1)
		}
		y := float64(chartPadding) + innerHeight*(1-(v-minValue)/(maxValue-minValue))
		fmt.Fprintf(&points, "%.1f,%.1f ", x, y)
	}

	var b strings.Builder
	fmt.Fprintf(&b, `<svg viewBox="0 0 %d %d" width="100%%" height="%d">`, chartWidth, chartHeight, chartHeight)
	fmt.Fprintf(&b, `<text class="axis" x="4" y="%d">%s</text>`, chartPadding, formatValue(maxValue))
	fmt.Fprintf(&b, `<text class="axis" x="4" y="%d">%s</text>`, chartHeight-chartPadding+14, formatValue(minValue))
	fmt.Fprintf(&b, `<text class="axis" x="%d" y="%d">epoch %d</text>`, chartPadding, chartHeight-8, epochs[0])
	fmt.Fprintf(&b, `<text class="axis" x="%d" y="%d" text-anchor="end">epoch %d</text>`, chartWidth-4, chartHeight-8, epochs[len(epochs)-1])
	fmt.Fprintf(&b, `<polyline points="%s"/>`, strings.TrimSpace(points.String()))
	b.WriteString("</svg>\n")
	return b.String()
}

func formatValue(v float64) string {
	return strconv.FormatFloat(v, 'g', 4, 64)
}
