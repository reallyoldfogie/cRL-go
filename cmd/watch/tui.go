package main

import (
	"fmt"
	"strings"

	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

// probabilityBarWidth is the character width of each action's
// probability bar in decisionPanel, chosen to fit comfortably beside a
// typical toy-environment grid in an 80-column terminal.
const probabilityBarWidth = 20

// rewardHistoryLen bounds how many past steps' rewards sparkline charts,
// so the sparkline stays a fixed width regardless of episode length.
const rewardHistoryLen = 30

// sparkChars renders a series of float32 values as a sequence of
// Unicode block characters, one per value, height-encoding each value
// relative to the series' own min/max — a compact, dependency-free
// trend indicator (no charting library, consistent with this module's
// stdlib-only approach elsewhere).
var sparkChars = []rune("▁▂▃▄▅▆▇█")

// sparkline renders values as a one-line Unicode block-character trend
// indicator. Returns "" for an empty series.
func sparkline(values []float32) string {
	if len(values) == 0 {
		return ""
	}

	minValue, maxValue := values[0], values[0]
	for _, v := range values {
		if v < minValue {
			minValue = v
		}
		if v > maxValue {
			maxValue = v
		}
	}
	span := maxValue - minValue

	runes := make([]rune, len(values))
	for i, v := range values {
		if span == 0 {
			runes[i] = sparkChars[0]
			continue
		}
		level := int((v - minValue) / span * float32(len(sparkChars)-1))
		runes[i] = sparkChars[level]
	}
	return string(runes)
}

// decisionPanel renders decision as a small side panel: a horizontal
// bar for every action's probability (marking the sampled one), the
// critic's value estimate where available, the reward just earned, any
// algorithm-specific extra context (e.g. hierarchical's active
// subgoal), and a reward-history sparkline — everything printDecision
// (the plain one-line predecessor to this richer "watch mode") used to
// print, but laid out for composeSideBySide next to the environment's
// own rendered grid instead of appended after it — see
// docs/plans/15-agent-and-training-visualization.md, option (b).3.
func decisionPanel(decision rl.Decision, extra string, reward float32, rewardHistory []float32) []string {
	lines := make([]string, 0, len(decision.Probabilities)+5)
	lines = append(lines, "Action probabilities:")

	for i, p := range decision.Probabilities {
		marker := "  "
		if rl.Action(i) == decision.Action {
			marker = "> "
		}
		filled := int(p*float32(probabilityBarWidth) + 0.5)
		filled = max(0, min(filled, probabilityBarWidth))
		bar := strings.Repeat("#", filled) + strings.Repeat(" ", probabilityBarWidth-filled)
		lines = append(lines, fmt.Sprintf("%s%2d: [%s] %5.1f%%", marker, i, bar, p*100))
	}

	lines = append(lines, "")
	if decision.HasValue {
		lines = append(lines, fmt.Sprintf("Value:  %.3f", decision.Value))
	}
	lines = append(lines, fmt.Sprintf("Reward: %.2f", reward))
	if extra != "" {
		lines = append(lines, extra)
	}
	if len(rewardHistory) > 0 {
		lines = append(lines, fmt.Sprintf("Reward history: %s", sparkline(rewardHistory)))
	}
	return lines
}

// maxLineWidth returns the length of the longest line in lines, in
// runes, for sizing composeSideBySide's left column.
func maxLineWidth(lines []string) int {
	width := 0
	for _, line := range lines {
		if n := len([]rune(line)); n > width {
			width = n
		}
	}
	return width
}

// composeSideBySide lays left and right out as two columns, left padded
// to leftWidth, one combined line per row of whichever side is taller
// (padding the shorter side with blank lines).
func composeSideBySide(left, right []string, leftWidth int) []string {
	height := max(len(left), len(right))
	combined := make([]string, height)
	for i := range height {
		var l, r string
		if i < len(left) {
			l = left[i]
		}
		if i < len(right) {
			r = right[i]
		}
		combined[i] = fmt.Sprintf("%-*s  %s", leftWidth, l, r)
	}
	return combined
}

// printFrame clears the terminal and prints step's number, environment
// grid, and decision panel side by side — see decisionPanel and
// composeSideBySide.
func printFrame(step int, gridLines []string, decision rl.Decision, extra string, reward float32, rewardHistory []float32) {
	panel := decisionPanel(decision, extra, reward, rewardHistory)
	combined := composeSideBySide(gridLines, panel, maxLineWidth(gridLines))

	// ANSI "clear screen, move cursor to top-left" — a plain,
	// dependency-free way to redraw in place.
	fmt.Print("\033[H\033[2J")
	fmt.Printf("Step %d\n", step)
	for _, line := range combined {
		fmt.Println(line)
	}
}
