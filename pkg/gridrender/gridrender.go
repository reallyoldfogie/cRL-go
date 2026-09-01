// Package gridrender builds a human-readable snapshot of a grid
// environment's current state, for "watch mode" style output (see
// cmd/watch) rather than for feeding a policy network — nothing in
// this module's training paths uses this package. It has no knowledge
// of any particular environment's own Position/Action types, so
// pkg/snakeenv, pkg/gridworldenv, and pkg/hierarchicalgridworld can
// each build their own Render() on top of it without depending on one
// another.
package gridrender

// Grid is a rows-by-cols character grid, using the same coordinate
// convention as this module's grid environments: X grows to the right,
// Y grows "up".
type Grid struct {
	rows, cols int
	cells      [][]rune
}

// New returns a rows-by-cols Grid with every cell initialized to fill.
func New(rows, cols int, fill rune) *Grid {
	cells := make([][]rune, rows)
	for y := range cells {
		row := make([]rune, cols)
		for x := range row {
			row[x] = fill
		}
		cells[y] = row
	}
	return &Grid{rows: rows, cols: cols, cells: cells}
}

// Set places r at (x, y). It is a no-op if (x, y) falls outside the
// grid, so a caller can place a possibly out-of-bounds entity (e.g. an
// agent that has just left the grid) without bounds-checking it itself
// first.
func (g *Grid) Set(x, y int, r rune) {
	if y < 0 || y >= g.rows || x < 0 || x >= g.cols {
		return
	}
	g.cells[y][x] = r
}

// Lines returns the grid as one string per row, with the highest-Y row
// first, so an environment whose "up" action increases Y (as every grid
// environment in this module does) renders with up visually up the
// page.
func (g *Grid) Lines() []string {
	lines := make([]string, g.rows)
	for i := range lines {
		lines[i] = string(g.cells[g.rows-1-i])
	}
	return lines
}
