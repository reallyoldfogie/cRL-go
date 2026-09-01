package gridrender

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewFillsEveryCell(t *testing.T) {
	grid := New(2, 3, '.')
	assert.Equal(t, []string{"...", "..."}, grid.Lines())
}

func TestSetPlacesMarkerAtGivenCoordinate(t *testing.T) {
	grid := New(3, 3, '.')
	grid.Set(1, 0, '@') // bottom row, middle column

	lines := grid.Lines()
	assert.Equal(t, []string{"...", "...", ".@."}, lines)
}

func TestSetIsNoOpOutOfBounds(t *testing.T) {
	grid := New(2, 2, '.')
	grid.Set(-1, 0, '@')
	grid.Set(0, -1, '@')
	grid.Set(2, 0, '@')
	grid.Set(0, 2, '@')

	assert.Equal(t, []string{"..", ".."}, grid.Lines())
}

func TestLinesOrdersHighestYFirst(t *testing.T) {
	grid := New(3, 1, '.')
	grid.Set(0, 0, 'A')
	grid.Set(0, 1, 'B')
	grid.Set(0, 2, 'C')

	// Y=2 (highest) should render first, Y=0 (lowest) last, so an
	// ActionUp (Y++) visibly moves a marker up the page.
	assert.Equal(t, []string{"C", "B", "A"}, grid.Lines())
}
