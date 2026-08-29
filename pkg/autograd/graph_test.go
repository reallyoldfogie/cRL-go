package autograd

import (
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/mat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestBuildGraphSingleRootUnchanged confirms BuildGraph's refactor to
// delegate to BuildGraphMulti didn't change its single-root behavior:
// the topological order still ends with the requested output.
func TestBuildGraphSingleRootUnchanged(t *testing.T) {
	a := &Var{Val: &mat.Matrix{Rows: 2, Cols: 1, Data: []float32{1, 2}}}
	b := &Var{Val: &mat.Matrix{Rows: 2, Cols: 1, Data: []float32{3, 4}}}

	out, err := Add(a, b)
	require.NoError(t, err)

	graph := BuildGraph(out)
	require.NotEmpty(t, graph.Vars)
	assert.Same(t, out, graph.Vars[len(graph.Vars)-1])

	graph.Forward()
	assert.Equal(t, []float32{4, 6}, out.Val.Data)
}

// TestBuildGraphMultiComputesBothRootsFromSharedTrunk models an
// actor-critic-style network: two independent heads (double and triple)
// fed by the same shared trunk Var. A single BuildGraphMulti + one
// Forward call must compute both heads correctly, and the shared trunk
// node must appear exactly once in the topological order (not be
// recomputed or duplicated per root).
func TestBuildGraphMultiComputesBothRootsFromSharedTrunk(t *testing.T) {
	trunk := &Var{Val: &mat.Matrix{Rows: 2, Cols: 1, Data: []float32{1, 2}}}

	doubleWeight := &Var{Val: &mat.Matrix{Rows: 2, Cols: 1, Data: []float32{2, 2}}}
	tripleWeight := &Var{Val: &mat.Matrix{Rows: 2, Cols: 1, Data: []float32{3, 3}}}

	headA, err := Mul(trunk, doubleWeight)
	require.NoError(t, err)
	headB, err := Mul(trunk, tripleWeight)
	require.NoError(t, err)

	graph := BuildGraphMulti(headA, headB)

	occurrences := 0
	for _, v := range graph.Vars {
		if v == trunk {
			occurrences++
		}
	}
	assert.Equal(t, 1, occurrences, "a Var shared by multiple roots must appear exactly once in the topological order")

	graph.Forward()
	assert.Equal(t, []float32{2, 4}, headA.Val.Data)
	assert.Equal(t, []float32{3, 6}, headB.Val.Data)
}
