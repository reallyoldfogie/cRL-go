package mat

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func matFromRows(rows [][]float32) *Matrix {
	numRows := len(rows)
	numCols := 0
	if numRows > 0 {
		numCols = len(rows[0])
	}

	m := New(numRows, numCols)
	for rowIndex, row := range rows {
		copy(m.Data[rowIndex*numCols:(rowIndex+1)*numCols], row)
	}
	return m
}

func TestMatMulVariants(t *testing.T) {
	// a = [[1, 2, 3], [4, 5, 6]]           (2x3)
	// b = [[7, 8], [9, 10], [11, 12]]      (3x2)
	// a * b = [[58, 64], [139, 154]]       (2x2)
	a := matFromRows([][]float32{{1, 2, 3}, {4, 5, 6}})
	b := matFromRows([][]float32{{7, 8}, {9, 10}, {11, 12}})

	t.Run("no transpose", func(t *testing.T) {
		out := New(2, 2)
		require.NoError(t, out.MatMul(a, b, true, false, false))
		assert.Equal(t, []float32{58, 64, 139, 154}, out.Data)
	})

	t.Run("transpose a", func(t *testing.T) {
		// aT is (3x2); (aT)^T * b == a * b, so passing aT with transposeA
		// should reproduce the same result as the non-transposed case.
		aT := matFromRows([][]float32{{1, 4}, {2, 5}, {3, 6}})
		out := New(2, 2)
		require.NoError(t, out.MatMul(aT, b, true, true, false))
		assert.Equal(t, []float32{58, 64, 139, 154}, out.Data)
	})

	t.Run("transpose b", func(t *testing.T) {
		bT := matFromRows([][]float32{{7, 9, 11}, {8, 10, 12}})
		out := New(2, 2)
		require.NoError(t, out.MatMul(a, bT, true, false, true))
		assert.Equal(t, []float32{58, 64, 139, 154}, out.Data)
	})

	t.Run("transpose both", func(t *testing.T) {
		aT := matFromRows([][]float32{{1, 4}, {2, 5}, {3, 6}})
		bT := matFromRows([][]float32{{7, 9, 11}, {8, 10, 12}})
		out := New(2, 2)
		require.NoError(t, out.MatMul(aT, bT, true, true, true))
		assert.Equal(t, []float32{58, 64, 139, 154}, out.Data)
	})

	t.Run("accumulate instead of zeroing", func(t *testing.T) {
		out := matFromRows([][]float32{{1, 1}, {1, 1}})
		require.NoError(t, out.MatMul(a, b, false, false, false))
		assert.Equal(t, []float32{59, 65, 140, 155}, out.Data)
	})

	t.Run("shape mismatch is reported", func(t *testing.T) {
		out := New(2, 2)
		wrongShapeB := New(2, 2)
		err := out.MatMul(a, wrongShapeB, true, false, false)
		assert.Error(t, err)
	})
}

func TestSoftmaxSumsToOneAndStaysFinite(t *testing.T) {
	inputs := [][]float32{
		{1, 2, 3, 4, 5},
		{0, 0, 0},
		{-1000, -1000, -1000},
		{1000, 1, 1}, // would overflow expf without the max-subtraction trick
		{100000, -100000},
	}

	for _, in := range inputs {
		inMat := &Matrix{Rows: 1, Cols: len(in), Data: append([]float32(nil), in...)}
		out := New(1, len(in))
		require.NoError(t, out.Softmax(inMat))

		var sum float32
		for _, v := range out.Data {
			require.False(t, math.IsNaN(float64(v)), "softmax output must not be NaN, got %v for input %v", out.Data, in)
			require.False(t, math.IsInf(float64(v), 0), "softmax output must not be Inf, got %v for input %v", out.Data, in)
			assert.GreaterOrEqual(t, v, float32(0))
			sum += v
		}
		assert.InDelta(t, 1.0, sum, 1e-4)
	}
}

func TestReinforceLossClampsNearZeroProbabilities(t *testing.T) {
	probs := &Matrix{Rows: 1, Cols: 2, Data: []float32{0, 1}}
	advantages := &Matrix{Rows: 1, Cols: 2, Data: []float32{1, 1}}
	out := New(1, 2)

	require.NoError(t, out.ReinforceLoss(probs, advantages))

	// probs[0] == 0 would produce -log(0) == +Inf without the epsilon
	// clamp; instead it should be a large but finite number.
	assert.False(t, math.IsInf(float64(out.Data[0]), 0))
	assert.False(t, math.IsNaN(float64(out.Data[0])))
	// probs[1] == 1 -> -log(1) * 1 == 0.
	assert.InDelta(t, 0.0, out.Data[1], 1e-6)
}

func TestReinforceAddGradAccumulates(t *testing.T) {
	probs := &Matrix{Rows: 1, Cols: 1, Data: []float32{0.5}}
	rt := &Matrix{Rows: 1, Cols: 1, Data: []float32{2}}
	grad := &Matrix{Rows: 1, Cols: 1, Data: []float32{1}}
	dst := &Matrix{Rows: 1, Cols: 1, Data: []float32{10}} // pre-existing gradient to accumulate onto

	require.NoError(t, dst.ReinforceAddGrad(probs, rt, grad))

	// 10 + 1 * (-2 / 0.5) == 10 - 4 == 6
	assert.InDelta(t, 6.0, dst.Data[0], 1e-6)
}

func TestReLUAndReLUAddGrad(t *testing.T) {
	in := &Matrix{Rows: 1, Cols: 3, Data: []float32{-1, 0, 2}}
	out := New(1, 3)
	require.NoError(t, out.ReLU(in))
	assert.Equal(t, []float32{0, 0, 2}, out.Data)

	grad := &Matrix{Rows: 1, Cols: 3, Data: []float32{5, 5, 5}}
	dst := New(1, 3)
	require.NoError(t, dst.ReLUAddGrad(in, grad))
	// Gradient only flows through where the input was strictly positive.
	assert.Equal(t, []float32{0, 0, 5}, dst.Data)
}

func TestSoftmaxAddGrad(t *testing.T) {
	softmaxOut := &Matrix{Rows: 1, Cols: 2, Data: []float32{0.25, 0.75}}
	grad := &Matrix{Rows: 1, Cols: 2, Data: []float32{1, 0}}
	dst := New(1, 2)

	require.NoError(t, dst.SoftmaxAddGrad(softmaxOut, grad))

	// dot = 1*0.25 + 0*0.75 = 0.25
	// dst[0] = 0.25 * (1 - 0.25) = 0.1875
	// dst[1] = 0.75 * (0 - 0.25) = -0.1875
	assert.InDelta(t, 0.1875, dst.Data[0], 1e-6)
	assert.InDelta(t, -0.1875, dst.Data[1], 1e-6)
}

func TestFillRandStaysWithinBounds(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	m := New(10, 10)
	m.FillRand(rng, -0.5, 0.5)

	for _, v := range m.Data {
		assert.GreaterOrEqual(t, v, float32(-0.5))
		assert.Less(t, v, float32(0.5))
	}
}
