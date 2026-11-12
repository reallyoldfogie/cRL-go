package onnxinfer

import (
	"fmt"
	"math"

	onnxpb "github.com/reallyoldfogie/cRL-go/pkg/onnx"
)

func expf(v float32) float32 { return float32(math.Exp(float64(v))) }

// findAttr returns node's attribute named name, or nil if absent.
func findAttr(node *onnxpb.NodeProto, name string) *onnxpb.AttributeProto {
	for _, a := range node.GetAttribute() {
		if a.GetName() == name {
			return a
		}
	}
	return nil
}

func intAttrOr(node *onnxpb.NodeProto, name string, def int64) int64 {
	a := findAttr(node, name)
	if a == nil {
		return def
	}
	return a.GetI()
}

func floatAttrOr(node *onnxpb.NodeProto, name string, def float32) float32 {
	a := findAttr(node, name)
	if a == nil {
		return def
	}
	return a.GetF()
}

// evalGemm implements the subset of the ONNX Gemm operator this project
// needs: 2D A and B, y = alpha*(A or A^T)@(B or B^T) + beta*C, with C
// broadcasting over rows when it has fewer rows than the result (a [N]
// or [1,N] bias row, exactly pkg/onnxexport's shape). Attributes default
// per the ONNX spec (alpha=1, beta=1, transA=0, transB=0); this project's
// own exporter always sets transB=1 and leaves the rest at their
// defaults, but every attribute is still honored rather than assumed, so
// a hand-edited or future-exporter-produced model isn't silently
// misread.
func evalGemm(node *onnxpb.NodeProto, a, b, c *Tensor) (*Tensor, error) {
	alpha := floatAttrOr(node, "alpha", 1)
	beta := floatAttrOr(node, "beta", 1)
	transA := intAttrOr(node, "transA", 0) != 0
	transB := intAttrOr(node, "transB", 0) != 0

	aRows, aCols, err := dims2D(a, transA)
	if err != nil {
		return nil, fmt.Errorf("Gemm A: %w", err)
	}
	bRows, bCols, err := dims2D(b, transB)
	if err != nil {
		return nil, fmt.Errorf("Gemm B: %w", err)
	}
	if aCols != bRows {
		return nil, fmt.Errorf("Gemm: A is %dx%d (transA=%v), B is %dx%d (transB=%v): inner dimensions %d and %d don't match", aRows, aCols, transA, bRows, bCols, transB, aCols, bRows)
	}

	out := make([]float32, aRows*bCols)
	for i := 0; i < aRows; i++ {
		for j := 0; j < bCols; j++ {
			var sum float32
			for k := 0; k < aCols; k++ {
				sum += at2D(a, transA, i, k) * at2D(b, transB, k, j)
			}
			out[i*bCols+j] = alpha * sum
		}
	}

	if c != nil {
		if err := addBiasBroadcast(out, aRows, bCols, beta, c); err != nil {
			return nil, fmt.Errorf("Gemm C: %w", err)
		}
	}

	return &Tensor{Dims: []int64{int64(aRows), int64(bCols)}, Data: out}, nil
}

// addBiasBroadcast adds beta*c onto out ([rows, cols] row-major, already
// allocated) per the ONNX Gemm bias-broadcast rules this project's
// exporters produce: c may be a 1D [cols] vector (broadcasting across
// every row — pkg/onnxexport's own bias shape, see biasTensor's doc
// comment) or a 2D [1, cols] or [rows, cols] tensor.
func addBiasBroadcast(out []float32, rows, cols int, beta float32, c *Tensor) error {
	switch len(c.Dims) {
	case 1:
		if int(c.Dims[0]) != cols {
			return fmt.Errorf("1D bias has %d elements, want %d", c.Dims[0], cols)
		}
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				out[i*cols+j] += beta * c.Data[j]
			}
		}
		return nil
	case 2:
		cRows, cCols, err := dims2D(c, false)
		if err != nil {
			return err
		}
		if cCols != cols || (cRows != rows && cRows != 1) {
			return fmt.Errorf("C is %dx%d, not broadcastable onto result %dx%d", cRows, cCols, rows, cols)
		}
		for i := 0; i < rows; i++ {
			cRow := i
			if cRows == 1 {
				cRow = 0
			}
			for j := 0; j < cols; j++ {
				out[i*cols+j] += beta * c.Data[cRow*cCols+j]
			}
		}
		return nil
	default:
		return fmt.Errorf("bias tensor has %d dimensions (%v), want 1 or 2", len(c.Dims), c.Dims)
	}
}

// dims2D returns t's (rows, cols) as actually indexed given transpose,
// i.e. already swapped if transpose is set, matching the shape Gemm
// multiplies with.
func dims2D(t *Tensor, transpose bool) (rows, cols int, err error) {
	r, err := t.rows()
	if err != nil {
		return 0, 0, err
	}
	c, err := t.cols()
	if err != nil {
		return 0, 0, err
	}
	if transpose {
		return c, r, nil
	}
	return r, c, nil
}

// at2D reads t's logical (row, col) element, accounting for transpose
// (t's physical storage is never actually transposed — only the index
// math is).
func at2D(t *Tensor, transpose bool, row, col int) float32 {
	physCols := int(t.Dims[1])
	if transpose {
		row, col = col, row
	}
	return t.Data[row*physCols+col]
}

// evalElementwise applies op to a and b, which must have identical
// shapes — no broadcasting, since every Add/Mul this project's exporter
// emits (the LSTM gate decomposition) only ever combines two tensors of
// the same [1, hiddenSize] shape.
func evalElementwise(a, b *Tensor, op func(x, y float32) float32) (*Tensor, error) {
	if len(a.Data) != len(b.Data) {
		return nil, fmt.Errorf("shape mismatch: %v vs %v (no broadcasting supported)", a.Dims, b.Dims)
	}
	out := make([]float32, len(a.Data))
	for i := range a.Data {
		out[i] = op(a.Data[i], b.Data[i])
	}
	return &Tensor{Dims: append([]int64(nil), a.Dims...), Data: out}, nil
}

func evalUnary(a *Tensor, op func(float32) float32) *Tensor {
	out := make([]float32, len(a.Data))
	for i, v := range a.Data {
		out[i] = op(v)
	}
	return &Tensor{Dims: append([]int64(nil), a.Dims...), Data: out}
}

func sigmoidf(v float32) float32 { return 1 / (1 + expf(-v)) }

func tanhf(v float32) float32 { return float32(math.Tanh(float64(v))) }

func evalRelu(a *Tensor) *Tensor {
	out := make([]float32, len(a.Data))
	for i, v := range a.Data {
		if v > 0 {
			out[i] = v
		}
	}
	return &Tensor{Dims: append([]int64(nil), a.Dims...), Data: out}
}

// evalSoftmax implements Softmax over the last axis of a 2D tensor —
// the only axis configuration pkg/onnxexport ever emits (axis=-1, i.e.
// axis==len(dims)-1). Any other axis attribute is a hard error rather
// than silently computed wrong.
func evalSoftmax(node *onnxpb.NodeProto, a *Tensor) (*Tensor, error) {
	rows, cols, err := dims2D(a, false)
	if err != nil {
		return nil, fmt.Errorf("Softmax: %w", err)
	}

	axis := intAttrOr(node, "axis", -1)
	if axis < 0 {
		axis += int64(len(a.Dims))
	}
	if axis != int64(len(a.Dims)-1) {
		return nil, fmt.Errorf("Softmax: only the last axis is supported, got axis=%d over dims %v", intAttrOr(node, "axis", -1), a.Dims)
	}

	out := make([]float32, len(a.Data))
	for i := 0; i < rows; i++ {
		row := a.Data[i*cols : (i+1)*cols]
		var max float32 = row[0]
		for _, v := range row[1:] {
			if v > max {
				max = v
			}
		}
		var sum float32
		expRow := make([]float32, cols)
		for j, v := range row {
			e := expf(v - max)
			expRow[j] = e
			sum += e
		}
		for j, e := range expRow {
			out[i*cols+j] = e / sum
		}
	}
	return &Tensor{Dims: append([]int64(nil), a.Dims...), Data: out}, nil
}
