// Package mat implements a small dense, row-major float32 matrix type and
// the handful of operations (elementwise arithmetic, matrix multiplication,
// activation functions, and the REINFORCE policy-gradient loss) needed by
// the rest of this module.
//
// This is a deliberately minimal, dependency-free reimplementation of
// mat.c/mat.h from github.com/harshbhatt7585/cRL. See docs/04-numerical-stability-notes.md
// for why specific operations (softmax, the REINFORCE loss) are implemented
// the way they are.
package mat

import (
	"fmt"
	"math"
	"math/rand/v2"
)

// probEpsilon is the smallest probability value used when a probability is
// about to be passed to log() or used as a divisor, preventing -Inf/NaN
// results when a policy assigns (near) zero probability to an action.
const probEpsilon = 1e-8

// Matrix is a dense, row-major matrix of float32 values.
type Matrix struct {
	Rows int
	Cols int
	Data []float32
}

// New allocates a zero-valued Rows x Cols matrix.
func New(rows, cols int) *Matrix {
	return &Matrix{
		Rows: rows,
		Cols: cols,
		Data: make([]float32, rows*cols),
	}
}

// Clear zeroes every element of dst.
func (dst *Matrix) Clear() {
	clear(dst.Data)
}

// Fill sets every element of dst to value.
func (dst *Matrix) Fill(value float32) {
	for i := range dst.Data {
		dst.Data[i] = value
	}
}

// FillRand fills dst with independent uniform samples in [lower, upper),
// drawn from rng. Callers own rng, so concurrent callers must use distinct
// generators (see docs/05-porting-notes.md for why this package never
// touches a shared/global random source).
func (dst *Matrix) FillRand(rng *rand.Rand, lower, upper float32) {
	for i := range dst.Data {
		dst.Data[i] = rng.Float32()*(upper-lower) + lower
	}
}

// Scale multiplies every element of dst by factor, in place.
func (dst *Matrix) Scale(factor float32) {
	for i := range dst.Data {
		dst.Data[i] *= factor
	}
}

func checkSameShape(a, b *Matrix) error {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		return fmt.Errorf("mat: shape mismatch: %dx%d vs %dx%d", a.Rows, a.Cols, b.Rows, b.Cols)
	}
	return nil
}

// Add computes dst = a + b elementwise.
func (dst *Matrix) Add(a, b *Matrix) error {
	if err := checkSameShape(a, b); err != nil {
		return err
	}
	if err := checkSameShape(dst, a); err != nil {
		return err
	}

	for i := range dst.Data {
		dst.Data[i] = a.Data[i] + b.Data[i]
	}
	return nil
}

// Sub computes dst = a - b elementwise.
func (dst *Matrix) Sub(a, b *Matrix) error {
	if err := checkSameShape(a, b); err != nil {
		return err
	}
	if err := checkSameShape(dst, a); err != nil {
		return err
	}

	for i := range dst.Data {
		dst.Data[i] = a.Data[i] - b.Data[i]
	}
	return nil
}

// MatMul computes dst = op(a) * op(b), where op(x) is x or its transpose
// depending on transposeA/transposeB. If zeroOut is true, dst is cleared
// first; otherwise results are accumulated into dst's existing values
// (used by the autograd backward pass to accumulate gradients).
func (dst *Matrix) MatMul(a, b *Matrix, zeroOut, transposeA, transposeB bool) error {
	aRows, aCols := a.Rows, a.Cols
	if transposeA {
		aRows, aCols = a.Cols, a.Rows
	}

	bRows, bCols := b.Rows, b.Cols
	if transposeB {
		bRows, bCols = b.Cols, b.Rows
	}

	if aCols != bRows {
		return fmt.Errorf("mat: matmul inner dimension mismatch: a is %dx%d, b is %dx%d (after transposition)", aRows, aCols, bRows, bCols)
	}
	if dst.Rows != aRows || dst.Cols != bCols {
		return fmt.Errorf("mat: matmul output shape mismatch: want %dx%d, got %dx%d", aRows, bCols, dst.Rows, dst.Cols)
	}

	if zeroOut {
		dst.Clear()
	}

	switch {
	case !transposeA && !transposeB:
		matMulNN(dst, a, b)
	case !transposeA && transposeB:
		matMulNT(dst, a, b)
	case transposeA && !transposeB:
		matMulTN(dst, a, b)
	default:
		matMulTT(dst, a, b)
	}
	return nil
}

// matMulNN computes dst += a * b.
func matMulNN(dst, a, b *Matrix) {
	for i := range dst.Rows {
		for k := range a.Cols {
			aVal := a.Data[k+i*a.Cols]
			for j := range dst.Cols {
				dst.Data[j+i*dst.Cols] += aVal * b.Data[j+k*b.Cols]
			}
		}
	}
}

// matMulNT computes dst += a * b^T.
func matMulNT(dst, a, b *Matrix) {
	for i := range dst.Rows {
		for k := range a.Cols {
			aVal := a.Data[k+i*a.Cols]
			for j := range dst.Cols {
				dst.Data[j+i*dst.Cols] += aVal * b.Data[k+j*b.Cols]
			}
		}
	}
}

// matMulTN computes dst += a^T * b.
func matMulTN(dst, a, b *Matrix) {
	for i := range dst.Rows {
		for k := range a.Rows {
			aVal := a.Data[i+k*a.Cols]
			for j := range dst.Cols {
				dst.Data[j+i*dst.Cols] += aVal * b.Data[j+k*b.Cols]
			}
		}
	}
}

// matMulTT computes dst += a^T * b^T.
func matMulTT(dst, a, b *Matrix) {
	for i := range dst.Rows {
		for k := range a.Rows {
			aVal := a.Data[i+k*a.Cols]
			for j := range dst.Cols {
				dst.Data[j+i*dst.Cols] += aVal * b.Data[k+j*b.Cols]
			}
		}
	}
}

// ReLU computes dst = max(0, in) elementwise.
func (dst *Matrix) ReLU(in *Matrix) error {
	if err := checkSameShape(dst, in); err != nil {
		return err
	}

	for i, v := range in.Data {
		if v > 0 {
			dst.Data[i] = v
		} else {
			dst.Data[i] = 0
		}
	}
	return nil
}

// Softmax computes dst = softmax(in), a probability distribution over all
// elements of in. The maximum value is subtracted before exponentiating
// (the standard "max-subtraction trick") so that exp() never overflows,
// which would otherwise happen for even moderately large input values.
// See docs/04-numerical-stability-notes.md for why this is necessary.
func (dst *Matrix) Softmax(in *Matrix) error {
	if err := checkSameShape(dst, in); err != nil {
		return err
	}
	if len(in.Data) == 0 {
		return nil
	}

	maxValue := in.Data[0]
	for _, v := range in.Data[1:] {
		maxValue = max(maxValue, v)
	}

	var sum float32
	for i, v := range in.Data {
		e := float32(math.Exp(float64(v - maxValue)))
		dst.Data[i] = e
		sum += e
	}

	dst.Scale(1.0 / sum)
	return nil
}

// ReinforceLoss computes the REINFORCE policy-gradient loss,
// dst[i] = -log(max(probs[i], probEpsilon)) * advantages[i],
// elementwise. See docs/03-policy-gradients-and-reinforce.md for what this
// loss means and why it's shaped this way.
func (dst *Matrix) ReinforceLoss(probs, advantages *Matrix) error {
	if err := checkSameShape(probs, advantages); err != nil {
		return err
	}
	if err := checkSameShape(dst, advantages); err != nil {
		return err
	}

	for i, p := range probs.Data {
		clamped := max(p, probEpsilon)
		dst.Data[i] = float32(-math.Log(float64(clamped))) * advantages.Data[i]
	}
	return nil
}

// ReinforceAddGrad accumulates the gradient of ReinforceLoss with respect
// to probs into dst (dst[i] += grad[i] * (-rt[i] / max(probs[i], probEpsilon))).
// rt holds the same advantage values that were passed to ReinforceLoss.
func (dst *Matrix) ReinforceAddGrad(probs, rt, grad *Matrix) error {
	if err := checkSameShape(probs, rt); err != nil {
		return err
	}
	if err := checkSameShape(grad, probs); err != nil {
		return err
	}
	if err := checkSameShape(dst, probs); err != nil {
		return err
	}

	for i, p := range probs.Data {
		clamped := max(p, probEpsilon)
		dst.Data[i] += grad.Data[i] * (-rt.Data[i] / clamped)
	}
	return nil
}

// ReLUAddGrad accumulates the gradient of ReLU with respect to its input
// into dst: dst[i] += grad[i] if in[i] > 0, otherwise 0 is added.
func (dst *Matrix) ReLUAddGrad(in, grad *Matrix) error {
	if err := checkSameShape(dst, in); err != nil {
		return err
	}
	if err := checkSameShape(dst, grad); err != nil {
		return err
	}

	for i, v := range in.Data {
		if v > 0 {
			dst.Data[i] += grad.Data[i]
		}
	}
	return nil
}

// SoftmaxAddGrad accumulates the gradient of Softmax with respect to its
// input into dst, using the standard softmax Jacobian-vector product:
// dst[i] += softmaxOut[i] * (grad[i] - dot(grad, softmaxOut)).
// See docs/02-autograd-and-backpropagation.md for a derivation.
func (dst *Matrix) SoftmaxAddGrad(softmaxOut, grad *Matrix) error {
	if err := checkSameShape(dst, softmaxOut); err != nil {
		return err
	}
	if err := checkSameShape(grad, softmaxOut); err != nil {
		return err
	}

	var dot float32
	for i, g := range grad.Data {
		dot += g * softmaxOut.Data[i]
	}

	for i, s := range softmaxOut.Data {
		dst.Data[i] += s * (grad.Data[i] - dot)
	}
	return nil
}
