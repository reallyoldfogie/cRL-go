// Package mat implements a small dense, row-major float32 matrix type and
// the handful of operations (elementwise arithmetic, matrix multiplication,
// activation functions, and the REINFORCE policy-gradient loss) needed by
// the rest of this module.
//
// This is a deliberately minimal reimplementation of mat.c/mat.h from
// github.com/harshbhatt7585/cRL, with one dependency (github.com/chewxy/math32,
// a float32-native math library — see its use in Sigmoid/Tanh/Exp/Log/Softmax
// below) in place of Go's float64-only standard math package. See
// docs/04-numerical-stability-notes.md for why specific operations (softmax,
// the REINFORCE loss) are implemented the way they are.
package mat

import (
	"fmt"
	"math/rand/v2"

	"github.com/chewxy/math32"
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

// SetColumn copies data (which must have exactly dst.Rows elements)
// into column col of dst. Used to build a batched (Rows, BatchSize)
// matrix one sample (one column) at a time, e.g. seeding a recurrent
// training batch's per-timestep input from several chunks' individual
// observations.
func (dst *Matrix) SetColumn(col int, data []float32) {
	for r, v := range data {
		dst.Data[r*dst.Cols+col] = v
	}
}

// Column copies column col of m into dst (which must have exactly
// m.Rows elements) — the strided-read counterpart to SetColumn, e.g.
// extracting one batch member's own policy output back out of a
// batched (Rows, BatchSize) result.
func (m *Matrix) Column(col int, dst []float32) {
	for r := range dst {
		dst[r] = m.Data[r*m.Cols+col]
	}
}

// ClearColumn zeroes column col of dst.
func (dst *Matrix) ClearColumn(col int) {
	for r := range dst.Rows {
		dst.Data[r*dst.Cols+col] = 0
	}
}

// FillColumn sets every element of column col of dst to value.
func (dst *Matrix) FillColumn(col int, value float32) {
	for r := range dst.Rows {
		dst.Data[r*dst.Cols+col] = value
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

// Mul computes dst = a * b elementwise (the Hadamard product, not the
// matrix product — see MatMul for that).
func (dst *Matrix) Mul(a, b *Matrix) error {
	if err := checkSameShape(a, b); err != nil {
		return err
	}
	if err := checkSameShape(dst, a); err != nil {
		return err
	}

	for i := range dst.Data {
		dst.Data[i] = a.Data[i] * b.Data[i]
	}
	return nil
}

// Min computes dst = min(a, b) elementwise.
func (dst *Matrix) Min(a, b *Matrix) error {
	if err := checkSameShape(a, b); err != nil {
		return err
	}
	if err := checkSameShape(dst, a); err != nil {
		return err
	}

	for i := range dst.Data {
		dst.Data[i] = min(a.Data[i], b.Data[i])
	}
	return nil
}

// Max computes dst = max(a, b) elementwise.
func (dst *Matrix) Max(a, b *Matrix) error {
	if err := checkSameShape(a, b); err != nil {
		return err
	}
	if err := checkSameShape(dst, a); err != nil {
		return err
	}

	for i := range dst.Data {
		dst.Data[i] = max(a.Data[i], b.Data[i])
	}
	return nil
}

// Neg computes dst = -in elementwise.
func (dst *Matrix) Neg(in *Matrix) error {
	if err := checkSameShape(dst, in); err != nil {
		return err
	}

	for i, v := range in.Data {
		dst.Data[i] = -v
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

// AddBias computes dst = in + bias, broadcasting bias (which must be
// (in.Rows, 1)) across every one of in's columns: dst[r][c] = in[r][c]
// + bias[r][0]. Unlike Add (which requires identical shapes on both
// sides), this is how a per-feature bias applies uniformly across a
// batch of samples stacked as columns — e.g. a batched recurrent
// training chunk, where in is (HiddenSize, BatchSize) but a layer's
// bias is still just (HiddenSize, 1), one value per feature regardless
// of batch size.
func (dst *Matrix) AddBias(in, bias *Matrix) error {
	if bias.Cols != 1 || bias.Rows != in.Rows {
		return fmt.Errorf("mat: addbias shape mismatch: in is %dx%d, bias is %dx%d, want bias %dx1", in.Rows, in.Cols, bias.Rows, bias.Cols, in.Rows)
	}
	if err := checkSameShape(dst, in); err != nil {
		return err
	}

	for r := range in.Rows {
		b := bias.Data[r]
		rowStart := r * in.Cols
		for c := range in.Cols {
			dst.Data[rowStart+c] = in.Data[rowStart+c] + b
		}
	}
	return nil
}

// BiasAddGrad accumulates AddBias' gradient with respect to its bias
// operand into dst (which must be (grad.Rows, 1)): dst[r][0] +=
// sum_c(grad[r][c]) — summed, not averaged, matching how every other
// gradient in this package accumulates (see e.g. MulAddGrad). AddBias'
// gradient with respect to its other (in) operand needs no dedicated
// function: d(in+bias)/d(in) is 1 everywhere, so it's the same
// pass-through accumulation as plain Add's own "a" operand (dst.Add(dst,
// grad)).
func (dst *Matrix) BiasAddGrad(grad *Matrix) error {
	if dst.Cols != 1 || dst.Rows != grad.Rows {
		return fmt.Errorf("mat: biasaddgrad shape mismatch: dst is %dx%d, grad is %dx%d, want dst %dx1", dst.Rows, dst.Cols, grad.Rows, grad.Cols, grad.Rows)
	}

	for r := range grad.Rows {
		var sum float32
		rowStart := r * grad.Cols
		for c := range grad.Cols {
			sum += grad.Data[rowStart+c]
		}
		dst.Data[r] += sum
	}
	return nil
}

// MulBroadcastCols computes dst = a * row, broadcasting row (which
// must be (1, a.Cols)) down every one of a's rows: dst[r][c] = a[r][c]
// * row[0][c]. This broadcasts a per-batch-column scalar into the
// one-hot row a sampled action occupies — e.g. PPO's value loss, one
// scalar per minibatch sample — which MatMul cannot express once a's
// column count exceeds row's row count (MatMul's shared contraction
// dimension requires them equal, collapsing to this exact broadcast
// only when a.Cols == 1).
func (dst *Matrix) MulBroadcastCols(a, row *Matrix) error {
	if row.Rows != 1 || row.Cols != a.Cols {
		return fmt.Errorf("mat: mulbroadcastcols shape mismatch: a is %dx%d, row is %dx%d, want row 1x%d", a.Rows, a.Cols, row.Rows, row.Cols, a.Cols)
	}
	if err := checkSameShape(dst, a); err != nil {
		return err
	}

	for r := range a.Rows {
		rowStart := r * a.Cols
		for c := range a.Cols {
			dst.Data[rowStart+c] = a.Data[rowStart+c] * row.Data[c]
		}
	}
	return nil
}

// MulBroadcastColsAddGradA accumulates MulBroadcastCols' gradient with
// respect to its a operand into dst: dst[r][c] += grad[r][c] *
// row[0][c] (row broadcast down every row, exactly as the forward pass
// broadcasts it).
func (dst *Matrix) MulBroadcastColsAddGradA(row, grad *Matrix) error {
	if err := checkSameShape(dst, grad); err != nil {
		return err
	}
	if row.Rows != 1 || row.Cols != dst.Cols {
		return fmt.Errorf("mat: mulbroadcastcolsaddgrada shape mismatch: dst is %dx%d, row is %dx%d, want row 1x%d", dst.Rows, dst.Cols, row.Rows, row.Cols, dst.Cols)
	}

	for r := range dst.Rows {
		rowStart := r * dst.Cols
		for c := range dst.Cols {
			dst.Data[rowStart+c] += grad.Data[rowStart+c] * row.Data[c]
		}
	}
	return nil
}

// MulBroadcastColsAddGradRow accumulates MulBroadcastCols' gradient
// with respect to its row operand into dst (which must be (1, a.Cols)):
// dst[0][c] += sum_r(a[r][c] * grad[r][c]).
func (dst *Matrix) MulBroadcastColsAddGradRow(a, grad *Matrix) error {
	if dst.Rows != 1 || dst.Cols != a.Cols {
		return fmt.Errorf("mat: mulbroadcastcolsaddgradrow shape mismatch: dst is %dx%d, a is %dx%d, want dst 1x%d", dst.Rows, dst.Cols, a.Rows, a.Cols, a.Cols)
	}
	if err := checkSameShape(a, grad); err != nil {
		return err
	}

	for c := range a.Cols {
		var sum float32
		for r := range a.Rows {
			idx := r*a.Cols + c
			sum += a.Data[idx] * grad.Data[idx]
		}
		dst.Data[c] += sum
	}
	return nil
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

// Softmax computes dst = softmax(in): a probability distribution over
// every column's rows, computed independently per column (so Softmax
// of a (Rows, 1) vector, in's only shape until batched recurrent
// training needed more, is exactly its one column's distribution — the
// per-column computation below reduces to the original whole-matrix one
// whenever Cols == 1). The maximum value within each column is
// subtracted before exponentiating (the standard "max-subtraction
// trick") so that exp() never overflows, which would otherwise happen
// for even moderately large input values. See
// docs/04-numerical-stability-notes.md for why this is necessary.
func (dst *Matrix) Softmax(in *Matrix) error {
	if err := checkSameShape(dst, in); err != nil {
		return err
	}
	if len(in.Data) == 0 {
		return nil
	}

	for c := range in.Cols {
		maxValue := in.Data[c]
		for r := 1; r < in.Rows; r++ {
			maxValue = max(maxValue, in.Data[r*in.Cols+c])
		}

		var sum float32
		for r := range in.Rows {
			idx := r*in.Cols + c
			e := math32.Exp(in.Data[idx] - maxValue)
			dst.Data[idx] = e
			sum += e
		}

		invSum := 1.0 / sum
		for r := range in.Rows {
			dst.Data[r*in.Cols+c] *= invSum
		}
	}
	return nil
}

// Sigmoid computes dst = 1 / (1 + exp(-in)) elementwise.
func (dst *Matrix) Sigmoid(in *Matrix) error {
	if err := checkSameShape(dst, in); err != nil {
		return err
	}

	for i, v := range in.Data {
		dst.Data[i] = 1.0 / (1.0 + math32.Exp(-v))
	}
	return nil
}

// Tanh computes dst = tanh(in) elementwise.
func (dst *Matrix) Tanh(in *Matrix) error {
	if err := checkSameShape(dst, in); err != nil {
		return err
	}

	for i, v := range in.Data {
		dst.Data[i] = math32.Tanh(v)
	}
	return nil
}

// Exp computes dst = exp(in) elementwise.
func (dst *Matrix) Exp(in *Matrix) error {
	if err := checkSameShape(dst, in); err != nil {
		return err
	}

	for i, v := range in.Data {
		dst.Data[i] = math32.Exp(v)
	}
	return nil
}

// Log computes dst = log(max(in, probEpsilon)) elementwise, using the
// same epsilon clamp as ReinforceLoss so that Log(0) (or any near-zero
// or negative input) cannot produce -Inf or NaN.
func (dst *Matrix) Log(in *Matrix) error {
	if err := checkSameShape(dst, in); err != nil {
		return err
	}

	for i, v := range in.Data {
		clamped := max(v, probEpsilon)
		dst.Data[i] = math32.Log(clamped)
	}
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
		dst.Data[i] = -math32.Log(clamped) * advantages.Data[i]
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

// SigmoidAddGrad accumulates the gradient of Sigmoid with respect to its
// input into dst: dst[i] += grad[i] * sigmoidOut[i] * (1 - sigmoidOut[i]),
// the standard sigmoid derivative expressed in terms of the forward
// output (sigmoidOut) rather than re-deriving it from the pre-activation
// input, mirroring SoftmaxAddGrad's own use of its forward output below.
func (dst *Matrix) SigmoidAddGrad(sigmoidOut, grad *Matrix) error {
	if err := checkSameShape(dst, sigmoidOut); err != nil {
		return err
	}
	if err := checkSameShape(dst, grad); err != nil {
		return err
	}

	for i, s := range sigmoidOut.Data {
		dst.Data[i] += grad.Data[i] * s * (1 - s)
	}
	return nil
}

// TanhAddGrad accumulates the gradient of Tanh with respect to its input
// into dst: dst[i] += grad[i] * (1 - tanhOut[i]^2), the standard tanh
// derivative expressed in terms of the forward output (tanhOut).
func (dst *Matrix) TanhAddGrad(tanhOut, grad *Matrix) error {
	if err := checkSameShape(dst, tanhOut); err != nil {
		return err
	}
	if err := checkSameShape(dst, grad); err != nil {
		return err
	}

	for i, t := range tanhOut.Data {
		dst.Data[i] += grad.Data[i] * (1 - t*t)
	}
	return nil
}

// SoftmaxAddGrad accumulates the gradient of Softmax with respect to
// its input into dst, using the standard softmax Jacobian-vector
// product computed independently per column (mirroring Softmax's own
// per-column forward pass — see its doc comment for why this reduces to
// the original whole-matrix computation whenever Cols == 1):
// dst[r][c] += softmaxOut[r][c] * (grad[r][c] - dot_c), where dot_c is
// column c's own dot(grad[:,c], softmaxOut[:,c]), not a dot product
// across the whole matrix. See docs/02-autograd-and-backpropagation.md
// for a derivation (of the single-column case; each column here is
// exactly that derivation applied independently).
func (dst *Matrix) SoftmaxAddGrad(softmaxOut, grad *Matrix) error {
	if err := checkSameShape(dst, softmaxOut); err != nil {
		return err
	}
	if err := checkSameShape(grad, softmaxOut); err != nil {
		return err
	}

	for c := range softmaxOut.Cols {
		var dot float32
		for r := range softmaxOut.Rows {
			idx := r*softmaxOut.Cols + c
			dot += grad.Data[idx] * softmaxOut.Data[idx]
		}

		for r := range softmaxOut.Rows {
			idx := r*softmaxOut.Cols + c
			dst.Data[idx] += softmaxOut.Data[idx] * (grad.Data[idx] - dot)
		}
	}
	return nil
}

// MulAddGrad accumulates the gradient of elementwise Mul with respect to
// one of its two inputs into dst: dst[i] += other[i] * grad[i], where
// other is the *other* input's forward value (since d(a*b)/da = b and,
// symmetrically, d(a*b)/db = a).
func (dst *Matrix) MulAddGrad(other, grad *Matrix) error {
	if err := checkSameShape(dst, other); err != nil {
		return err
	}
	if err := checkSameShape(dst, grad); err != nil {
		return err
	}

	for i := range dst.Data {
		dst.Data[i] += other.Data[i] * grad.Data[i]
	}
	return nil
}

// MinAddGrad accumulates the gradient of elementwise Min(a, b) into dst,
// which must be the gradient buffer belonging to a if dstIsA is true, or
// to b otherwise: dst[i] += grad[i] wherever the input dst belongs to is
// the smaller of the two at that position. At a tie (a[i] == b[i]), the
// gradient is routed to a only, so that the two calls needed to
// backpropagate through Min (one per input) never double-count a tied
// element between them — mirroring how ReLUAddGrad picks one side of its
// own kink at zero rather than splitting the gradient.
func (dst *Matrix) MinAddGrad(a, b, grad *Matrix, dstIsA bool) error {
	if err := checkSameShape(a, b); err != nil {
		return err
	}
	if err := checkSameShape(dst, a); err != nil {
		return err
	}
	if err := checkSameShape(dst, grad); err != nil {
		return err
	}

	for i := range dst.Data {
		aIsSmallerOrTied := a.Data[i] <= b.Data[i]
		if aIsSmallerOrTied == dstIsA {
			dst.Data[i] += grad.Data[i]
		}
	}
	return nil
}

// MaxAddGrad accumulates the gradient of elementwise Max(a, b) into dst,
// mirroring MinAddGrad exactly but for the larger of the two inputs:
// dst[i] += grad[i] wherever the input dst belongs to is the larger of
// the two at that position, with ties routed to a only.
func (dst *Matrix) MaxAddGrad(a, b, grad *Matrix, dstIsA bool) error {
	if err := checkSameShape(a, b); err != nil {
		return err
	}
	if err := checkSameShape(dst, a); err != nil {
		return err
	}
	if err := checkSameShape(dst, grad); err != nil {
		return err
	}

	for i := range dst.Data {
		aIsLargerOrTied := a.Data[i] >= b.Data[i]
		if aIsLargerOrTied == dstIsA {
			dst.Data[i] += grad.Data[i]
		}
	}
	return nil
}

// NegAddGrad accumulates the gradient of elementwise Neg into dst:
// dst[i] -= grad[i], since d(-x)/dx = -1.
func (dst *Matrix) NegAddGrad(grad *Matrix) error {
	if err := checkSameShape(dst, grad); err != nil {
		return err
	}

	for i := range dst.Data {
		dst.Data[i] -= grad.Data[i]
	}
	return nil
}

// ExpAddGrad accumulates the gradient of elementwise Exp into dst:
// dst[i] += grad[i] * expOut[i]. expOut is Exp's already-computed
// forward output rather than its input, since d(exp(x))/dx = exp(x) and
// reusing the forward value avoids recomputing it.
func (dst *Matrix) ExpAddGrad(expOut, grad *Matrix) error {
	if err := checkSameShape(dst, expOut); err != nil {
		return err
	}
	if err := checkSameShape(dst, grad); err != nil {
		return err
	}

	for i := range dst.Data {
		dst.Data[i] += grad.Data[i] * expOut.Data[i]
	}
	return nil
}

// LogAddGrad accumulates the gradient of elementwise Log into dst:
// dst[i] += grad[i] / max(in[i], probEpsilon), matching Log's own
// forward clamp so the two stay consistent for inputs at or below
// probEpsilon.
func (dst *Matrix) LogAddGrad(in, grad *Matrix) error {
	if err := checkSameShape(dst, in); err != nil {
		return err
	}
	if err := checkSameShape(dst, grad); err != nil {
		return err
	}

	for i, v := range in.Data {
		clamped := max(v, probEpsilon)
		dst.Data[i] += grad.Data[i] / clamped
	}
	return nil
}
