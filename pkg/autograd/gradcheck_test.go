package autograd

// This file validates the autograd engine using gradient checking: the
// standard ML-engineering technique of comparing an analytically computed
// gradient (from Backward) against a numerical estimate obtained by
// perturbing each input by a small epsilon and measuring how the output
// changes. See docs/02-autograd-and-backpropagation.md for why this is a
// reliable way to catch backward-pass bugs.

import (
	"math/rand/v2"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/mat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	gradCheckEpsilon   = 1e-2
	gradCheckTolerance = 5e-2
)

// sumMatrix sums every element of m as a float64, used as the scalar
// objective for gradient checking (Backward's semantics are equivalent to
// computing d(sum(output))/d(input), since the output's gradient is
// seeded with all-ones).
func sumMatrix(m *mat.Matrix) float64 {
	var sum float64
	for _, v := range m.Data {
		sum += float64(v)
	}
	return sum
}

// numericalGradient estimates d(sum(output.Val))/d(x.Val) via central
// finite differences, re-running forward after every perturbation.
func numericalGradient(x, output *Var, forward func()) *mat.Matrix {
	grad := mat.New(x.Val.Rows, x.Val.Cols)

	for i := range x.Val.Data {
		original := x.Val.Data[i]

		x.Val.Data[i] = original + gradCheckEpsilon
		forward()
		plus := sumMatrix(output.Val)

		x.Val.Data[i] = original - gradCheckEpsilon
		forward()
		minus := sumMatrix(output.Val)

		x.Val.Data[i] = original
		grad.Data[i] = float32((plus - minus) / (2 * gradCheckEpsilon))
	}

	forward() // leave output.Val in its unperturbed state
	return grad
}

func assertMatricesClose(t *testing.T, want, got *mat.Matrix, tolerance float64) {
	t.Helper()
	require.Equal(t, want.Rows, got.Rows)
	require.Equal(t, want.Cols, got.Cols)

	for i := range want.Data {
		assert.InDelta(t, want.Data[i], got.Data[i], tolerance, "element %d differs", i)
	}
}

func TestGradientCheckMatMul(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))

	w := NewVar(2, 3, FlagRequiresGrad)
	w.Val.FillRand(rng, -1, 1)
	x := NewVar(3, 1, FlagRequiresGrad)
	x.Val.FillRand(rng, -1, 1)

	out, err := MatMul(w, x)
	require.NoError(t, err)

	graph := BuildGraph(out)
	graph.Forward()
	graph.Backward()

	assertMatricesClose(t, w.Grad, numericalGradient(w, out, graph.Forward), gradCheckTolerance)
	assertMatricesClose(t, x.Grad, numericalGradient(x, out, graph.Forward), gradCheckTolerance)
}

func TestGradientCheckAdd(t *testing.T) {
	a := &Var{Flags: FlagRequiresGrad, Val: &mat.Matrix{Rows: 2, Cols: 2, Data: []float32{1, 2, 3, 4}}}
	a.Grad = mat.New(2, 2)
	b := &Var{Flags: FlagRequiresGrad, Val: &mat.Matrix{Rows: 2, Cols: 2, Data: []float32{5, 6, 7, 8}}}
	b.Grad = mat.New(2, 2)

	out, err := Add(a, b)
	require.NoError(t, err)

	graph := BuildGraph(out)
	graph.Forward()
	graph.Backward()

	assertMatricesClose(t, a.Grad, numericalGradient(a, out, graph.Forward), gradCheckTolerance)
	assertMatricesClose(t, b.Grad, numericalGradient(b, out, graph.Forward), gradCheckTolerance)
}

func TestGradientCheckSub(t *testing.T) {
	a := &Var{Flags: FlagRequiresGrad, Val: &mat.Matrix{Rows: 2, Cols: 1, Data: []float32{1, 2}}}
	a.Grad = mat.New(2, 1)
	b := &Var{Flags: FlagRequiresGrad, Val: &mat.Matrix{Rows: 2, Cols: 1, Data: []float32{3, 4}}}
	b.Grad = mat.New(2, 1)

	out, err := Sub(a, b)
	require.NoError(t, err)

	graph := BuildGraph(out)
	graph.Forward()
	graph.Backward()

	assertMatricesClose(t, a.Grad, numericalGradient(a, out, graph.Forward), gradCheckTolerance)
	assertMatricesClose(t, b.Grad, numericalGradient(b, out, graph.Forward), gradCheckTolerance)
}

func TestGradientCheckReLU(t *testing.T) {
	// Fixed, non-zero values: gradient checking a kink exactly at 0 is
	// inherently unstable (left/right derivatives disagree), so this
	// intentionally avoids landing exactly on the kink.
	x := &Var{Flags: FlagRequiresGrad, Val: &mat.Matrix{Rows: 4, Cols: 1, Data: []float32{-0.7, -0.3, 0.2, 0.9}}}
	x.Grad = mat.New(4, 1)

	out, err := ReLU(x)
	require.NoError(t, err)

	graph := BuildGraph(out)
	graph.Forward()
	graph.Backward()

	assertMatricesClose(t, x.Grad, numericalGradient(x, out, graph.Forward), gradCheckTolerance)
}

func TestGradientCheckSigmoid(t *testing.T) {
	// Composed with a fixed-weight MatMul for the same reason
	// TestGradientCheckSoftmax below does: sigmoid alone has no
	// particular constant-sum property, but a weighted combination
	// gives a cleaner non-trivial scalar objective to check.
	x := &Var{Flags: FlagRequiresGrad, Val: &mat.Matrix{Rows: 4, Cols: 1, Data: []float32{-0.7, -0.3, 0.2, 0.9}}}
	x.Grad = mat.New(4, 1)

	sig, err := Sigmoid(x)
	require.NoError(t, err)

	weights := &Var{Val: &mat.Matrix{Rows: 1, Cols: 4, Data: []float32{1, 2, 3, 4}}}
	out, err := MatMul(weights, sig)
	require.NoError(t, err)

	graph := BuildGraph(out)
	graph.Forward()
	graph.Backward()

	assertMatricesClose(t, x.Grad, numericalGradient(x, out, graph.Forward), gradCheckTolerance)
}

func TestGradientCheckTanh(t *testing.T) {
	x := &Var{Flags: FlagRequiresGrad, Val: &mat.Matrix{Rows: 4, Cols: 1, Data: []float32{-0.7, -0.3, 0.2, 0.9}}}
	x.Grad = mat.New(4, 1)

	tanh, err := Tanh(x)
	require.NoError(t, err)

	weights := &Var{Val: &mat.Matrix{Rows: 1, Cols: 4, Data: []float32{1, 2, 3, 4}}}
	out, err := MatMul(weights, tanh)
	require.NoError(t, err)

	graph := BuildGraph(out)
	graph.Forward()
	graph.Backward()

	assertMatricesClose(t, x.Grad, numericalGradient(x, out, graph.Forward), gradCheckTolerance)
}

func TestGradientCheckSoftmax(t *testing.T) {
	// sum(softmax(x)) == 1 for any x, so its gradient w.r.t. x is
	// trivially zero and wouldn't exercise Backward meaningfully.
	// Composing softmax with a fixed-weight MatMul produces a
	// non-constant scalar objective, giving a real gradient to check.
	x := &Var{Flags: FlagRequiresGrad, Val: &mat.Matrix{Rows: 4, Cols: 1, Data: []float32{0.5, -1.2, 2.0, 0.1}}}
	x.Grad = mat.New(4, 1)

	probs, err := Softmax(x)
	require.NoError(t, err)

	weights := &Var{Val: &mat.Matrix{Rows: 1, Cols: 4, Data: []float32{1, 2, 3, 4}}}
	out, err := MatMul(weights, probs)
	require.NoError(t, err)

	graph := BuildGraph(out)
	graph.Forward()
	graph.Backward()

	assertMatricesClose(t, x.Grad, numericalGradient(x, out, graph.Forward), gradCheckTolerance)
}

func TestGradientCheckReinforceLoss(t *testing.T) {
	probs := &Var{Flags: FlagRequiresGrad, Val: &mat.Matrix{Rows: 3, Cols: 1, Data: []float32{0.2, 0.5, 0.3}}}
	probs.Grad = mat.New(3, 1)
	advantages := &Var{Val: &mat.Matrix{Rows: 3, Cols: 1, Data: []float32{1.0, -2.0, 0.5}}}

	out, err := ReinforceLoss(probs, advantages)
	require.NoError(t, err)

	graph := BuildGraph(out)
	graph.Forward()
	graph.Backward()

	assertMatricesClose(t, probs.Grad, numericalGradient(probs, out, graph.Forward), gradCheckTolerance)
}

// TestGradientCheckReinforceLossWithActionMask confirms the masking
// approach pkg/policy.TrainingNetwork and pkg/actorcritic.TrainingNetwork
// use — adding a large negative bias to a masked action's pre-softmax
// logit, rather than renormalizing Softmax's output afterward — produces
// correct gradients for both the masked action and the surviving legal
// ones, exercising exactly the Add-then-Softmax-then-ReinforceLoss
// composition those packages build. See
// docs/plans/19-training-time-action-masking.md.
func TestGradientCheckReinforceLossWithActionMask(t *testing.T) {
	logits := &Var{Flags: FlagRequiresGrad, Val: &mat.Matrix{Rows: 3, Cols: 1, Data: []float32{0.4, -0.9, 1.3}}}
	logits.Grad = mat.New(3, 1)

	// Action 1 is masked out.
	maskBias := &Var{Val: &mat.Matrix{Rows: 3, Cols: 1, Data: []float32{0, -1e9, 0}}}

	maskedLogits, err := Add(logits, maskBias)
	require.NoError(t, err)

	probs, err := Softmax(maskedLogits)
	require.NoError(t, err)

	advantages := &Var{Val: &mat.Matrix{Rows: 3, Cols: 1, Data: []float32{1.0, -2.0, 0.5}}}
	out, err := ReinforceLoss(probs, advantages)
	require.NoError(t, err)

	graph := BuildGraph(out)
	graph.Forward()
	require.Equal(t, float32(0), probs.Val.Data[1], "the masked action must have exactly 0 probability")

	graph.Backward()

	assertMatricesClose(t, logits.Grad, numericalGradient(logits, out, graph.Forward), gradCheckTolerance)
}

func TestGradientCheckMul(t *testing.T) {
	rng := rand.New(rand.NewPCG(13, 17))

	a := NewVar(3, 1, FlagRequiresGrad)
	a.Val.FillRand(rng, -1, 1)
	b := NewVar(3, 1, FlagRequiresGrad)
	b.Val.FillRand(rng, -1, 1)

	out, err := Mul(a, b)
	require.NoError(t, err)

	graph := BuildGraph(out)
	graph.Forward()
	graph.Backward()

	assertMatricesClose(t, a.Grad, numericalGradient(a, out, graph.Forward), gradCheckTolerance)
	assertMatricesClose(t, b.Grad, numericalGradient(b, out, graph.Forward), gradCheckTolerance)
}

func TestGradientCheckNeg(t *testing.T) {
	rng := rand.New(rand.NewPCG(19, 23))

	x := NewVar(4, 1, FlagRequiresGrad)
	x.Val.FillRand(rng, -1, 1)

	out, err := Neg(x)
	require.NoError(t, err)

	graph := BuildGraph(out)
	graph.Forward()
	graph.Backward()

	assertMatricesClose(t, x.Grad, numericalGradient(x, out, graph.Forward), gradCheckTolerance)
}

func TestGradientCheckExp(t *testing.T) {
	rng := rand.New(rand.NewPCG(29, 31))

	x := NewVar(4, 1, FlagRequiresGrad)
	x.Val.FillRand(rng, -1, 1)

	out, err := Exp(x)
	require.NoError(t, err)

	graph := BuildGraph(out)
	graph.Forward()
	graph.Backward()

	assertMatricesClose(t, x.Grad, numericalGradient(x, out, graph.Forward), gradCheckTolerance)
}

func TestGradientCheckLog(t *testing.T) {
	// Fixed, comfortably-positive values, well away from Log's
	// probEpsilon clamp, so the check exercises the ordinary
	// d/dx log(x) = 1/x branch rather than the clamp.
	x := &Var{Flags: FlagRequiresGrad, Val: &mat.Matrix{Rows: 4, Cols: 1, Data: []float32{0.2, 0.5, 1.0, 2.0}}}
	x.Grad = mat.New(4, 1)

	out, err := Log(x)
	require.NoError(t, err)

	graph := BuildGraph(out)
	graph.Forward()
	graph.Backward()

	assertMatricesClose(t, x.Grad, numericalGradient(x, out, graph.Forward), gradCheckTolerance)
}

func TestGradientCheckMin(t *testing.T) {
	// Fixed, comfortably-distinct values at every position (no ties, and
	// no pair closer than numericalGradient's perturbation could flip):
	// gradient-checking Min at a tie is inherently unstable in the same
	// way TestGradientCheckReLU avoids landing exactly on ReLU's kink at
	// zero.
	a := &Var{Flags: FlagRequiresGrad, Val: &mat.Matrix{Rows: 3, Cols: 1, Data: []float32{-1.0, 0.5, 2.0}}}
	a.Grad = mat.New(3, 1)
	b := &Var{Flags: FlagRequiresGrad, Val: &mat.Matrix{Rows: 3, Cols: 1, Data: []float32{1.0, -0.5, 0.1}}}
	b.Grad = mat.New(3, 1)

	out, err := Min(a, b)
	require.NoError(t, err)

	graph := BuildGraph(out)
	graph.Forward()
	graph.Backward()

	assertMatricesClose(t, a.Grad, numericalGradient(a, out, graph.Forward), gradCheckTolerance)
	assertMatricesClose(t, b.Grad, numericalGradient(b, out, graph.Forward), gradCheckTolerance)
}

func TestGradientCheckMax(t *testing.T) {
	// Fixed, comfortably-distinct values at every position, for the same
	// tie-instability reason TestGradientCheckMin avoids randomly
	// generated inputs.
	a := &Var{Flags: FlagRequiresGrad, Val: &mat.Matrix{Rows: 3, Cols: 1, Data: []float32{-1.0, 0.5, 2.0}}}
	a.Grad = mat.New(3, 1)
	b := &Var{Flags: FlagRequiresGrad, Val: &mat.Matrix{Rows: 3, Cols: 1, Data: []float32{1.0, -0.5, 0.1}}}
	b.Grad = mat.New(3, 1)

	out, err := Max(a, b)
	require.NoError(t, err)

	graph := BuildGraph(out)
	graph.Forward()
	graph.Backward()

	assertMatricesClose(t, a.Grad, numericalGradient(a, out, graph.Forward), gradCheckTolerance)
	assertMatricesClose(t, b.Grad, numericalGradient(b, out, graph.Forward), gradCheckTolerance)
}

func TestGradientCheckClamp(t *testing.T) {
	// Values both inside and outside [lo, hi], each comfortably clear of
	// the boundary itself (clamp's kinks at lo/hi are as unstable to
	// gradient-check as ReLU's kink at zero).
	x := &Var{Flags: FlagRequiresGrad, Val: &mat.Matrix{Rows: 4, Cols: 1, Data: []float32{-2.0, 0.0, 0.5, 3.0}}}
	x.Grad = mat.New(4, 1)

	out, err := Clamp(x, -1.0, 1.0)
	require.NoError(t, err)

	graph := BuildGraph(out)
	graph.Forward()
	graph.Backward()

	assertMatricesClose(t, x.Grad, numericalGradient(x, out, graph.Forward), gradCheckTolerance)
}

// TestClampBoundsValuesOutsideRange confirms Clamp's forward pass
// actually clips out-of-range values rather than passing them through,
// which is the property the PPO clipped-surrogate objective
// (docs/plans/03-gae-and-ppo-objective.md) depends on: a probability
// ratio pushed far outside [1-ε, 1+ε] must be capped at the boundary,
// not left at its raw (unclamped) value.
func TestClampBoundsValuesOutsideRange(t *testing.T) {
	x := NewVar(5, 1, FlagNone)
	x.Val.Data = []float32{-10, -1, 0, 1, 10}

	out, err := Clamp(x, -1, 1)
	require.NoError(t, err)

	graph := BuildGraph(out)
	graph.Forward()

	assert.Equal(t, []float32{-1, -1, 0, 1, 1}, out.Val.Data)
}

// TestGradientCheckExpOfSubComposition composes Sub and Exp the same way
// a PPO probability ratio would (exp(logNew - logOld)), confirming the
// two new ops chain correctly rather than only working in isolation.
func TestGradientCheckExpOfSubComposition(t *testing.T) {
	rng := rand.New(rand.NewPCG(37, 41))

	a := NewVar(3, 1, FlagRequiresGrad)
	a.Val.FillRand(rng, -1, 1)
	b := NewVar(3, 1, FlagRequiresGrad)
	b.Val.FillRand(rng, -1, 1)

	diff, err := Sub(a, b)
	require.NoError(t, err)
	out, err := Exp(diff)
	require.NoError(t, err)

	graph := BuildGraph(out)
	graph.Forward()
	graph.Backward()

	assertMatricesClose(t, a.Grad, numericalGradient(a, out, graph.Forward), gradCheckTolerance)
	assertMatricesClose(t, b.Grad, numericalGradient(b, out, graph.Forward), gradCheckTolerance)
}

// TestGradientCheckSmallNetworkEndToEnd builds a graph with the same shape
// as the policy network's forward+loss computation (matmul -> add -> relu
// -> matmul -> add -> softmax -> reinforce loss) and checks the gradient
// with respect to the first layer's weights, exercising every op and the
// graph traversal together.
func TestGradientCheckSmallNetworkEndToEnd(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))

	input := NewVar(4, 1, FlagNone)
	input.Val.FillRand(rng, -1, 1)

	w0 := NewVar(5, 4, FlagRequiresGrad)
	w0.Val.FillRand(rng, -0.5, 0.5)
	b0 := NewVar(5, 1, FlagRequiresGrad)
	b0.Val.FillRand(rng, -0.5, 0.5)

	w1 := NewVar(3, 5, FlagRequiresGrad)
	w1.Val.FillRand(rng, -0.5, 0.5)
	b1 := NewVar(3, 1, FlagRequiresGrad)
	b1.Val.FillRand(rng, -0.5, 0.5)

	advantages := NewVar(3, 1, FlagNone)
	advantages.Val.FillRand(rng, -1, 1)

	z0, err := MatMul(w0, input)
	require.NoError(t, err)
	z0b, err := Add(z0, b0)
	require.NoError(t, err)
	a0, err := ReLU(z0b)
	require.NoError(t, err)

	z1, err := MatMul(w1, a0)
	require.NoError(t, err)
	z1b, err := Add(z1, b1)
	require.NoError(t, err)

	probs, err := Softmax(z1b)
	require.NoError(t, err)

	loss, err := ReinforceLoss(probs, advantages)
	require.NoError(t, err)

	graph := BuildGraph(loss)
	graph.Forward()
	graph.Backward()

	assertMatricesClose(t, w0.Grad, numericalGradient(w0, loss, graph.Forward), gradCheckTolerance)
	assertMatricesClose(t, b0.Grad, numericalGradient(b0, loss, graph.Forward), gradCheckTolerance)
	assertMatricesClose(t, w1.Grad, numericalGradient(w1, loss, graph.Forward), gradCheckTolerance)
	assertMatricesClose(t, b1.Grad, numericalGradient(b1, loss, graph.Forward), gradCheckTolerance)
}

// --- Batched recurrent training primitives ---
//
// AddBias, MulBroadcastCols, and per-column Softmax/SoftmaxAddGrad only
// started doing genuinely different work once batchSize > 1 (at
// batchSize == 1 they reduce to plain Add/a scalar multiply/the
// original whole-matrix Softmax, already covered above and throughout
// this package's other tests). These gradient-check each one at
// batchSize == 3 specifically, so a bug that only shows up once columns
// stop being trivially singular — e.g. broadcasting along the wrong
// axis, or letting one column's dot-product/sum leak into another's —
// gets caught here rather than surfacing later as training that
// mysteriously doesn't learn.

func TestGradientCheckAddBiasBroadcastsAcrossBatchColumns(t *testing.T) {
	rng := rand.New(rand.NewPCG(10, 20))
	const rows, batchSize = 4, 3

	in := NewVar(rows, batchSize, FlagRequiresGrad)
	in.Val.FillRand(rng, -1, 1)
	bias := NewVar(rows, 1, FlagRequiresGrad)
	bias.Val.FillRand(rng, -1, 1)

	out, err := AddBias(in, bias)
	require.NoError(t, err)

	// Compose with a weight that varies per column so collapsing all
	// columns together (the bug a broadcast in the wrong direction would
	// produce) would change the gradient, unlike a sum that treats every
	// column identically.
	weight := NewVar(rows, batchSize, FlagNone)
	weight.Val.FillRand(rng, -2, 2)
	weighted, err := Mul(out, weight)
	require.NoError(t, err)

	graph := BuildGraph(weighted)
	graph.Forward()
	graph.Backward()

	assertMatricesClose(t, in.Grad, numericalGradient(in, weighted, graph.Forward), gradCheckTolerance)
	assertMatricesClose(t, bias.Grad, numericalGradient(bias, weighted, graph.Forward), gradCheckTolerance)
}

func TestGradientCheckMulBroadcastColsBroadcastsPerColumnScalar(t *testing.T) {
	rng := rand.New(rand.NewPCG(30, 40))
	const rows, batchSize = 5, 3

	a := NewVar(rows, batchSize, FlagRequiresGrad)
	a.Val.FillRand(rng, -1, 1)
	row := NewVar(1, batchSize, FlagRequiresGrad)
	row.Val.FillRand(rng, -1, 1)

	out, err := MulBroadcastCols(a, row)
	require.NoError(t, err)

	weight := NewVar(rows, batchSize, FlagNone)
	weight.Val.FillRand(rng, -2, 2)
	weighted, err := Mul(out, weight)
	require.NoError(t, err)

	graph := BuildGraph(weighted)
	graph.Forward()
	graph.Backward()

	assertMatricesClose(t, a.Grad, numericalGradient(a, weighted, graph.Forward), gradCheckTolerance)
	assertMatricesClose(t, row.Grad, numericalGradient(row, weighted, graph.Forward), gradCheckTolerance)
}

// TestGradientCheckBatchedSoftmaxIsIndependentPerColumn gradient-checks
// Softmax at batchSize > 1 end to end through ReinforceLoss (mirroring
// TestGradientCheckReinforceLossWithActionMask's single-column version),
// with every column given a different action mask and advantage. A
// softmax that accidentally normalized or differentiated across the
// whole matrix instead of per column would still pass a same-input,
// single-column test but fail here, since the columns' own
// distributions would then depend on each other.
func TestGradientCheckBatchedSoftmaxIsIndependentPerColumn(t *testing.T) {
	const outputSize, batchSize = 3, 3

	logits := NewVar(outputSize, batchSize, FlagRequiresGrad)
	logits.Val.Data = []float32{
		0.4, 1.1, -0.3,
		-0.9, 0.2, 0.8,
		1.3, -1.4, 0.5,
	}
	logits.Grad = mat.New(outputSize, batchSize)

	// A different action masked out per column (column 0: action 1;
	// column 1: action 0; column 2: action 2).
	maskBias := &Var{Val: &mat.Matrix{Rows: outputSize, Cols: batchSize, Data: []float32{
		0, -1e9, 0,
		-1e9, 0, 0,
		0, 0, -1e9,
	}}}

	maskedLogits, err := Add(logits, maskBias)
	require.NoError(t, err)

	probs, err := Softmax(maskedLogits)
	require.NoError(t, err)

	advantages := &Var{Val: &mat.Matrix{Rows: outputSize, Cols: batchSize, Data: []float32{
		1.0, 0, 0,
		0, -2.0, 0,
		0, 0, 0.5,
	}}}

	loss, err := ReinforceLoss(probs, advantages)
	require.NoError(t, err)

	graph := BuildGraph(loss)
	graph.Forward()
	graph.Backward()

	assertMatricesClose(t, logits.Grad, numericalGradient(logits, loss, graph.Forward), gradCheckTolerance)
}

// TestGradientCheckSmallNetworkWithEntropyBonusEndToEnd is
// TestGradientCheckSmallNetworkEndToEnd with an entropy bonus
// (entropyCoef*sum_a(-probs[a]*log(probs[a]))) subtracted from the
// REINFORCE loss, matching pkg/policy.NewTrainingNetwork's composition
// (see that package's buildEntropyBonus) — this is the shape of graph
// the entropy-regularization fix adds to pkg/reinforce's training
// network, so this gradient-checks that composition end to end rather
// than only the entropy terms in isolation.
func TestGradientCheckSmallNetworkWithEntropyBonusEndToEnd(t *testing.T) {
	rng := rand.New(rand.NewPCG(43, 47))

	input := NewVar(4, 1, FlagNone)
	input.Val.FillRand(rng, -1, 1)

	w0 := NewVar(5, 4, FlagRequiresGrad)
	w0.Val.FillRand(rng, -0.5, 0.5)
	b0 := NewVar(5, 1, FlagRequiresGrad)
	b0.Val.FillRand(rng, -0.5, 0.5)

	w1 := NewVar(3, 5, FlagRequiresGrad)
	w1.Val.FillRand(rng, -0.5, 0.5)
	b1 := NewVar(3, 1, FlagRequiresGrad)
	b1.Val.FillRand(rng, -0.5, 0.5)

	advantages := NewVar(3, 1, FlagNone)
	advantages.Val.FillRand(rng, -1, 1)

	z0, err := MatMul(w0, input)
	require.NoError(t, err)
	z0b, err := Add(z0, b0)
	require.NoError(t, err)
	a0, err := ReLU(z0b)
	require.NoError(t, err)

	z1, err := MatMul(w1, a0)
	require.NoError(t, err)
	z1b, err := Add(z1, b1)
	require.NoError(t, err)

	probs, err := Softmax(z1b)
	require.NoError(t, err)

	reinforceLoss, err := ReinforceLoss(probs, advantages)
	require.NoError(t, err)

	logProbs, err := Log(probs)
	require.NoError(t, err)
	perActionEntropy, err := Mul(probs, logProbs)
	require.NoError(t, err)
	positiveEntropy, err := Neg(perActionEntropy)
	require.NoError(t, err)
	entropyCoef := mat.New(3, 1)
	entropyCoef.Fill(0.1)
	entropyBonus, err := Mul(positiveEntropy, Constant(entropyCoef))
	require.NoError(t, err)
	negatedEntropyBonus, err := Neg(entropyBonus)
	require.NoError(t, err)

	loss, err := Add(reinforceLoss, negatedEntropyBonus)
	require.NoError(t, err)

	graph := BuildGraph(loss)
	graph.Forward()
	graph.Backward()

	assertMatricesClose(t, w0.Grad, numericalGradient(w0, loss, graph.Forward), gradCheckTolerance)
	assertMatricesClose(t, b0.Grad, numericalGradient(b0, loss, graph.Forward), gradCheckTolerance)
	assertMatricesClose(t, w1.Grad, numericalGradient(w1, loss, graph.Forward), gradCheckTolerance)
	assertMatricesClose(t, b1.Grad, numericalGradient(b1, loss, graph.Forward), gradCheckTolerance)
}
