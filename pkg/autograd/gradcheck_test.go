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
