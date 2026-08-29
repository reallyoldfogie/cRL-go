package autograd

import "github.com/reallyoldfogie/cRL-go/pkg/mat"

// sameShape is the Shape implementation shared by ops whose output shape
// equals every input's shape (Add, Sub, ReinforceLoss).
func sameShape(inputs ...*Var) (rows, cols int, ok bool) {
	if len(inputs) == 0 || inputs[0] == nil {
		return 0, 0, false
	}

	rows, cols = inputs[0].Val.Rows, inputs[0].Val.Cols
	for _, in := range inputs[1:] {
		if in == nil || in.Val.Rows != rows || in.Val.Cols != cols {
			return 0, 0, false
		}
	}
	return rows, cols, true
}

// reluOp is the ReLU activation: reluOp{} is a stateless, zero-size value
// constructed inline wherever a ReLU node is needed (see ReLU below), so
// no package-level Op instance is required.
type reluOp struct{}

func (reluOp) NumInputs() int { return 1 }

func (reluOp) Shape(inputs ...*Var) (int, int, bool) {
	if inputs[0] == nil {
		return 0, 0, false
	}
	return inputs[0].Val.Rows, inputs[0].Val.Cols, true
}

func (reluOp) Forward(v *Var) {
	_ = v.Val.ReLU(v.Inputs[0].Val)
}

func (reluOp) Backward(v *Var) {
	input := v.Inputs[0]
	if requiresGrad(input) {
		_ = input.Grad.ReLUAddGrad(input.Val, v.Grad)
	}
}

// softmaxOp is the softmax activation.
type softmaxOp struct{}

func (softmaxOp) NumInputs() int { return 1 }

func (softmaxOp) Shape(inputs ...*Var) (int, int, bool) {
	if inputs[0] == nil {
		return 0, 0, false
	}
	return inputs[0].Val.Rows, inputs[0].Val.Cols, true
}

func (softmaxOp) Forward(v *Var) {
	_ = v.Val.Softmax(v.Inputs[0].Val)
}

func (softmaxOp) Backward(v *Var) {
	input := v.Inputs[0]
	if requiresGrad(input) {
		_ = input.Grad.SoftmaxAddGrad(v.Val, v.Grad)
	}
}

// addOp is elementwise addition.
type addOp struct{}

func (addOp) NumInputs() int                        { return 2 }
func (addOp) Shape(inputs ...*Var) (int, int, bool) { return sameShape(inputs...) }

func (addOp) Forward(v *Var) {
	_ = v.Val.Add(v.Inputs[0].Val, v.Inputs[1].Val)
}

func (addOp) Backward(v *Var) {
	a, b := v.Inputs[0], v.Inputs[1]
	if requiresGrad(a) {
		_ = a.Grad.Add(a.Grad, v.Grad)
	}
	if requiresGrad(b) {
		_ = b.Grad.Add(b.Grad, v.Grad)
	}
}

// subOp is elementwise subtraction.
type subOp struct{}

func (subOp) NumInputs() int                        { return 2 }
func (subOp) Shape(inputs ...*Var) (int, int, bool) { return sameShape(inputs...) }

func (subOp) Forward(v *Var) {
	_ = v.Val.Sub(v.Inputs[0].Val, v.Inputs[1].Val)
}

func (subOp) Backward(v *Var) {
	a, b := v.Inputs[0], v.Inputs[1]
	if requiresGrad(a) {
		_ = a.Grad.Add(a.Grad, v.Grad)
	}
	if requiresGrad(b) {
		_ = b.Grad.Sub(b.Grad, v.Grad)
	}
}

// mulOp is elementwise (Hadamard) multiplication, distinct from
// matMulOp's matrix product.
type mulOp struct{}

func (mulOp) NumInputs() int                        { return 2 }
func (mulOp) Shape(inputs ...*Var) (int, int, bool) { return sameShape(inputs...) }

func (mulOp) Forward(v *Var) {
	_ = v.Val.Mul(v.Inputs[0].Val, v.Inputs[1].Val)
}

func (mulOp) Backward(v *Var) {
	a, b := v.Inputs[0], v.Inputs[1]
	if requiresGrad(a) {
		_ = a.Grad.MulAddGrad(b.Val, v.Grad)
	}
	if requiresGrad(b) {
		_ = b.Grad.MulAddGrad(a.Val, v.Grad)
	}
}

// minOp is elementwise minimum.
type minOp struct{}

func (minOp) NumInputs() int                        { return 2 }
func (minOp) Shape(inputs ...*Var) (int, int, bool) { return sameShape(inputs...) }

func (minOp) Forward(v *Var) {
	_ = v.Val.Min(v.Inputs[0].Val, v.Inputs[1].Val)
}

func (minOp) Backward(v *Var) {
	a, b := v.Inputs[0], v.Inputs[1]
	if requiresGrad(a) {
		_ = a.Grad.MinAddGrad(a.Val, b.Val, v.Grad, true)
	}
	if requiresGrad(b) {
		_ = b.Grad.MinAddGrad(a.Val, b.Val, v.Grad, false)
	}
}

// maxOp is elementwise maximum.
type maxOp struct{}

func (maxOp) NumInputs() int                        { return 2 }
func (maxOp) Shape(inputs ...*Var) (int, int, bool) { return sameShape(inputs...) }

func (maxOp) Forward(v *Var) {
	_ = v.Val.Max(v.Inputs[0].Val, v.Inputs[1].Val)
}

func (maxOp) Backward(v *Var) {
	a, b := v.Inputs[0], v.Inputs[1]
	if requiresGrad(a) {
		_ = a.Grad.MaxAddGrad(a.Val, b.Val, v.Grad, true)
	}
	if requiresGrad(b) {
		_ = b.Grad.MaxAddGrad(a.Val, b.Val, v.Grad, false)
	}
}

// negOp is elementwise negation.
type negOp struct{}

func (negOp) NumInputs() int { return 1 }

func (negOp) Shape(inputs ...*Var) (int, int, bool) {
	if inputs[0] == nil {
		return 0, 0, false
	}
	return inputs[0].Val.Rows, inputs[0].Val.Cols, true
}

func (negOp) Forward(v *Var) {
	_ = v.Val.Neg(v.Inputs[0].Val)
}

func (negOp) Backward(v *Var) {
	input := v.Inputs[0]
	if requiresGrad(input) {
		_ = input.Grad.NegAddGrad(v.Grad)
	}
}

// expOp is elementwise exponentiation.
type expOp struct{}

func (expOp) NumInputs() int { return 1 }

func (expOp) Shape(inputs ...*Var) (int, int, bool) {
	if inputs[0] == nil {
		return 0, 0, false
	}
	return inputs[0].Val.Rows, inputs[0].Val.Cols, true
}

func (expOp) Forward(v *Var) {
	_ = v.Val.Exp(v.Inputs[0].Val)
}

func (expOp) Backward(v *Var) {
	input := v.Inputs[0]
	if requiresGrad(input) {
		_ = input.Grad.ExpAddGrad(v.Val, v.Grad)
	}
}

// logOp is elementwise natural log.
type logOp struct{}

func (logOp) NumInputs() int { return 1 }

func (logOp) Shape(inputs ...*Var) (int, int, bool) {
	if inputs[0] == nil {
		return 0, 0, false
	}
	return inputs[0].Val.Rows, inputs[0].Val.Cols, true
}

func (logOp) Forward(v *Var) {
	_ = v.Val.Log(v.Inputs[0].Val)
}

func (logOp) Backward(v *Var) {
	input := v.Inputs[0]
	if requiresGrad(input) {
		_ = input.Grad.LogAddGrad(input.Val, v.Grad)
	}
}

// matMulOp is matrix multiplication: out = a * b.
type matMulOp struct{}

func (matMulOp) NumInputs() int { return 2 }

func (matMulOp) Shape(inputs ...*Var) (int, int, bool) {
	a, b := inputs[0], inputs[1]
	if a == nil || b == nil || a.Val.Cols != b.Val.Rows {
		return 0, 0, false
	}
	return a.Val.Rows, b.Val.Cols, true
}

func (matMulOp) Forward(v *Var) {
	_ = v.Val.MatMul(v.Inputs[0].Val, v.Inputs[1].Val, true, false, false)
}

// Backward applies the standard matrix-calculus rule for out = a * b:
//
//	dL/da = dL/dout * b^T
//	dL/db = a^T * dL/dout
func (matMulOp) Backward(v *Var) {
	a, b := v.Inputs[0], v.Inputs[1]
	if requiresGrad(a) {
		_ = a.Grad.MatMul(v.Grad, b.Val, false, false, true)
	}
	if requiresGrad(b) {
		_ = b.Grad.MatMul(a.Val, v.Grad, false, true, false)
	}
}

// reinforceLossOp computes the REINFORCE policy-gradient loss from action
// probabilities and an advantage signal. See
// docs/03-policy-gradients-and-reinforce.md for what this means.
type reinforceLossOp struct{}

func (reinforceLossOp) NumInputs() int                        { return 2 }
func (reinforceLossOp) Shape(inputs ...*Var) (int, int, bool) { return sameShape(inputs...) }

func (reinforceLossOp) Forward(v *Var) {
	_ = v.Val.ReinforceLoss(v.Inputs[0].Val, v.Inputs[1].Val)
}

func (reinforceLossOp) Backward(v *Var) {
	probs, advantages := v.Inputs[0], v.Inputs[1]
	if requiresGrad(probs) {
		_ = probs.Grad.ReinforceAddGrad(probs.Val, advantages.Val, v.Grad)
	}
}

// ReLU returns a new Var computing elementwise max(0, input).
func ReLU(input *Var) (*Var, error) {
	return newNode(reluOp{}, input)
}

// Softmax returns a new Var computing a probability distribution over
// input's elements.
func Softmax(input *Var) (*Var, error) {
	return newNode(softmaxOp{}, input)
}

// Add returns a new Var computing a + b elementwise.
func Add(a, b *Var) (*Var, error) {
	return newNode(addOp{}, a, b)
}

// Sub returns a new Var computing a - b elementwise.
func Sub(a, b *Var) (*Var, error) {
	return newNode(subOp{}, a, b)
}

// Mul returns a new Var computing a * b elementwise (the Hadamard
// product; see MatMul for the matrix product).
func Mul(a, b *Var) (*Var, error) {
	return newNode(mulOp{}, a, b)
}

// Min returns a new Var computing elementwise min(a, b).
func Min(a, b *Var) (*Var, error) {
	return newNode(minOp{}, a, b)
}

// Max returns a new Var computing elementwise max(a, b).
func Max(a, b *Var) (*Var, error) {
	return newNode(maxOp{}, a, b)
}

// Clamp returns a new Var computing elementwise min(max(x, lo), hi),
// commonly used to bound a value (e.g. a PPO probability ratio) into a
// fixed range. lo and hi are baked into constant Vars matching x's
// shape, built once at the point Clamp is called, so they don't need to
// be reconstructed on every Forward call.
func Clamp(x *Var, lo, hi float32) (*Var, error) {
	loMat := mat.New(x.Val.Rows, x.Val.Cols)
	loMat.Fill(lo)
	hiMat := mat.New(x.Val.Rows, x.Val.Cols)
	hiMat.Fill(hi)

	clampedLow, err := Max(x, Constant(loMat))
	if err != nil {
		return nil, err
	}
	return Min(clampedLow, Constant(hiMat))
}

// Neg returns a new Var computing elementwise -input.
func Neg(input *Var) (*Var, error) {
	return newNode(negOp{}, input)
}

// Exp returns a new Var computing elementwise exp(input).
func Exp(input *Var) (*Var, error) {
	return newNode(expOp{}, input)
}

// Log returns a new Var computing elementwise log(max(input, probEpsilon)),
// clamped the same way mat.Matrix.Log is, so it never produces -Inf or NaN.
func Log(input *Var) (*Var, error) {
	return newNode(logOp{}, input)
}

// MatMul returns a new Var computing the matrix product a * b.
func MatMul(a, b *Var) (*Var, error) {
	return newNode(matMulOp{}, a, b)
}

// ReinforceLoss returns a new Var computing the REINFORCE loss of probs
// (action probabilities) against advantages.
func ReinforceLoss(probs, advantages *Var) (*Var, error) {
	return newNode(reinforceLossOp{}, probs, advantages)
}
