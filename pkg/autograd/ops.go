package autograd

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

// MatMul returns a new Var computing the matrix product a * b.
func MatMul(a, b *Var) (*Var, error) {
	return newNode(matMulOp{}, a, b)
}

// ReinforceLoss returns a new Var computing the REINFORCE loss of probs
// (action probabilities) against advantages.
func ReinforceLoss(probs, advantages *Var) (*Var, error) {
	return newNode(reinforceLossOp{}, probs, advantages)
}
