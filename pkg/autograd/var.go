// Package autograd implements a small reverse-mode automatic
// differentiation ("autodiff") engine: a graph of Var nodes connected by
// Ops, where a forward pass computes values and a backward pass computes
// gradients via the chain rule. See docs/02-autograd-and-backpropagation.md
// for an explanation of what this means and why it works.
//
// This is a reimplementation of autograd.c/autograd.h from
// github.com/harshbhatt7585/cRL, restructured around Go interfaces instead
// of the original's hand-rolled function-pointer vtable (VarType).
package autograd

import (
	"fmt"

	"github.com/reallyoldfogie/cRL-go/pkg/mat"
)

// VarFlag holds bit flags describing a Var's role in the graph.
type VarFlag uint32

const (
	// FlagNone marks a Var with no special role (e.g. a constant input).
	FlagNone VarFlag = 0
	// FlagRequiresGrad marks a Var whose gradient should be computed
	// during the backward pass.
	FlagRequiresGrad VarFlag = 1 << 0
	// FlagParameter marks a Var as a learnable parameter (weight/bias).
	// Parameter gradients are not cleared at the start of each backward
	// pass, so callers can accumulate gradients across multiple forward/
	// backward calls (e.g. across every step of every trajectory in a
	// training batch) before applying a single optimizer step.
	FlagParameter VarFlag = 1 << 1
)

// Op is a differentiable operation that can be attached to a Var. Forward
// computes v.Val from v.Inputs; Backward accumulates gradients into each
// input's Grad from v.Grad. Op implementations are stateless and are
// constructed fresh at each call site (see ops.go), so this package never
// needs a package-level Op registry or other shared mutable state.
type Op interface {
	// NumInputs reports how many inputs this op expects.
	NumInputs() int
	// Shape computes the output shape given the op's inputs, returning
	// ok=false if the inputs are missing or have incompatible shapes.
	Shape(inputs ...*Var) (rows, cols int, ok bool)
	// Forward computes v.Val from v.Inputs.
	Forward(v *Var)
	// Backward accumulates gradients into each input's Grad, reading
	// v.Grad (the gradient of the final objective with respect to v).
	Backward(v *Var)
}

// Var is one node in a computation graph: either a leaf (Op == nil, e.g.
// model input or a learnable parameter) or the result of applying Op to
// Inputs.
type Var struct {
	Flags VarFlag

	// Val holds this node's current value, computed by the forward pass
	// (or set directly by the caller for leaf nodes).
	Val *mat.Matrix
	// Grad holds the gradient of the final objective with respect to
	// Val, computed by the backward pass. Grad is nil unless
	// FlagRequiresGrad is set.
	Grad *mat.Matrix

	Op     Op
	Inputs []*Var
}

// NewVar creates a leaf Var of the given shape. If flags includes
// FlagRequiresGrad, a same-shaped Grad matrix is also allocated.
func NewVar(rows, cols int, flags VarFlag) *Var {
	v := &Var{
		Flags: flags,
		Val:   mat.New(rows, cols),
	}
	if flags&FlagRequiresGrad != 0 {
		v.Grad = mat.New(rows, cols)
	}
	return v
}

func requiresGrad(v *Var) bool {
	return v != nil && v.Flags&FlagRequiresGrad != 0
}

// Constant wraps an existing matrix as a graph leaf that does not track a
// gradient. val is aliased, not copied, so mutations to it (or to the Var
// returned here) are visible to anyone else holding val. This is used to
// build cheap, read-only inference graphs that share parameters with a
// separate training graph without copying weight data (see
// docs/05-porting-notes.md for why this matters for concurrent rollout
// collection).
func Constant(val *mat.Matrix) *Var {
	return &Var{Val: val, Flags: FlagNone}
}

// Parameter wraps an existing matrix as a leaf that accumulates a
// gradient across Backward calls (FlagRequiresGrad|FlagParameter). val is
// aliased, not copied. The caller is responsible for clearing Grad after
// applying an optimizer step (see Graph.Backward's documentation on why
// parameter gradients are never cleared automatically).
func Parameter(val *mat.Matrix) *Var {
	return &Var{
		Val:   val,
		Flags: FlagRequiresGrad | FlagParameter,
		Grad:  mat.New(val.Rows, val.Cols),
	}
}

// newNode builds a Var representing op(inputs...), computing its shape and
// propagating FlagRequiresGrad from the inputs (a node needs a gradient if
// any of its inputs do).
func newNode(op Op, inputs ...*Var) (*Var, error) {
	if len(inputs) != op.NumInputs() {
		return nil, fmt.Errorf("autograd: %T expects %d input(s), got %d", op, op.NumInputs(), len(inputs))
	}

	rows, cols, ok := op.Shape(inputs...)
	if !ok {
		return nil, fmt.Errorf("autograd: %T: incompatible input shapes", op)
	}

	flags := FlagNone
	for _, in := range inputs {
		if requiresGrad(in) {
			flags = FlagRequiresGrad
			break
		}
	}

	v := NewVar(rows, cols, flags)
	v.Op = op
	v.Inputs = inputs
	return v, nil
}
