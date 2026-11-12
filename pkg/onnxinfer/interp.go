// Package onnxinfer runs the small, fixed set of ONNX graphs
// pkg/onnxexport produces. It is deliberately not a general ONNX
// executor: Run fails loudly on any node type it doesn't recognize
// rather than attempting a fallback, because this package exists to
// replay exactly what this project's own exporter emits — not to load
// an arbitrary third-party .onnx file. See docs/plans/23-ONNX-support.md
// §4.4 and §1 ("No generic ONNX importer").
package onnxinfer

import (
	"fmt"

	onnxpb "github.com/reallyoldfogie/cRL-go/pkg/onnx"
	"google.golang.org/protobuf/proto"
)

// Tensor is a dense, row-major float32 tensor — the only element type
// and layout this package ever produces or consumes, matching both
// ONNX's own on-disk convention and cRL-go/pkg/mat.Matrix's.
type Tensor struct {
	Dims []int64
	Data []float32
}

// rows/cols treat t as a 2D matrix; every tensor this package's
// supported ops touch is 2D (see package doc comment).
func (t *Tensor) rows() (int, error) {
	if len(t.Dims) != 2 {
		return 0, fmt.Errorf("onnxinfer: expected a 2D tensor, got dims %v", t.Dims)
	}
	return int(t.Dims[0]), nil
}

func (t *Tensor) cols() (int, error) {
	if len(t.Dims) != 2 {
		return 0, fmt.Errorf("onnxinfer: expected a 2D tensor, got dims %v", t.Dims)
	}
	return int(t.Dims[1]), nil
}

// Model is a loaded ONNX model ready to Run.
type Model struct {
	proto        *onnxpb.ModelProto
	initializers map[string]*Tensor
}

// Load parses an ONNX model from its serialized protobuf bytes (as
// written by pkg/onnxexport, then proto.Marshal'd to a .onnx file) and
// extracts its float32 initializers.
func Load(data []byte) (*Model, error) {
	mp := &onnxpb.ModelProto{}
	if err := proto.Unmarshal(data, mp); err != nil {
		return nil, fmt.Errorf("onnxinfer: unmarshaling model: %w", err)
	}
	graph := mp.GetGraph()
	if graph == nil {
		return nil, fmt.Errorf("onnxinfer: model has no graph")
	}

	initializers := make(map[string]*Tensor, len(graph.GetInitializer()))
	for _, t := range graph.GetInitializer() {
		if t.GetDataType() != int32(onnxpb.TensorProto_FLOAT) {
			return nil, fmt.Errorf("onnxinfer: initializer %q: unsupported data type %d (only FLOAT is supported)", t.GetName(), t.GetDataType())
		}
		initializers[t.GetName()] = &Tensor{
			Dims: append([]int64(nil), t.GetDims()...),
			Data: append([]float32(nil), t.GetFloatData()...),
		}
	}

	return &Model{proto: mp, initializers: initializers}, nil
}

// Run executes the model's graph against the given named inputs, in the
// node order the graph already stores them (pkg/onnxexport always
// emits topologically-sorted nodes — this package trusts that rather
// than re-deriving it), returning every named graph output.
func (m *Model) Run(inputs map[string]*Tensor) (map[string]*Tensor, error) {
	graph := m.proto.GetGraph()

	values := make(map[string]*Tensor, len(m.initializers)+len(inputs))
	for name, t := range m.initializers {
		values[name] = t
	}
	for name, t := range inputs {
		values[name] = t
	}

	for _, node := range graph.GetNode() {
		outs, err := evalNode(node, values)
		if err != nil {
			return nil, fmt.Errorf("onnxinfer: node %q (%s): %w", node.GetName(), node.GetOpType(), err)
		}
		if len(outs) != len(node.GetOutput()) {
			return nil, fmt.Errorf("onnxinfer: node %q (%s): produced %d outputs, graph declares %d", node.GetName(), node.GetOpType(), len(outs), len(node.GetOutput()))
		}
		for i, name := range node.GetOutput() {
			values[name] = outs[i]
		}
	}

	result := make(map[string]*Tensor, len(graph.GetOutput()))
	for _, out := range graph.GetOutput() {
		t, ok := values[out.GetName()]
		if !ok {
			return nil, fmt.Errorf("onnxinfer: graph output %q was never produced", out.GetName())
		}
		result[out.GetName()] = t
	}
	return result, nil
}

// evalNode dispatches on the node's op type. Unrecognized op types are a
// hard error, not a fallback — see the package doc comment.
func evalNode(node *onnxpb.NodeProto, values map[string]*Tensor) ([]*Tensor, error) {
	in := func(i int) (*Tensor, error) {
		if i >= len(node.GetInput()) {
			return nil, fmt.Errorf("expected at least %d input(s), node declares %d", i+1, len(node.GetInput()))
		}
		name := node.GetInput()[i]
		t, ok := values[name]
		if !ok {
			return nil, fmt.Errorf("input %q was never produced", name)
		}
		return t, nil
	}

	switch node.GetOpType() {
	case "Gemm":
		a, err := in(0)
		if err != nil {
			return nil, err
		}
		b, err := in(1)
		if err != nil {
			return nil, err
		}
		// C (the bias operand) is optional per the ONNX spec — this
		// project's LSTM decomposition (pkg/onnxexport's recurrent
		// export) relies on that to compute Wx*a and Wh*hPrev as two
		// separate bias-free Gemms summed by a following Add, with the
		// gate's actual bias folded into only one of them.
		var c *Tensor
		if len(node.GetInput()) > 2 {
			c, err = in(2)
			if err != nil {
				return nil, err
			}
		}
		out, err := evalGemm(node, a, b, c)
		if err != nil {
			return nil, err
		}
		return []*Tensor{out}, nil

	case "Relu":
		a, err := in(0)
		if err != nil {
			return nil, err
		}
		return []*Tensor{evalRelu(a)}, nil

	case "Softmax":
		a, err := in(0)
		if err != nil {
			return nil, err
		}
		out, err := evalSoftmax(node, a)
		if err != nil {
			return nil, err
		}
		return []*Tensor{out}, nil

	case "Add":
		a, err := in(0)
		if err != nil {
			return nil, err
		}
		b, err := in(1)
		if err != nil {
			return nil, err
		}
		out, err := evalElementwise(a, b, func(x, y float32) float32 { return x + y })
		if err != nil {
			return nil, fmt.Errorf("Add: %w", err)
		}
		return []*Tensor{out}, nil

	case "Mul":
		a, err := in(0)
		if err != nil {
			return nil, err
		}
		b, err := in(1)
		if err != nil {
			return nil, err
		}
		out, err := evalElementwise(a, b, func(x, y float32) float32 { return x * y })
		if err != nil {
			return nil, fmt.Errorf("Mul: %w", err)
		}
		return []*Tensor{out}, nil

	case "Sigmoid":
		a, err := in(0)
		if err != nil {
			return nil, err
		}
		return []*Tensor{evalUnary(a, sigmoidf)}, nil

	case "Tanh":
		a, err := in(0)
		if err != nil {
			return nil, err
		}
		return []*Tensor{evalUnary(a, tanhf)}, nil

	default:
		return nil, fmt.Errorf("unsupported op %q — onnxinfer only executes the fixed op set pkg/onnxexport emits, not arbitrary ONNX models (see package doc comment)", node.GetOpType())
	}
}
