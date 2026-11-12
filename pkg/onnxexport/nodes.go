package onnxexport

import (
	"github.com/reallyoldfogie/cRL-go/pkg/mat"
	onnxpb "github.com/reallyoldfogie/cRL-go/pkg/onnx"
	"google.golang.org/protobuf/proto"
)

// matrixTensor builds a float32 TensorProto initializer from a cRL-go
// mat.Matrix: both are dense, row-major [Rows, Cols] — the flattened Data
// slice needs no reordering to become a valid TensorProto.
func matrixTensor(name string, m *mat.Matrix) *onnxpb.TensorProto {
	return &onnxpb.TensorProto{
		Name:      proto.String(name),
		Dims:      []int64{int64(m.Rows), int64(m.Cols)},
		DataType:  proto.Int32(int32(onnxpb.TensorProto_FLOAT)),
		FloatData: append([]float32(nil), m.Data...),
	}
}

// biasTensor builds a 1D float32 TensorProto initializer from a cRL-go
// bias mat.Matrix, which is always stored [N, 1] (cRL-go's column-vector
// convention: z = W@x + B). ONNX Gemm's bias operand instead broadcasts
// a [N] (or [1,N]) row across the result, so this flattens to a bare
// [N] shape rather than reusing matrixTensor's [Rows, Cols] — the
// underlying Data is already exactly N elements in the right order
// either way; only the declared shape differs.
func biasTensor(name string, m *mat.Matrix) *onnxpb.TensorProto {
	return &onnxpb.TensorProto{
		Name:      proto.String(name),
		Dims:      []int64{int64(m.Rows)},
		DataType:  proto.Int32(int32(onnxpb.TensorProto_FLOAT)),
		FloatData: append([]float32(nil), m.Data...),
	}
}

// float32Input builds a fixed-shape [1, width] float32 graph input/output
// descriptor — every tensor this exporter's graphs carry is a single
// (batch=1) row vector.
func float32Input(name string, width int) *onnxpb.ValueInfoProto {
	return &onnxpb.ValueInfoProto{
		Name: proto.String(name),
		Type: &onnxpb.TypeProto{
			Value: &onnxpb.TypeProto_TensorType{
				TensorType: &onnxpb.TypeProto_Tensor{
					ElemType: proto.Int32(int32(onnxpb.TensorProto_FLOAT)),
					Shape: &onnxpb.TensorShapeProto{
						Dim: []*onnxpb.TensorShapeProto_Dimension{
							{Value: &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: 1}},
							{Value: &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: int64(width)}},
						},
					},
				},
			},
		},
	}
}

// gemmNode builds a Gemm node computing input @ weight^T + bias (transB=1
// — see ExportFeedforward's doc comment for why transB, not a transposed
// weight tensor, reconciles cRL-go's column-vector convention with ONNX's
// row-vector one).
func gemmNode(name, input, weight, bias, output string) *onnxpb.NodeProto {
	return &onnxpb.NodeProto{
		Name:   proto.String(name),
		OpType: proto.String("Gemm"),
		Input:  []string{input, weight, bias},
		Output: []string{output},
		Attribute: []*onnxpb.AttributeProto{
			intAttr("transB", 1),
		},
	}
}

func reluNode(name, input, output string) *onnxpb.NodeProto {
	return &onnxpb.NodeProto{
		Name:   proto.String(name),
		OpType: proto.String("Relu"),
		Input:  []string{input},
		Output: []string{output},
	}
}

// softmaxNode builds a Softmax node over the last axis — the only axis
// this exporter's [1, N] row-vector tensors ever have more than one
// element along.
func softmaxNode(name, input, output string) *onnxpb.NodeProto {
	return &onnxpb.NodeProto{
		Name:   proto.String(name),
		OpType: proto.String("Softmax"),
		Input:  []string{input},
		Output: []string{output},
		Attribute: []*onnxpb.AttributeProto{
			intAttr("axis", -1),
		},
	}
}

func intAttr(name string, v int64) *onnxpb.AttributeProto {
	return &onnxpb.AttributeProto{
		Name: proto.String(name),
		Type: onnxpb.AttributeProto_INT.Enum(),
		I:    proto.Int64(v),
	}
}

func stringEntry(key, value string) *onnxpb.StringStringEntryProto {
	return &onnxpb.StringStringEntryProto{
		Key:   proto.String(key),
		Value: proto.String(value),
	}
}
