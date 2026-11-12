// Package onnx holds the pure-Go ONNX IR types generated (via protoc +
// protoc-gen-go — see README.md) from the official onnx/onnx schema, plus
// (eventually, in later phases of docs/plans/23-ONNX-support.md) the
// exporter and interpreter built on top of them.
//
// This file is Phase 1's feasibility proof: hand-build the smallest
// interesting model (one Gemm feeding one Relu, with real initializers and
// attributes), marshal it, and unmarshal it back, before any real cRL-go
// weights are involved. It also writes the model to testdata/ so it can be
// opened in Netron by hand — Netron needs nothing from this package, only a
// well-formed .onnx file.
package onnx

import (
	"os"
	"path/filepath"
	"testing"

	"google.golang.org/protobuf/proto"
)

// opsetVersion is the ai.onnx (default-domain) opset this package targets.
// 21 is used rather than this vendored schema's own ceiling (28, as of the
// pinned onnx v1.23.2 release — see README.md) because every op this
// project needs (Gemm, Relu, Softmax, Sigmoid, Tanh, Add, Mul) has had
// stable semantics since long before opset 21, and staying well under the
// ceiling maximizes compatibility with whatever reads these files (Netron,
// onnxruntime, or anything else).
const opsetVersion = 21

// irVersion is onnx.proto's Version_IR_VERSION constant for the pinned
// v1.23.2 schema (see README.md) — the IR version every model this package
// produces declares itself as targeting.
const irVersion = int64(Version_IR_VERSION)

// floatTensor builds a TensorProto initializer: a named, float32 tensor
// with the given shape and flattened row-major data.
func floatTensor(name string, dims []int64, data []float32) *TensorProto {
	return &TensorProto{
		Name:      proto.String(name),
		Dims:      dims,
		DataType:  proto.Int32(int32(TensorProto_FLOAT)),
		FloatData: data,
	}
}

// float32ValueInfo builds a graph input/output descriptor for a float32
// tensor of the given fixed shape (every dimension is a concrete
// dim_value, never a symbolic dim_param — this package never produces
// dynamic shapes).
func float32ValueInfo(name string, dims []int64) *ValueInfoProto {
	shapeDims := make([]*TensorShapeProto_Dimension, len(dims))
	for i, d := range dims {
		shapeDims[i] = &TensorShapeProto_Dimension{
			Value: &TensorShapeProto_Dimension_DimValue{DimValue: d},
		}
	}
	return &ValueInfoProto{
		Name: proto.String(name),
		Type: &TypeProto{
			Value: &TypeProto_TensorType{
				TensorType: &TypeProto_Tensor{
					ElemType: proto.Int32(int32(TensorProto_FLOAT)),
					Shape:    &TensorShapeProto{Dim: shapeDims},
				},
			},
		},
	}
}

// intAttribute builds an INT-typed NodeProto attribute (e.g. Gemm's
// transB).
func intAttribute(name string, v int64) *AttributeProto {
	return &AttributeProto{
		Name: proto.String(name),
		Type: AttributeProto_INT.Enum(),
		I:    proto.Int64(v),
	}
}

// buildTrivialGemmReluModel hand-builds the smallest model worth proving
// the pipeline on: a single Linear-layer-shaped Gemm (x [1,2] @ W^T [2,3] +
// B [3] -> [1,3], using transB so W is stored [out_features, in_features]
// the way a real exported Linear layer's weight matrix would be) followed
// by Relu. This exercises initializers, a non-default node attribute,
// value-shape plumbing, and node chaining — everything Phase 2's real
// exporter needs, without touching any real cRL-go weights yet.
func buildTrivialGemmReluModel() *ModelProto {
	w := floatTensor("W", []int64{3, 2}, []float32{
		1, 0,
		0, 1,
		1, 1,
	})
	b := floatTensor("B", []int64{3}, []float32{0, 0, 0})

	gemm := &NodeProto{
		OpType:    proto.String("Gemm"),
		Input:     []string{"x", "W", "B"},
		Output:    []string{"y1"},
		Attribute: []*AttributeProto{intAttribute("transB", 1)},
	}
	relu := &NodeProto{
		OpType: proto.String("Relu"),
		Input:  []string{"y1"},
		Output: []string{"y"},
	}

	graph := &GraphProto{
		Name:        proto.String("trivial_gemm_relu"),
		Node:        []*NodeProto{gemm, relu},
		Initializer: []*TensorProto{w, b},
		Input:       []*ValueInfoProto{float32ValueInfo("x", []int64{1, 2})},
		Output:      []*ValueInfoProto{float32ValueInfo("y", []int64{1, 3})},
	}

	return &ModelProto{
		IrVersion:       proto.Int64(irVersion),
		ProducerName:    proto.String("cRL-go/pkg/onnx (phase 1 smoke test)"),
		ProducerVersion: proto.String("0.0.0-phase1"),
		OpsetImport: []*OperatorSetIdProto{
			{Domain: proto.String(""), Version: proto.Int64(opsetVersion)},
		},
		Graph: graph,
		DocString: proto.String(
			"Phase 1 feasibility fixture for docs/plans/23-ONNX-support.md: " +
				"not a trained model, just x -> Gemm(transB) -> Relu -> y.",
		),
	}
}

// TestTrivialGemmReluModelRoundTrip builds the fixture above, marshals it,
// writes it to testdata/ (so it can be opened in Netron by hand — see
// README.md), and unmarshals it back into a fresh ModelProto, asserting
// the result matches what was built byte-for-byte on every field this
// package actually cares about.
func TestTrivialGemmReluModelRoundTrip(t *testing.T) {
	want := buildTrivialGemmReluModel()

	data, err := proto.Marshal(want)
	if err != nil {
		t.Fatalf("proto.Marshal: %v", err)
	}

	if err := os.MkdirAll("testdata", 0o755); err != nil {
		t.Fatalf("creating testdata dir: %v", err)
	}
	fixturePath := filepath.Join("testdata", "trivial_gemm_relu.onnx")
	if err := os.WriteFile(fixturePath, data, 0o644); err != nil {
		t.Fatalf("writing %s: %v", fixturePath, err)
	}

	got := &ModelProto{}
	if err := proto.Unmarshal(data, got); err != nil {
		t.Fatalf("proto.Unmarshal: %v", err)
	}

	if got.GetIrVersion() != want.GetIrVersion() {
		t.Errorf("IrVersion = %d, want %d", got.GetIrVersion(), want.GetIrVersion())
	}
	if len(got.GetOpsetImport()) != 1 || got.GetOpsetImport()[0].GetVersion() != opsetVersion {
		t.Errorf("OpsetImport = %v, want [{domain:\"\" version:%d}]", got.GetOpsetImport(), opsetVersion)
	}

	gotGraph, wantGraph := got.GetGraph(), want.GetGraph()
	if gotGraph == nil {
		t.Fatal("round-tripped model has no graph")
	}

	if len(gotGraph.GetNode()) != len(wantGraph.GetNode()) {
		t.Fatalf("got %d nodes, want %d", len(gotGraph.GetNode()), len(wantGraph.GetNode()))
	}
	for i, wantNode := range wantGraph.GetNode() {
		gotNode := gotGraph.GetNode()[i]
		if gotNode.GetOpType() != wantNode.GetOpType() {
			t.Errorf("node[%d].OpType = %q, want %q", i, gotNode.GetOpType(), wantNode.GetOpType())
		}
		if len(gotNode.GetInput()) != len(wantNode.GetInput()) {
			t.Errorf("node[%d].Input = %v, want %v", i, gotNode.GetInput(), wantNode.GetInput())
		}
	}

	if len(gotGraph.GetInitializer()) != 2 {
		t.Fatalf("got %d initializers, want 2", len(gotGraph.GetInitializer()))
	}
	gotW := gotGraph.GetInitializer()[0]
	wantW := wantGraph.GetInitializer()[0]
	if gotW.GetName() != "W" {
		t.Fatalf("initializer[0].Name = %q, want \"W\"", gotW.GetName())
	}
	if diff := len(gotW.GetFloatData()) - len(wantW.GetFloatData()); diff != 0 {
		t.Fatalf("initializer W has %d float values, want %d", len(gotW.GetFloatData()), len(wantW.GetFloatData()))
	}
	for i, v := range wantW.GetFloatData() {
		if gotW.GetFloatData()[i] != v {
			t.Errorf("initializer W[%d] = %v, want %v", i, gotW.GetFloatData()[i], v)
		}
	}

	// transB must survive the round trip, or the Gemm node silently means
	// something else on load than what was exported.
	gemmNode := gotGraph.GetNode()[0]
	if len(gemmNode.GetAttribute()) != 1 {
		t.Fatalf("Gemm node has %d attributes, want 1 (transB)", len(gemmNode.GetAttribute()))
	}
	transB := gemmNode.GetAttribute()[0]
	if transB.GetName() != "transB" || transB.GetI() != 1 {
		t.Errorf("Gemm attribute = {name:%q i:%d}, want {name:\"transB\" i:1}", transB.GetName(), transB.GetI())
	}

	t.Logf("wrote %s (%d bytes) — open it in Netron to confirm it renders: Gemm -> Relu, x[1,2] -> y[1,3]", fixturePath, len(data))
}
