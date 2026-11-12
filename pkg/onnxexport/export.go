// Package onnxexport converts a trained cRL-go/pkg/actorcritic model into
// an ONNX ModelProto (pkg/onnx's generated IR types). It is a hand-written
// exporter, not a generic graph walker: actorcritic.Params/RecurrentParams
// are always one of exactly two known shapes (see
// docs/plans/23-ONNX-support.md §2), so every node this package emits is
// explicit rather than discovered.
//
// This file covers the feedforward case (Params) — Phase 2 of that plan.
// The recurrent case (RecurrentParams, with its independent policy/value
// LSTM branches) is Phase 3.
package onnxexport

import (
	"fmt"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	onnxpb "github.com/reallyoldfogie/cRL-go/pkg/onnx"
	"google.golang.org/protobuf/proto"
)

// Graph input/output names, exported so a caller (a numeric-equivalence
// test today; the standalone inference binary's manifest in Phase 4)
// never needs to hardcode its own copy of these strings.
const (
	ObservationInputName = "observation"
	PolicyOutputName     = "policy"
	ValueOutputName      = "value"
)

// hiddenLayerOutputName returns the name of hidden layer i's post-ReLU
// output — "observation" for i == -1 (the trunk's own input), so the
// first layer's Gemm can use the same naming logic as every later one.
func hiddenLayerOutputName(i int) string {
	if i < 0 {
		return ObservationInputName
	}
	return fmt.Sprintf("hidden%d.out", i)
}

// ExportFeedforward builds an ONNX ModelProto computing exactly what
// actorcritic.NewInferenceNetwork(params) computes for a single
// observation: the trunk (Hidden[0..n-1], each Gemm+Relu) feeding a
// softmax policy head and a scalar value head.
//
// cRL-go's own convention is column-vector (Layer.W is [out,in],
// multiplied against an [in,1] input), while ONNX Gemm's idiomatic,
// Netron-friendly convention is row-vector-batch ([1,in] x [in,out]).
// Rather than transposing any weight data, every Gemm node here sets
// transB=1 and uses the matrix exactly as Params already stores it
// ([out,in]) as the B operand — y = x @ W^T + b with x a [1,in] row and W
// stored [out,in] is numerically identical to cRL-go's z = W @ x with x
// an [in,1] column; only the shape convention (row vs. column) differs,
// which transB reconciles for free.
//
// envID is recorded in the model's metadata (not read back by anything
// yet — Phase 4's manifest will carry the real schema-compatibility
// check); it exists now so every exported model is traceable to the
// environment it was trained against from the moment export exists.
func ExportFeedforward(params *actorcritic.Params, envID string) (*onnxpb.ModelProto, error) {
	if params == nil {
		return nil, fmt.Errorf("onnxexport: params must not be nil")
	}

	var nodes []*onnxpb.NodeProto
	var initializers []*onnxpb.TensorProto

	prev := hiddenLayerOutputName(-1)
	for i, layer := range params.Hidden {
		wName, bName := fmt.Sprintf("hidden%d.w", i), fmt.Sprintf("hidden%d.b", i)
		preName, outName := fmt.Sprintf("hidden%d.pre", i), hiddenLayerOutputName(i)

		initializers = append(initializers,
			matrixTensor(wName, layer.W),
			biasTensor(bName, layer.B),
		)
		nodes = append(nodes,
			gemmNode(fmt.Sprintf("hidden%d.gemm", i), prev, wName, bName, preName),
			reluNode(fmt.Sprintf("hidden%d.relu", i), preName, outName),
		)
		prev = outName
	}

	initializers = append(initializers,
		matrixTensor("policy.w", params.Wpi),
		biasTensor("policy.b", params.Bpi),
		matrixTensor("value.w", params.Wv),
		biasTensor("value.b", params.Bv),
	)
	nodes = append(nodes,
		gemmNode("policy.gemm", prev, "policy.w", "policy.b", "policy.logits"),
		softmaxNode("policy.softmax", "policy.logits", PolicyOutputName),
		gemmNode("value.gemm", prev, "value.w", "value.b", ValueOutputName),
	)

	graph := &onnxpb.GraphProto{
		Name:        proto.String("crl_go_actorcritic_feedforward"),
		Node:        nodes,
		Initializer: initializers,
		Input: []*onnxpb.ValueInfoProto{
			float32Input(ObservationInputName, params.InputSize()),
		},
		Output: []*onnxpb.ValueInfoProto{
			float32Input(PolicyOutputName, params.OutputSize()),
			float32Input(ValueOutputName, 1),
		},
	}

	return &onnxpb.ModelProto{
		IrVersion:       proto.Int64(onnxpb.IRVersion),
		ProducerName:    proto.String("cRL-go/pkg/onnxexport"),
		ProducerVersion: proto.String("0.0.0-phase2"),
		Domain:          proto.String("cRL-go"),
		OpsetImport: []*onnxpb.OperatorSetIdProto{
			{Domain: proto.String(""), Version: proto.Int64(onnxpb.OpsetVersion)},
		},
		Graph: graph,
		MetadataProps: []*onnxpb.StringStringEntryProto{
			stringEntry("source_framework", "cRL-go/pkg/actorcritic"),
			stringEntry("architecture", "actorcritic.Params (feedforward)"),
			stringEntry("environment_id", envID),
			stringEntry("input_size", fmt.Sprint(params.InputSize())),
			stringEntry("hidden_size", fmt.Sprint(params.HiddenSize())),
			stringEntry("num_hidden_layers", fmt.Sprint(params.NumHiddenLayers())),
			stringEntry("output_size", fmt.Sprint(params.OutputSize())),
		},
	}, nil
}
