package onnxexport

import (
	"fmt"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/mat"
	onnxpb "github.com/reallyoldfogie/cRL-go/pkg/onnx"
	"google.golang.org/protobuf/proto"
)

// policyPrefix/valuePrefix name every node and tensor belonging to one
// of RecurrentParams' two independent branches (see
// docs/plans/23-ONNX-support.md §2: PolicyTrunk+PolicyLSTM and
// ValueTrunk+ValueLSTM are separate, not a shared backbone with two
// heads). The *_h_in/*_c_in/*_h_out/*_c_out constants below are built
// from these same prefixes, so a graph's actual node output names and
// the names this package exports for external callers (the
// numeric-equivalence test today; Phase 4's manifest) can never drift
// apart.
const (
	policyPrefix = "policy_"
	valuePrefix  = "value_"

	// PolicyHiddenStateInputName/PolicyCellStateInputName and their
	// Value* counterparts are the graph's four recurrent-state inputs —
	// zeroed by the caller at episode start, then threaded from the
	// previous step's *_h_out/*_c_out, exactly as
	// actorcritic.RecurrentActor.Act threads
	// HPrevIn/CPrevIn/ValueHPrevIn/ValueCPrevIn between calls.
	PolicyHiddenStateInputName = policyPrefix + "h_in"
	PolicyCellStateInputName   = policyPrefix + "c_in"
	ValueHiddenStateInputName  = valuePrefix + "h_in"
	ValueCellStateInputName    = valuePrefix + "c_in"

	PolicyHiddenStateOutputName = policyPrefix + "h_out"
	PolicyCellStateOutputName   = policyPrefix + "c_out"
	ValueHiddenStateOutputName  = valuePrefix + "h_out"
	ValueCellStateOutputName    = valuePrefix + "c_out"
)

// recurrentPolicyTrunk, recurrentValueTrunk, recurrentPolicyLSTM, and
// recurrentValueLSTM re-derive actorcritic.RecurrentParams' own
// (unexported) PolicyTrunk/ValueTrunk/PolicyLSTM/ValueLSTM-or-Trunk/LSTM
// fallback from its exported fields, since this package can't call
// those unexported methods directly. See RecurrentParams' doc comment:
// a value built before the policy/value branches were split only sets
// Trunk/LSTM, leaving the split fields nil.
func recurrentPolicyTrunk(p *actorcritic.RecurrentParams) *actorcritic.Params {
	if p.PolicyTrunk != nil {
		return p.PolicyTrunk
	}
	return p.Trunk
}

func recurrentValueTrunk(p *actorcritic.RecurrentParams) *actorcritic.Params {
	if p.ValueTrunk != nil {
		return p.ValueTrunk
	}
	return p.Trunk
}

func recurrentPolicyLSTM(p *actorcritic.RecurrentParams) *actorcritic.LSTMLayer {
	if p.PolicyLSTM != nil {
		return p.PolicyLSTM
	}
	return p.LSTM
}

func recurrentValueLSTM(p *actorcritic.RecurrentParams) *actorcritic.LSTMLayer {
	if p.ValueLSTM != nil {
		return p.ValueLSTM
	}
	return p.LSTM
}

// appendLSTMGate appends one gate's decomposed computation —
//
//	pre = Wx@a + Wh@hPrev + b
//	gate = activation(pre)
//
// — as two bias-free Gemms (Wx@a+b and Wh@hPrev) summed by an Add, then
// one activation node, rather than ONNX's native fused LSTM operator.
// See docs/plans/23-ONNX-support.md §4.2: this sidesteps remapping
// cRL-go's split F/I/O/G gate matrices into the native op's fused,
// differently-ordered (i,o,f,c) tensor layout, at the cost of more
// nodes — a deliberate trade given this is hand-decomposed anyway, not
// discovered from a real LSTM-shaped graph.
func appendLSTMGate(nodes []*onnxpb.NodeProto, initializers []*onnxpb.TensorProto, prefix, gate string, wx, wh, b *mat.Matrix, aName, hPrevName string, activation func(name, input, output string) *onnxpb.NodeProto) (outNodes []*onnxpb.NodeProto, outInitializers []*onnxpb.TensorProto, outName string) {
	wxName := fmt.Sprintf("%s%s.wx", prefix, gate)
	whName := fmt.Sprintf("%s%s.wh", prefix, gate)
	bName := fmt.Sprintf("%s%s.b", prefix, gate)
	wxOutName := fmt.Sprintf("%s%s.wx_out", prefix, gate)
	whOutName := fmt.Sprintf("%s%s.wh_out", prefix, gate)
	preName := fmt.Sprintf("%s%s.pre", prefix, gate)
	outName = fmt.Sprintf("%s%s", prefix, gate)

	initializers = append(initializers,
		matrixTensor(wxName, wx),
		matrixTensor(whName, wh),
		biasTensor(bName, b),
	)
	nodes = append(nodes,
		gemmNode(fmt.Sprintf("%s%s.wx_gemm", prefix, gate), aName, wxName, bName, wxOutName),
		gemmNodeNoBias(fmt.Sprintf("%s%s.wh_gemm", prefix, gate), hPrevName, whName, whOutName),
		addNode(fmt.Sprintf("%s%s.add", prefix, gate), wxOutName, whOutName, preName),
		activation(fmt.Sprintf("%s%s.activation", prefix, gate), preName, outName),
	)
	return nodes, initializers, outName
}

// appendLSTMBranch appends one branch's full computation — trunk
// (appendTrunkHidden), LSTM cell (four appendLSTMGate calls plus the
// cOut/hOut combination), and that branch's output head — returning the
// extended nodes/initializers plus the branch's hOut/cOut names (which
// double as graph outputs: see ExportRecurrent).
//
//	f = sigmoid(gate); i = sigmoid(gate); o = sigmoid(gate); g = tanh(gate)
//	cOut = f*cPrev + i*g
//	hOut = o*tanh(cOut)
//
// matching actorcritic.buildRecurrentForward exactly (see recurrent_network.go).
func appendLSTMBranch(nodes []*onnxpb.NodeProto, initializers []*onnxpb.TensorProto, prefix string, trunk *actorcritic.Params, lstm *actorcritic.LSTMLayer, hPrevName, cPrevName string) (outNodes []*onnxpb.NodeProto, outInitializers []*onnxpb.TensorProto, hOutName, cOutName string) {
	nodes, initializers, trunkOut := appendTrunkHidden(nodes, initializers, prefix, ObservationInputName, trunk.Hidden)

	nodes, initializers, fOut := appendLSTMGate(nodes, initializers, prefix, "f", lstm.WxF, lstm.WhF, lstm.BF, trunkOut, hPrevName, sigmoidNode)
	nodes, initializers, iOut := appendLSTMGate(nodes, initializers, prefix, "i", lstm.WxI, lstm.WhI, lstm.BI, trunkOut, hPrevName, sigmoidNode)
	nodes, initializers, oOut := appendLSTMGate(nodes, initializers, prefix, "o", lstm.WxO, lstm.WhO, lstm.BO, trunkOut, hPrevName, sigmoidNode)
	nodes, initializers, gOut := appendLSTMGate(nodes, initializers, prefix, "g", lstm.WxG, lstm.WhG, lstm.BG, trunkOut, hPrevName, tanhNode)

	fc := fmt.Sprintf("%sf_c", prefix)
	ig := fmt.Sprintf("%si_g", prefix)
	cOutName = fmt.Sprintf("%sc_out", prefix)
	tanhCOut := fmt.Sprintf("%stanh_c_out", prefix)
	hOutName = fmt.Sprintf("%sh_out", prefix)

	nodes = append(nodes,
		mulNode(fmt.Sprintf("%sf_c.mul", prefix), fOut, cPrevName, fc),
		mulNode(fmt.Sprintf("%si_g.mul", prefix), iOut, gOut, ig),
		addNode(fmt.Sprintf("%sc_out.add", prefix), fc, ig, cOutName),
		tanhNode(fmt.Sprintf("%stanh_c_out.tanh", prefix), cOutName, tanhCOut),
		mulNode(fmt.Sprintf("%sh_out.mul", prefix), oOut, tanhCOut, hOutName),
	)

	return nodes, initializers, hOutName, cOutName
}

// ExportRecurrent builds an ONNX ModelProto computing exactly what
// actorcritic.NewRecurrentInferenceNetwork(params) computes for a single
// step: two fully independent branches (policy: PolicyTrunk+PolicyLSTM
// feeding Wpi/Bpi+Softmax; value: ValueTrunk+ValueLSTM feeding Wv/Bv),
// sharing only the raw observation input — see
// docs/plans/23-ONNX-support.md §2. The caller owns recurrent state
// exactly as actorcritic.RecurrentActor does: feed zeros at episode
// start, then thread each step's four *_h_out/*_c_out outputs into the
// next step's four *_h_in/*_c_in inputs.
//
// envID is recorded in the model's metadata — see ExportFeedforward's
// doc comment for why.
func ExportRecurrent(params *actorcritic.RecurrentParams, envID string) (*onnxpb.ModelProto, error) {
	if params == nil {
		return nil, fmt.Errorf("onnxexport: params must not be nil")
	}

	policyTrunk := recurrentPolicyTrunk(params)
	valueTrunk := recurrentValueTrunk(params)
	policyLSTM := recurrentPolicyLSTM(params)
	valueLSTM := recurrentValueLSTM(params)

	var nodes []*onnxpb.NodeProto
	var initializers []*onnxpb.TensorProto

	nodes, initializers, policyHOut, policyCOut := appendLSTMBranch(nodes, initializers, policyPrefix, policyTrunk, policyLSTM, PolicyHiddenStateInputName, PolicyCellStateInputName)
	nodes, initializers, valueHOut, valueCOut := appendLSTMBranch(nodes, initializers, valuePrefix, valueTrunk, valueLSTM, ValueHiddenStateInputName, ValueCellStateInputName)

	initializers = append(initializers,
		matrixTensor("policy.w", policyTrunk.Wpi),
		biasTensor("policy.b", policyTrunk.Bpi),
		matrixTensor("value.w", valueTrunk.Wv),
		biasTensor("value.b", valueTrunk.Bv),
	)
	nodes = append(nodes,
		gemmNode("policy.gemm", policyHOut, "policy.w", "policy.b", "policy.logits"),
		softmaxNode("policy.softmax", "policy.logits", PolicyOutputName),
		gemmNode("value.gemm", valueHOut, "value.w", "value.b", ValueOutputName),
	)

	hiddenSize := params.HiddenSize()
	graph := &onnxpb.GraphProto{
		Name:        proto.String("crl_go_actorcritic_recurrent"),
		Node:        nodes,
		Initializer: initializers,
		Input: []*onnxpb.ValueInfoProto{
			float32Input(ObservationInputName, params.InputSize()),
			float32Input(PolicyHiddenStateInputName, hiddenSize),
			float32Input(PolicyCellStateInputName, hiddenSize),
			float32Input(ValueHiddenStateInputName, hiddenSize),
			float32Input(ValueCellStateInputName, hiddenSize),
		},
		Output: []*onnxpb.ValueInfoProto{
			float32Input(PolicyOutputName, params.OutputSize()),
			float32Input(ValueOutputName, 1),
			float32Input(policyHOut, hiddenSize),
			float32Input(policyCOut, hiddenSize),
			float32Input(valueHOut, hiddenSize),
			float32Input(valueCOut, hiddenSize),
		},
	}

	return &onnxpb.ModelProto{
		IrVersion:       proto.Int64(onnxpb.IRVersion),
		ProducerName:    proto.String("cRL-go/pkg/onnxexport"),
		ProducerVersion: proto.String("0.0.0-phase3"),
		Domain:          proto.String("cRL-go"),
		OpsetImport: []*onnxpb.OperatorSetIdProto{
			{Domain: proto.String(""), Version: proto.Int64(onnxpb.OpsetVersion)},
		},
		Graph: graph,
		MetadataProps: []*onnxpb.StringStringEntryProto{
			stringEntry("source_framework", "cRL-go/pkg/actorcritic"),
			stringEntry("architecture", "actorcritic.RecurrentParams (LSTM, decomposed)"),
			stringEntry("environment_id", envID),
			stringEntry("input_size", fmt.Sprint(params.InputSize())),
			stringEntry("hidden_size", fmt.Sprint(hiddenSize)),
			stringEntry("num_hidden_layers", fmt.Sprint(params.NumHiddenLayers())),
			stringEntry("output_size", fmt.Sprint(params.OutputSize())),
		},
	}, nil
}
