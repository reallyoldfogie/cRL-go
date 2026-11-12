package onnxexport

import (
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"google.golang.org/protobuf/proto"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/onnxinfer"
)

// TestExportFeedforwardMatchesInferenceNetwork is Phase 2's
// numeric-equivalence test (docs/plans/23-ONNX-support.md §5): build a
// real actorcritic.Params, run cRL-go's own forward pass
// (actorcritic.NewInferenceNetwork) over a random observation, then run
// the exact same Params exported to ONNX and reloaded through
// onnxinfer, and confirm both sides agree within a tight numerical
// tolerance — not a generic ONNX conformance test, but proof that this
// project's own exporter and interpreter faithfully replay cRL-go's own
// actorcritic computation.
func TestExportFeedforwardMatchesInferenceNetwork(t *testing.T) {
	const (
		inputSize       = 6
		hiddenSize      = 5
		outputSize      = 4
		numHiddenLayers = 3
		tolerance       = 1e-5
	)

	rng := rand.New(rand.NewPCG(1, 2))
	params := actorcritic.NewParams(rng, inputSize, hiddenSize, outputSize, numHiddenLayers)

	obs := make([]float32, inputSize)
	for i := range obs {
		obs[i] = rng.Float32()*4 - 2 // roughly [-2, 2), well outside [0,1) where every ReLU/Softmax edge case looks the same
	}

	// Reference: cRL-go's own forward pass.
	net, err := actorcritic.NewInferenceNetwork(params)
	if err != nil {
		t.Fatalf("NewInferenceNetwork: %v", err)
	}
	copy(net.Input.Val.Data, obs)
	net.Graph.Forward()
	wantPolicy := append([]float32(nil), net.PolicyOutput.Val.Data...)
	wantValue := net.ValueOutput.Val.Data[0]

	// Candidate: export to ONNX, reload, run through the pure-Go
	// interpreter.
	model, err := ExportFeedforward(params, "test-env:v1")
	if err != nil {
		t.Fatalf("ExportFeedforward: %v", err)
	}
	data, err := proto.Marshal(model)
	if err != nil {
		t.Fatalf("proto.Marshal: %v", err)
	}

	if err := os.MkdirAll("testdata", 0o755); err != nil {
		t.Fatalf("creating testdata dir: %v", err)
	}
	fixturePath := filepath.Join("testdata", "actorcritic_feedforward.onnx")
	if err := os.WriteFile(fixturePath, data, 0o644); err != nil {
		t.Fatalf("writing %s: %v", fixturePath, err)
	}
	t.Logf("wrote %s (%d bytes, %d hidden layers) — open it in Netron to confirm it renders the full trunk+heads", fixturePath, len(data), numHiddenLayers)

	loaded, err := onnxinfer.Load(data)
	if err != nil {
		t.Fatalf("onnxinfer.Load: %v", err)
	}
	outs, err := loaded.Run(map[string]*onnxinfer.Tensor{
		ObservationInputName: {Dims: []int64{1, int64(inputSize)}, Data: obs},
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	gotPolicyTensor, ok := outs[PolicyOutputName]
	if !ok {
		t.Fatalf("missing output %q; got %v", PolicyOutputName, keysOf(outs))
	}
	gotValueTensor, ok := outs[ValueOutputName]
	if !ok {
		t.Fatalf("missing output %q; got %v", ValueOutputName, keysOf(outs))
	}

	gotPolicy := gotPolicyTensor.Data
	if len(gotPolicy) != len(wantPolicy) {
		t.Fatalf("policy output has %d values, want %d", len(gotPolicy), len(wantPolicy))
	}
	for i := range wantPolicy {
		if diff := abs32(gotPolicy[i] - wantPolicy[i]); diff > tolerance {
			t.Errorf("policy[%d] = %v, want %v (diff %v > tolerance %v)", i, gotPolicy[i], wantPolicy[i], diff, tolerance)
		}
	}

	if len(gotValueTensor.Data) != 1 {
		t.Fatalf("value output has %d values, want 1", len(gotValueTensor.Data))
	}
	gotValue := gotValueTensor.Data[0]
	if diff := abs32(gotValue - wantValue); diff > tolerance {
		t.Errorf("value = %v, want %v (diff %v > tolerance %v)", gotValue, wantValue, diff, tolerance)
	}
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

func keysOf(m map[string]*onnxinfer.Tensor) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}
