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

// TestExportRecurrentMatchesInferenceNetwork is Phase 3's
// numeric-equivalence test: build a real actorcritic.RecurrentParams,
// drive several steps of a random observation sequence through cRL-go's
// own actorcritic.NewRecurrentInferenceNetwork (threading HPrevIn/CPrevIn/
// ValueHPrevIn/ValueCPrevIn between steps exactly as
// RecurrentActor.Act does), and compare against the same Params
// exported to ONNX and run through onnxinfer, threading its own four
// state tensors between steps the same way. State threading, not just
// single-step correctness, is the point: a bug in how hOut/cOut feed
// back into the next step's h_in/c_in would only show up after step 0.
func TestExportRecurrentMatchesInferenceNetwork(t *testing.T) {
	const (
		inputSize       = 5
		hiddenSize      = 4
		outputSize      = 3
		numHiddenLayers = 2
		numSteps        = 5
		tolerance       = 1e-4
	)

	rng := rand.New(rand.NewPCG(7, 11))
	params := actorcritic.NewRecurrentParams(rng, inputSize, hiddenSize, outputSize, numHiddenLayers)

	net, err := actorcritic.NewRecurrentInferenceNetwork(params)
	if err != nil {
		t.Fatalf("NewRecurrentInferenceNetwork: %v", err)
	}

	model, err := ExportRecurrent(params, "test-env:recurrent-v1")
	if err != nil {
		t.Fatalf("ExportRecurrent: %v", err)
	}
	data, err := proto.Marshal(model)
	if err != nil {
		t.Fatalf("proto.Marshal: %v", err)
	}

	if err := os.MkdirAll("testdata", 0o755); err != nil {
		t.Fatalf("creating testdata dir: %v", err)
	}
	fixturePath := filepath.Join("testdata", "actorcritic_recurrent.onnx")
	if err := os.WriteFile(fixturePath, data, 0o644); err != nil {
		t.Fatalf("writing %s: %v", fixturePath, err)
	}
	t.Logf("wrote %s (%d bytes) — open it in Netron to confirm it renders two independent LSTM branches sharing one observation input", fixturePath, len(data))

	loaded, err := onnxinfer.Load(data)
	if err != nil {
		t.Fatalf("onnxinfer.Load: %v", err)
	}

	zeroState := func() *onnxinfer.Tensor {
		return &onnxinfer.Tensor{Dims: []int64{1, hiddenSize}, Data: make([]float32, hiddenSize)}
	}
	policyH, policyC := zeroState(), zeroState()
	valueH, valueC := zeroState(), zeroState()

	for step := 0; step < numSteps; step++ {
		obs := make([]float32, inputSize)
		for i := range obs {
			obs[i] = rng.Float32()*4 - 2
		}

		// Reference: cRL-go's own recurrent forward pass, carrying
		// (h, c) across steps via net's own Prev/Out fields.
		copy(net.Input.Val.Data, obs)
		net.Graph.Forward()
		wantPolicy := append([]float32(nil), net.PolicyOutput.Val.Data...)
		wantValue := net.ValueOutput.Val.Data[0]
		copy(net.HPrevIn.Val.Data, net.HOut.Val.Data)
		copy(net.CPrevIn.Val.Data, net.COut.Val.Data)
		copy(net.ValueHPrevIn.Val.Data, net.ValueHOut.Val.Data)
		copy(net.ValueCPrevIn.Val.Data, net.ValueCOut.Val.Data)

		// Candidate: the exported model, run through onnxinfer, carrying
		// its own four state tensors across steps the same way.
		outs, err := loaded.Run(map[string]*onnxinfer.Tensor{
			ObservationInputName:       {Dims: []int64{1, inputSize}, Data: obs},
			PolicyHiddenStateInputName: policyH,
			PolicyCellStateInputName:   policyC,
			ValueHiddenStateInputName:  valueH,
			ValueCellStateInputName:    valueC,
		})
		if err != nil {
			t.Fatalf("step %d: Run: %v", step, err)
		}

		gotPolicy := outs[PolicyOutputName].Data
		if len(gotPolicy) != len(wantPolicy) {
			t.Fatalf("step %d: policy output has %d values, want %d", step, len(gotPolicy), len(wantPolicy))
		}
		for i := range wantPolicy {
			if diff := abs32(gotPolicy[i] - wantPolicy[i]); diff > tolerance {
				t.Errorf("step %d: policy[%d] = %v, want %v (diff %v > tolerance %v)", step, i, gotPolicy[i], wantPolicy[i], diff, tolerance)
			}
		}

		gotValue := outs[ValueOutputName].Data[0]
		if diff := abs32(gotValue - wantValue); diff > tolerance {
			t.Errorf("step %d: value = %v, want %v (diff %v > tolerance %v)", step, gotValue, wantValue, diff, tolerance)
		}

		policyH, policyC = outs[PolicyHiddenStateOutputName], outs[PolicyCellStateOutputName]
		valueH, valueC = outs[ValueHiddenStateOutputName], outs[ValueCellStateOutputName]
	}
}
