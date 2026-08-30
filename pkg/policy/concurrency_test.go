package policy

import (
	"math/rand/v2"
	"sync"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// applyOneGradientStep runs one full forward/backward/gradient-step
// cycle against net, exercising the exact same Val-mutating code path
// TestApplyGradientStepUpdatesParameters (network_test.go) already
// checks for correctness; this helper exists purely to drive that cycle
// repeatedly in the tests below. The actual mutation (ApplyGradientStep)
// is guarded by params.Lock/Unlock, exactly as
// pkg/reinforce.Trainer.RunEpoch guards it — params must be the same
// Params net was built from.
func applyOneGradientStep(params *Params, net *TrainingNetwork, rng *rand.Rand, learningRate float32) {
	net.ZeroGrad()
	net.Input.Val.FillRand(rng, -1, 1)
	net.Advantage.Val.Clear()
	net.Advantage.Val.Data[0] = 1.0
	net.Graph.Forward()
	net.Graph.Backward()

	params.Lock()
	net.ApplyGradientStep(learningRate, 1)
	params.Unlock()
}

// TestSnapshotIsUnaffectedBySubsequentGradientStep proves Snapshot
// performs a deep copy, not an alias: a gradient step applied to the
// live Params after Snapshot was taken must not change the snapshot's
// weights.
func TestSnapshotIsUnaffectedBySubsequentGradientStep(t *testing.T) {
	rng := rand.New(rand.NewPCG(51, 52))
	params := NewParams(rng, 6, 4, 3)

	snapshot := params.Snapshot()
	// B2 (the output layer's bias) is used for this comparison rather
	// than an earlier layer: its gradient is always nonzero whenever
	// advantage is nonzero (set below), since it sits immediately
	// before the softmax with no ReLU in between to zero out its
	// gradient path for an unlucky random seed, unlike W0/B0/W1/B1.
	before := append([]float32(nil), snapshot.B2.Data...)

	net, err := NewTrainingNetwork(params)
	require.NoError(t, err)
	applyOneGradientStep(params, net, rng, 0.5)

	assert.NotEqual(t, before, params.B2.Data, "sanity: the live params should have actually changed")
	assert.Equal(t, before, snapshot.B2.Data, "snapshot's weights must not change when the live params are updated afterward")
}

// TestActorRefreshUpdatesInternalSnapshot confirms Refresh actually
// replaces Actor's internal snapshot with the live Params' current
// weights, rather than continuing to serve the snapshot taken at
// NewActor time.
func TestActorRefreshUpdatesInternalSnapshot(t *testing.T) {
	rng := rand.New(rand.NewPCG(61, 62))
	params := NewParams(rng, 6, 4, 3)

	actor, err := NewActor(params)
	require.NoError(t, err)
	// See TestSnapshotIsUnaffectedBySubsequentGradientStep for why B2,
	// not W0, is used for this comparison.
	before := append([]float32(nil), actor.params.Load().B2.Data...)

	net, err := NewTrainingNetwork(params)
	require.NoError(t, err)
	applyOneGradientStep(params, net, rng, 0.5)

	require.NoError(t, actor.Refresh(params))
	after := actor.params.Load().B2.Data

	assert.NotEqual(t, before, after, "Refresh should pick up the training update")
}

// TestActorRefreshRejectsNilParams mirrors NewActor's nil check.
func TestActorRefreshRejectsNilParams(t *testing.T) {
	rng := rand.New(rand.NewPCG(63, 64))
	params := NewParams(rng, 6, 4, 3)

	actor, err := NewActor(params)
	require.NoError(t, err)

	assert.Error(t, actor.Refresh(nil))
}

// TestParamsSnapshotAndActorAreRaceFreeUnderConcurrentTraining is the
// stress test docs/plans/09-concurrency-safe-live-inference.md calls
// for: one goroutine repeatedly applies gradient steps to a live Params
// (guarded by Params.Lock, as pkg/reinforce.Trainer.RunEpoch does)
// while several other goroutines concurrently call Params.Snapshot,
// Actor.Act, and Actor.Refresh against that same Params. `go test
// -race` failing this test would mean the locking/snapshot/atomic
// design has a real data race.
func TestParamsSnapshotAndActorAreRaceFreeUnderConcurrentTraining(t *testing.T) {
	rng := rand.New(rand.NewPCG(71, 72))
	params := NewParams(rng, 6, 4, 3)

	net, err := NewTrainingNetwork(params)
	require.NoError(t, err)

	actor, err := NewActor(params)
	require.NoError(t, err)

	obs := rl.Observation{Values: []float32{0.1, -0.2, 0.3, 0.4, -0.5, 0.6}}

	var wg sync.WaitGroup

	wg.Add(1)
	go func() {
		defer wg.Done()
		trainRNG := rand.New(rand.NewPCG(1, 2))
		for range 200 {
			applyOneGradientStep(params, net, trainRNG, 0.01)
		}
	}()

	for reader := range 4 {
		wg.Add(1)
		go func(seed int) {
			defer wg.Done()
			readerRNG := rand.New(rand.NewPCG(uint64(seed), uint64(seed+1)))
			for range 100 {
				_ = params.Snapshot()

				_, err := actor.Act(obs, nil, readerRNG)
				assert.NoError(t, err)

				assert.NoError(t, actor.Refresh(params))
			}
		}(reader)
	}

	wg.Wait()
}
