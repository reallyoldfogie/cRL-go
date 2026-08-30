package actorcritic

import (
	"math/rand/v2"
	"sync"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// applyOneGradientStep applies one gradient step to net/params, mutating
// every parameter's Val by a fixed nonzero gradient. Unlike
// pkg/policy's identically-named test helper, this doesn't drive a
// Forward/Backward pass: actorcritic.TrainingNetwork deliberately has no
// Graph/Loss of its own (that's composed by pkg/ppo on top of it), so
// gradients are set directly on the parameter Vars, mirroring how
// adam_test.go exercises Adam.Step without a full graph. The actual
// mutation (ApplyGradientStep) is guarded by params.Lock/Unlock, exactly
// as pkg/ppo.Trainer.RunEpoch guards it — params must be the same Params
// net was built from.
func applyOneGradientStep(params *Params, net *TrainingNetwork, learningRate float32) {
	net.ZeroGrad()
	for _, p := range net.Parameters() {
		for i := range p.Grad.Data {
			p.Grad.Data[i] = 1.0
		}
	}

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
	before := append([]float32(nil), snapshot.W0.Data...)

	net, err := NewTrainingNetwork(params)
	require.NoError(t, err)
	applyOneGradientStep(params, net, 0.5)

	assert.NotEqual(t, before, params.W0.Data, "sanity: the live params should have actually changed")
	assert.Equal(t, before, snapshot.W0.Data, "snapshot's weights must not change when the live params are updated afterward")
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
	before := append([]float32(nil), actor.params.Load().W0.Data...)

	net, err := NewTrainingNetwork(params)
	require.NoError(t, err)
	applyOneGradientStep(params, net, 0.5)

	require.NoError(t, actor.Refresh(params))
	after := actor.params.Load().W0.Data

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
// (guarded by Params.Lock, as pkg/ppo.Trainer.RunEpoch does) while
// several other goroutines concurrently call Params.Snapshot,
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
		for range 200 {
			applyOneGradientStep(params, net, 0.01)
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
