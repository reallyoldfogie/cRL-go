package ppo

import (
	"context"
	"math"
	"math/rand/v2"
	"runtime"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/autograd"
	"github.com/reallyoldfogie/cRL-go/pkg/config"
	"github.com/reallyoldfogie/cRL-go/pkg/mat"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/reallyoldfogie/cRL-go/pkg/snakeenv"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func smallRecurrentTestSettings() config.Settings {
	s := smallTestSettings()
	s.HiddenSize = 6
	s.RecurrentChunkLen = 3
	return s
}

// TestRecurrentRunEpochSmokeTestNoPanicsOrNaNs mirrors
// TestRunEpochSmokeTestNoPanicsOrNaNs for RecurrentTrainer: several
// epochs against a toy environment, with no NaNs or panics, confirming
// the whole chunked-BPTT pipeline (rollout collection, GAE, chunking,
// padding, shuffled-minibatch Adam updates) works end to end.
// TestNewRecurrentWiresDisableBLASSetting mirrors
// TestNewWiresDisableBLASSetting (trainer_test.go) for NewRecurrent -
// see newRecurrentTrainer.
func TestNewRecurrentWiresDisableBLASSetting(t *testing.T) {
	t.Cleanup(func() { mat.SetBLASBackend(true) })

	settings := smallRecurrentTestSettings()
	settings.DisableBLAS = true
	_, err := NewRecurrent(settings, snakeEnvFactory(settings.GridSize), nil)
	require.NoError(t, err)
	assert.False(t, mat.BLASBackendEnabled())

	settings.DisableBLAS = false
	_, err = NewRecurrent(settings, snakeEnvFactory(settings.GridSize), nil)
	require.NoError(t, err)
	assert.True(t, mat.BLASBackendEnabled())
}

func TestRecurrentRunEpochSmokeTestNoPanicsOrNaNs(t *testing.T) {
	settings := smallRecurrentTestSettings()
	trainer, err := NewRecurrent(settings, snakeEnvFactory(settings.GridSize), nil)
	require.NoError(t, err)

	for epoch := range settings.Epochs {
		stats, err := trainer.RunEpoch(context.Background(), epoch)
		require.NoError(t, err)

		assert.False(t, math.IsNaN(float64(stats.AverageReturn)), "epoch %d: average return is NaN", epoch)
		assert.GreaterOrEqual(t, stats.SampleCount, 0)
		assert.Equal(t, epoch, stats.Epoch)
	}
}

func TestRecurrentRunEpochIsDeterministicForAFixedSeed(t *testing.T) {
	settings := smallRecurrentTestSettings()

	trainerA, err := NewRecurrent(settings, snakeEnvFactory(settings.GridSize), nil)
	require.NoError(t, err)
	statsA, err := trainerA.RunEpoch(context.Background(), 0)
	require.NoError(t, err)

	trainerB, err := NewRecurrent(settings, snakeEnvFactory(settings.GridSize), nil)
	require.NoError(t, err)
	statsB, err := trainerB.RunEpoch(context.Background(), 0)
	require.NoError(t, err)

	assert.Equal(t, statsA, statsB)
}

func TestNewRecurrentRejectsNonPositiveChunkLen(t *testing.T) {
	settings := smallRecurrentTestSettings()
	settings.RecurrentChunkLen = 0
	_, err := NewRecurrent(settings, snakeEnvFactory(settings.GridSize), nil)
	assert.ErrorContains(t, err, "RecurrentChunkLen")
}

func TestNewRecurrentUsesProvidedInitialParams(t *testing.T) {
	settings := smallRecurrentTestSettings()

	rng := rand.New(rand.NewPCG(99, 99))
	initialParams := actorcritic.NewRecurrentParams(rng, snakeenv.StateVectorSize(settings.GridSize), settings.HiddenSize, snakeenv.NumActions, 2)

	trainer, err := NewRecurrent(settings, snakeEnvFactory(settings.GridSize), initialParams)
	require.NoError(t, err)
	assert.Same(t, initialParams, trainer.Params())
}

// TestUpdateWorkerCountRespectsReservedCPUs confirms updateWorkerCount
// actually reads settings.UpdateReservedCPUs rather than a fixed amount
// (found live, 2026-11-23: a 6-core host sharing the box with a live
// Minecraft server, Grafana, Prometheus and Docker needed this
// operator-tunable, not hardcoded - see that field's own doc comment).
func TestUpdateWorkerCountRespectsReservedCPUs(t *testing.T) {
	settings := smallRecurrentTestSettings()
	settings.MinibatchSize = 1000 // large enough to never be the binding cap in this test

	settings.UpdateReservedCPUs = 1
	assert.Equal(t, max(1, runtime.NumCPU()-1), updateWorkerCount(settings))

	settings.UpdateReservedCPUs = 2
	assert.Equal(t, max(1, runtime.NumCPU()-2), updateWorkerCount(settings))
}

// TestUpdateWorkerCountFloorsAtOneWhenReservedCPUsExceedsNumCPU confirms
// an operator setting UpdateReservedCPUs close to or above the host's
// own core count still leaves exactly one worker, rather than zero or
// negative - the single-worker, no-replicas path RunEpoch already
// predates this feature must still work.
func TestUpdateWorkerCountFloorsAtOneWhenReservedCPUsExceedsNumCPU(t *testing.T) {
	settings := smallRecurrentTestSettings()
	settings.MinibatchSize = 1000
	settings.UpdateReservedCPUs = runtime.NumCPU() + 10
	assert.Equal(t, 1, updateWorkerCount(settings))
}

// TestUpdateWorkerCountCapsAtMinibatchSize confirms the pre-existing
// MinibatchSize cap still applies on top of the reservation - a worker
// with no chunk ever assigned to it would just be wasted memory.
func TestUpdateWorkerCountCapsAtMinibatchSize(t *testing.T) {
	settings := smallRecurrentTestSettings()
	settings.UpdateReservedCPUs = 1
	settings.MinibatchSize = 1
	assert.Equal(t, 1, updateWorkerCount(settings))
}

// TestRecurrentChunkReplayMatchesCollectionOutputs proves the end-to-end
// invariant needed by recurrent PPO: replaying a recorded chunk with its
// recorded incoming LSTM state reproduces the policy distribution and value
// estimate that were produced while collecting the rollout.
func TestRecurrentChunkReplayMatchesCollectionOutputs(t *testing.T) {
	rng := rand.New(rand.NewPCG(17, 29))
	params := actorcritic.NewRecurrentParams(rng, snakeenv.StateVectorSize(4), 6, snakeenv.NumActions, 2)
	rollout, err := collectRecurrentTrajectory(context.Background(), params, snakeEnvFactory(4), 10, rand.New(rand.NewPCG(31, 47)))
	require.NoError(t, err)
	require.NotEmpty(t, rollout.PolicyOutputs)
	require.Len(t, rollout.PolicyOutputs, len(rollout.Episode.Transitions))
	require.Len(t, rollout.ValueOutputs, len(rollout.Episode.Transitions))

	advantages := make([]float32, len(rollout.Episode.Transitions))
	returns := make([]float32, len(rollout.Episode.Transitions))
	chunks := chunkRollout(rollout, advantages, returns, 3, params.InputSize())
	network, err := NewRecurrentTrainingNetwork(params, 3, 1, LossConfig{ClipEpsilon: 0.2, EntropyCoef: 0.01, ValueCoef: 0.5})
	require.NoError(t, err)

	stepIndex := 0
	for _, chunk := range chunks {
		network.SetSplitChunkColumn(0, chunk.Steps, chunk.PolicyHPrevStart, chunk.PolicyCPrevStart, chunk.ValueHPrevStart, chunk.ValueCPrevStart)
		network.Graph.Forward()
		for stepOffset, step := range chunk.Steps {
			if !step.Real {
				continue
			}
			require.Len(t, rollout.PolicyOutputs[stepIndex], params.OutputSize())
			for action, want := range rollout.PolicyOutputs[stepIndex] {
				assert.InDelta(t, want, network.Actor.PolicyOutput[stepOffset].Val.Data[action], 1e-6,
					"policy output mismatch at step %d action %d", stepIndex, action)
			}
			assert.InDelta(t, rollout.ValueOutputs[stepIndex], network.Actor.ValueOutput[stepOffset].Val.Data[0], 1e-6,
				"value output mismatch at step %d", stepIndex)
			stepIndex++
		}
	}
	assert.Equal(t, len(rollout.Episode.Transitions), stepIndex)
}

// TestTrainOnChunkMinibatchParallelMatchesSingleThreaded confirms
// splitting a minibatch's chunks across tr.updateReplicas (see its own
// doc comment) accumulates the exact same gradient, up to
// floating-point summation-order noise, as running every chunk through
// tr.network alone - the parallel update computes the same quantity
// faster, it doesn't change what gets learned. Compares Grad directly
// rather than the post-Adam.Step weights, since Adam's own moving
// averages would make a tiny floating-point difference in the input
// harder to bound than it is right where the two paths actually differ.
func TestTrainOnChunkMinibatchParallelMatchesSingleThreaded(t *testing.T) {
	settings := smallRecurrentTestSettings()
	settings.MinibatchSize = 8

	seedParams := func() *actorcritic.RecurrentParams {
		rng := rand.New(rand.NewPCG(17, 29))
		return actorcritic.NewRecurrentParams(rng, snakeenv.StateVectorSize(settings.GridSize), settings.HiddenSize, snakeenv.NumActions, 2)
	}

	var chunks []recurrentChunk
	for i := range 8 {
		rollout, err := collectRecurrentTrajectory(context.Background(), seedParams(), snakeEnvFactory(settings.GridSize), 30, rand.New(rand.NewPCG(uint64(31+i), uint64(47+i))))
		require.NoError(t, err)
		advantages, returns := ComputeGAE(rollout.Rollout, settings.Gamma, settings.GAELambda)
		chunks = append(chunks, chunkRollout(rollout, advantages, returns, settings.RecurrentChunkLen, seedParams().InputSize())...)
	}
	require.Greater(t, len(chunks), 1, "need more than one chunk to actually exercise more than one worker")

	trainerParallel, err := NewRecurrent(settings, snakeEnvFactory(settings.GridSize), seedParams())
	require.NoError(t, err)
	if len(trainerParallel.updateReplicas) == 0 {
		t.Skip("this host has only one usable core for the update phase - nothing to compare against")
	}

	trainerSingle, err := NewRecurrent(settings, snakeEnvFactory(settings.GridSize), seedParams())
	require.NoError(t, err)
	trainerSingle.updateReplicas = nil

	trainerParallel.trainOnChunkMinibatch(chunks)
	trainerSingle.trainOnChunkMinibatch(chunks)

	parallelParams := trainerParallel.network.Actor.Parameters()
	singleParams := trainerSingle.network.Actor.Parameters()
	require.Equal(t, len(singleParams), len(parallelParams))
	for i := range singleParams {
		for j := range singleParams[i].Grad.Data {
			assert.InDelta(t, singleParams[i].Grad.Data[j], parallelParams[i].Grad.Data[j], 1e-3,
				"parameter %d, element %d: parallel-accumulated gradient diverged from the single-threaded reference", i, j)
		}
	}
}

// TestBatchedChunkTrainingMatchesSequentialOneChunkAtATime is the
// cross-check batched recurrent training actually needs: it directly
// compares (a) the original approach — a batchSize=1
// RecurrentTrainingNetwork, ZeroGrad once, then one Forward/Backward
// call per chunk in turn, accumulating gradient across all of them —
// against (b) the new approach — a batchSize=len(chunks) network, every
// chunk occupying its own batch column, ZeroGrad once, exactly one
// Forward/Backward call for the whole batch. Both must produce the
// same total loss (summed across every chunk/column) and the same
// final gradient for every shared parameter.
//
// This is the test that actually matters for this change: a subtle
// batching bug (wrong axis in AddBias or Softmax, the value loss's
// MulBroadcastCols leaking one column into another, a dropped or
// misrouted batch column) would silently produce wrong gradients that
// still "run" without panicking or producing NaN — it would only ever
// show up later as training that mysteriously doesn't learn as well as
// it used to, which is exactly the class of bug a quick smoke test
// can't catch. Comparing against the known-correct sequential behavior
// catches it here instead.
func TestBatchedChunkTrainingMatchesSequentialOneChunkAtATime(t *testing.T) {
	settings := smallRecurrentTestSettings()

	rng := rand.New(rand.NewPCG(53, 59))
	params := actorcritic.NewRecurrentParams(rng, snakeenv.StateVectorSize(settings.GridSize), settings.HiddenSize, snakeenv.NumActions, 2)

	var chunks []recurrentChunk
	for i := range 5 {
		rollout, err := collectRecurrentTrajectory(context.Background(), params, snakeEnvFactory(settings.GridSize), 10, rand.New(rand.NewPCG(uint64(61+i), uint64(67+i))))
		require.NoError(t, err)
		advantages, returns := ComputeGAE(rollout.Rollout, settings.Gamma, settings.GAELambda)
		chunks = append(chunks, chunkRollout(rollout, advantages, returns, settings.RecurrentChunkLen, params.InputSize())...)
	}
	require.Greater(t, len(chunks), 1, "need more than one chunk for this comparison to mean anything")

	lossCfg := LossConfig{ClipEpsilon: settings.ClipEpsilon, EntropyCoef: settings.EntropyCoef, ValueCoef: settings.ValueCoef}

	sequentialNet, err := NewRecurrentTrainingNetwork(params, settings.RecurrentChunkLen, 1, lossCfg)
	require.NoError(t, err)
	sequentialNet.Actor.ZeroGrad()
	var sequentialLossSum float64
	for _, chunk := range chunks {
		sequentialNet.SetSplitChunkColumn(0, chunk.Steps, chunk.PolicyHPrevStart, chunk.PolicyCPrevStart, chunk.ValueHPrevStart, chunk.ValueCPrevStart)
		sequentialNet.Graph.Forward()
		sequentialLossSum += chunkSumFloat32(sequentialNet.Loss.Val.Data)
		sequentialNet.Graph.Backward()
	}

	batchedNet, err := NewRecurrentTrainingNetwork(params, settings.RecurrentChunkLen, len(chunks), lossCfg)
	require.NoError(t, err)
	batchedNet.Actor.ZeroGrad()
	for col, chunk := range chunks {
		batchedNet.SetSplitChunkColumn(col, chunk.Steps, chunk.PolicyHPrevStart, chunk.PolicyCPrevStart, chunk.ValueHPrevStart, chunk.ValueCPrevStart)
	}
	batchedNet.Graph.Forward()
	batchedLossSum := chunkSumFloat32(batchedNet.Loss.Val.Data)
	batchedNet.Graph.Backward()

	assert.InDelta(t, sequentialLossSum, batchedLossSum, 1e-2, "batched total loss must match the sum of each chunk's own sequential loss")

	sequentialParams := sequentialNet.Actor.Parameters()
	batchedParams := batchedNet.Actor.Parameters()
	require.Equal(t, len(sequentialParams), len(batchedParams))
	for i := range sequentialParams {
		require.Equal(t, len(sequentialParams[i].Grad.Data), len(batchedParams[i].Grad.Data), "parameter %d shape mismatch", i)
		for j := range sequentialParams[i].Grad.Data {
			assert.InDelta(t, sequentialParams[i].Grad.Data[j], batchedParams[i].Grad.Data[j], 1e-2,
				"parameter %d, element %d: batched gradient diverged from the sequential, one-chunk-at-a-time reference", i, j)
		}
	}
}

// chunkSumFloat32 sums data as a float64, mirroring sumMatrix
// (gradcheck_test.go) but for a plain []float32 rather than a
// *mat.Matrix, and distinctly named from pkg/actorcritic's own
// identically-purposed chunkSumMatrix since Go test files in different
// packages can't share helpers.
func chunkSumFloat32(data []float32) float64 {
	var sum float64
	for _, v := range data {
		sum += float64(v)
	}
	return sum
}

func TestMaskedRecurrentReplayRatioMatchesCollection(t *testing.T) {
	rng := rand.New(rand.NewPCG(71, 73))
	params := actorcritic.NewRecurrentParams(rng, 4, 6, 3, 2)
	mask := []bool{true, false, true}
	observation := rl.Observation{Values: []float32{0.2, -0.1, 0.3, 0.4}}

	collectionNet, err := actorcritic.NewRecurrentInferenceNetwork(params)
	require.NoError(t, err)
	copy(collectionNet.Input.Val.Data, observation.Values)
	collectionNet.HPrevIn.Val.Clear()
	collectionNet.CPrevIn.Val.Clear()
	collectionNet.Graph.Forward()
	const action rl.Action = 0
	oldLogProb := maskedActionLogProb(collectionNet.PolicyOutput.Val, mask, action)

	trainingNet, err := NewRecurrentTrainingNetwork(params, 1, 1, LossConfig{ClipEpsilon: 0.2, EntropyCoef: 0.01, ValueCoef: 0.5})
	require.NoError(t, err)
	trainingNet.SetChunkColumn(0, []recurrentTrainingStep{{
		Observation: observation,
		Action:      action,
		Mask:        mask,
		OldLogProb:  oldLogProb,
		Real:        true,
	}}, nil, nil)
	trainingNet.Graph.Forward()
	currentLogProb := actionLogProb(trainingNet.Actor.PolicyOutput[0].Val, action)
	ratio := math.Exp(float64(currentLogProb - oldLogProb))

	assert.InDelta(t, oldLogProb, currentLogProb, 1e-6)
	assert.InDelta(t, 1, ratio, 1e-6)
}

func TestRecurrentRealStepCountExcludesPadding(t *testing.T) {
	rollout := &RecurrentRollout{
		Rollout: &Rollout{Episode: episodeOfLength(5), LogProbs: make([]float32, 5)},
		HPrevs:  make([][]float32, 5),
		CPrevs:  make([][]float32, 5),
	}
	chunks := chunkRollout(rollout, make([]float32, 5), make([]float32, 5), 3, 4)
	assert.Len(t, chunks, 2)
	assert.Equal(t, 5, recurrentRealStepCount(chunks))
}

// TestChunkRolloutPadsAShortFinalChunkWithZeroedZeroAdvantageSteps
// confirms a rollout whose length isn't a multiple of chunkLen pads its
// last chunk out, rather than dropping it or producing a short slice,
// and that every padded entry carries Advantage 0 (so
// normalizeRecurrentAdvantages, and the network's own StepMask, both
// have something well-defined to work with).
func TestChunkRolloutPadsAShortFinalChunkWithZeroedAdvantageSteps(t *testing.T) {
	rollout := &RecurrentRollout{
		Rollout: &Rollout{
			Episode:  episodeOfLength(5),
			LogProbs: []float32{-0.1, -0.2, -0.3, -0.4, -0.5},
			Values:   []float32{1, 2, 3, 4, 5},
		},
		HPrevs: [][]float32{{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}},
		CPrevs: [][]float32{{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}},
	}
	advantages := []float32{0.1, 0.2, 0.3, 0.4, 0.5}
	returns := []float32{1, 2, 3, 4, 5}

	chunks := chunkRollout(rollout, advantages, returns, 3, 4)
	require.Len(t, chunks, 2, "5 steps at chunkLen=3 must produce 2 chunks (3 + 2 padded to 3)")

	require.Len(t, chunks[0].Steps, 3)
	for _, step := range chunks[0].Steps {
		assert.True(t, step.Real)
	}
	assert.Equal(t, []float32{0, 0}, chunks[0].HPrevStart)

	require.Len(t, chunks[1].Steps, 3)
	assert.True(t, chunks[1].Steps[0].Real)
	assert.True(t, chunks[1].Steps[1].Real)
	assert.False(t, chunks[1].Steps[2].Real, "the 6th slot pads the 5-step episode out to chunkLen=3")
	assert.Equal(t, float32(0), chunks[1].Steps[2].Advantage)
	assert.Len(t, chunks[1].Steps[2].Observation.Values, 4, "a padded step's placeholder observation must still have the right InputSize")
	assert.Equal(t, []float32{3, 3}, chunks[1].HPrevStart, "the second chunk starts from step 3's recorded incoming state")
}

// TestRecurrentPolicyAndValueAdamOperateOnDisjointParameters guards the
// invariant trainOnChunkMinibatch's independent clipping depends on:
// policyAdam and valueAdam must never share a *autograd.Var, and
// stepping one must never move a weight that belongs to the other. A
// future refactor that accidentally wires both optimizers to the same
// parameter list (e.g. copy-pasting PolicyParameters() into both
// newRecurrentTrainer calls) would still compile and still train
// something, just silently back out this fix.
func TestRecurrentPolicyAndValueAdamOperateOnDisjointParameters(t *testing.T) {
	settings := smallRecurrentTestSettings()
	trainer, err := NewRecurrent(settings, snakeEnvFactory(settings.GridSize), nil)
	require.NoError(t, err)

	policyParams := trainer.network.Actor.PolicyParameters()
	valueParams := trainer.network.Actor.ValueParameters()
	require.NotEmpty(t, policyParams)
	require.NotEmpty(t, valueParams)

	isPolicyVar := make(map[*autograd.Var]bool, len(policyParams))
	for _, p := range policyParams {
		isPolicyVar[p] = true
	}
	for _, v := range valueParams {
		assert.False(t, isPolicyVar[v], "a value-branch parameter is also in the policy branch's parameter list")
	}

	beforeValue := make([]float32, 0)
	for _, v := range valueParams {
		beforeValue = append(beforeValue, append([]float32(nil), v.Val.Data...)...)
	}

	for _, p := range policyParams {
		p.Grad.Fill(1000.0)
	}
	trainer.policyAdam.Step(1)

	idx := 0
	for _, v := range valueParams {
		for _, got := range v.Val.Data {
			assert.Equal(t, beforeValue[idx], got, "a value-branch weight moved after stepping only the policy optimizer")
			idx++
		}
	}
}

// TestRecurrentPolicyGradientClipIsUnaffectedByAValueGradientOutlier is
// the direct regression test for the bug docs/plans/21 left open: before
// this fix, policy and value shared one Adam/clip over their combined
// gradient, so a value-loss spike (observed live: value-gradient norms
// roughly three orders of magnitude larger than policy's) forced one
// shared clip-scale factor that shrank the policy's own, otherwise fine,
// gradient right along with it. With independent optimizers, the
// policy's own small gradient must be applied at its own, unclipped
// scale regardless of how large the value branch's gradient is that
// same step.
func TestRecurrentPolicyGradientClipIsUnaffectedByAValueGradientOutlier(t *testing.T) {
	settings := smallRecurrentTestSettings()
	settings.MaxGradNorm = 0.5
	trainer, err := NewRecurrent(settings, snakeEnvFactory(settings.GridSize), nil)
	require.NoError(t, err)

	policyParams := trainer.network.Actor.PolicyParameters()
	valueParams := trainer.network.Actor.ValueParameters()
	for _, p := range policyParams {
		p.Grad.Fill(0.01)
	}
	for _, v := range valueParams {
		v.Grad.Fill(1000.0)
	}

	policyNorm := trainer.policyAdam.GradientNorm(1)
	require.Less(t, policyNorm, settings.MaxGradNorm, "test setup: the policy gradient must start below the clip, or this proves nothing")

	assert.Equal(t, policyNorm, trainer.policyAdam.AppliedGradientNorm(1),
		"the policy branch's own small gradient must not be clipped just because the value branch's gradient is huge")
	assert.Equal(t, settings.MaxGradNorm, trainer.valueAdam.AppliedGradientNorm(1),
		"the value branch's outlier gradient should still be clipped on its own")
}

// TestRecurrentOptimizerStateRoundTripsBothBranchesIndependently checks
// OptimizerState/RestoreOptimizerState's concatenate-then-split: a
// boundary computed wrong (off-by-one, or swapped halves) would still
// often produce a state of the right total length, so this checks each
// branch's own restored Adam state individually, not just that the
// opaque combined blob round-trips byte-for-byte.
func TestRecurrentOptimizerStateRoundTripsBothBranchesIndependently(t *testing.T) {
	settings := smallRecurrentTestSettings()
	trainerA, err := NewRecurrent(settings, snakeEnvFactory(settings.GridSize), nil)
	require.NoError(t, err)

	for _, p := range trainerA.network.Actor.PolicyParameters() {
		p.Grad.Fill(0.3)
	}
	for _, v := range trainerA.network.Actor.ValueParameters() {
		v.Grad.Fill(0.7)
	}
	trainerA.policyAdam.Step(1)
	trainerA.valueAdam.Step(1)
	savedState := trainerA.OptimizerState()

	trainerB, err := NewRecurrent(settings, snakeEnvFactory(settings.GridSize), nil)
	require.NoError(t, err)
	require.NoError(t, trainerB.RestoreOptimizerState(savedState))

	assert.Equal(t, trainerA.policyAdam.State(), trainerB.policyAdam.State())
	assert.Equal(t, trainerA.valueAdam.State(), trainerB.valueAdam.State())
	assert.Equal(t, savedState, trainerB.OptimizerState())
}

func episodeOfLength(n int) *rl.Episode {
	transitions := make([]rl.Transition, n)
	for i := range transitions {
		transitions[i] = rl.Transition{Observation: rl.Observation{Values: []float32{0, 0, 0, 0}}, Action: 0, Reward: 0}
	}
	return &rl.Episode{Transitions: transitions}
}
