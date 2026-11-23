package ppo

import (
	"context"
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/config"
	"github.com/reallyoldfogie/cRL-go/pkg/snakeenv"
)

// benchmarkRecurrentChunks builds at least numChunks chunks of chunkLen
// steps each from fresh rollouts against params, for
// BenchmarkRecurrentChunkTraining to replay. Built once per benchmark,
// outside the timed loop, so every BatchSize sub-benchmark trains the
// exact same workload.
func benchmarkRecurrentChunks(b *testing.B, params *actorcritic.RecurrentParams, gridSize, episodeLen, chunkLen, numChunks int) []recurrentChunk {
	b.Helper()

	var chunks []recurrentChunk
	for i := 0; len(chunks) < numChunks; i++ {
		rollout, err := collectRecurrentTrajectory(context.Background(), params, snakeEnvFactory(gridSize), episodeLen, rand.New(rand.NewPCG(uint64(9000+i), uint64(90001+i))))
		if err != nil {
			b.Fatalf("collecting rollout: %v", err)
		}
		advantages, returns := ComputeGAE(rollout.Rollout, 0.99, 0.95)
		chunks = append(chunks, chunkRollout(rollout, advantages, returns, chunkLen, params.InputSize())...)
	}
	return chunks[:numChunks]
}

// BenchmarkRecurrentChunkTraining is the before/after for batching
// recurrent chunk training (see RecurrentTrainingNetwork's and
// trainNetworkOnChunks' own doc comments for the design).
//
// Its "batch=1" case trains the same fixed set of chunks exactly the
// way this trainer worked before batching: one chunk, one
// Forward/Backward call, every time. The larger-BatchSize cases are the
// same current production code path (trainNetworkOnChunks), just handed
// a bigger BatchSize — trainNetworkOnChunks and RecurrentTrainingNetwork
// don't special-case BatchSize 1, so "batch=1" here is a faithful
// reproduction of the old behavior rather than a separate implementation
// that could drift out of sync with it.
//
// Sizes below match configs/config.json's grid_size/hidden_size/
// minibatch_size (the shipped training config) so the numbers reflect
// real training, not a toy problem. Run with:
//
//	go test ./pkg/ppo/ -run '^$' -bench BenchmarkRecurrentChunkTraining -benchtime=5x
func BenchmarkRecurrentChunkTraining(b *testing.B) {
	const (
		gridSize   = 36  // configs/config.json: grid_size
		hiddenSize = 128 // configs/config.json: hidden_size
		episodeLen = 32
		chunkLen   = 16
		numChunks  = 64 // configs/config.json: minibatch_size
	)

	rng := rand.New(rand.NewPCG(11, 13))
	params := actorcritic.NewRecurrentParams(rng, snakeenv.StateVectorSize(gridSize), hiddenSize, snakeenv.NumActions, 2)
	chunks := benchmarkRecurrentChunks(b, params, gridSize, episodeLen, chunkLen, numChunks)
	cfg := LossConfig{ClipEpsilon: 0.2, EntropyCoef: 0.01, ValueCoef: 0.5}

	for _, batchSize := range []int{1, 2, 4, 8, 16, 32, 64} {
		b.Run(fmt.Sprintf("batch=%d", batchSize), func(b *testing.B) {
			network, err := NewRecurrentTrainingNetwork(params, chunkLen, batchSize, cfg)
			if err != nil {
				b.Fatalf("building network: %v", err)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				trainNetworkOnChunks(network, chunks, cfg)
			}
		})
	}
}

// newUpdateReplicasForBench mirrors newUpdateReplicas exactly, except
// batchSize is supplied directly instead of being derived from settings
// via maxUpdateBatchSize — letting BenchmarkRecurrentChunkTrainingFullWorkerSplit
// hold the worker count (and therefore how many cores trainOnChunkMinibatch
// actually uses) fixed while varying only the per-worker batch size.
func newUpdateReplicasForBench(params *actorcritic.RecurrentParams, settings config.Settings, lossCfg LossConfig, batchSize int) ([]*RecurrentTrainingNetwork, error) {
	extra := updateWorkerCount(settings) - 1
	replicas := make([]*RecurrentTrainingNetwork, extra)
	for i := range extra {
		replica, err := NewRecurrentTrainingNetwork(params, settings.RecurrentChunkLen, batchSize, lossCfg)
		if err != nil {
			return nil, fmt.Errorf("building update replica network %d: %w", i, err)
		}
		replicas[i] = replica
	}
	return replicas, nil
}

// newRecurrentTrainerForBench builds a *RecurrentTrainer by hand (same
// shape as newRecurrentTrainer, minus the env-derived param/settings
// validation this benchmark doesn't need), with every one of its
// updateWorkerCount(settings) networks (tr.network plus every replica)
// built at the given batchSize rather than at maxUpdateBatchSize(settings)
// — so passing batchSize=1 reproduces exactly how trainOnChunkMinibatch
// behaved before this package's chunk-batching was added (one chunk, one
// Forward/Backward call, per worker), still spread across the same
// updateWorkerCount(settings) goroutines/cores as today's actual
// behavior (batchSize=maxUpdateBatchSize(settings)) — isolating the
// effect of batching alone from the pre-existing per-core parallelism.
func newRecurrentTrainerForBench(params *actorcritic.RecurrentParams, settings config.Settings, batchSize int) (*RecurrentTrainer, error) {
	lossCfg := LossConfig{ClipEpsilon: settings.ClipEpsilon, EntropyCoef: settings.EntropyCoef, ValueCoef: settings.ValueCoef}

	network, err := NewRecurrentTrainingNetwork(params, settings.RecurrentChunkLen, batchSize, lossCfg)
	if err != nil {
		return nil, fmt.Errorf("building training network: %w", err)
	}
	replicas, err := newUpdateReplicasForBench(params, settings, lossCfg, batchSize)
	if err != nil {
		return nil, err
	}

	return &RecurrentTrainer{
		settings:       settings,
		params:         params,
		network:        network,
		updateReplicas: replicas,
		policyAdam:     actorcritic.NewAdamWithGradClip(network.Actor.PolicyParameters(), settings.LearningRate, settings.MaxGradNorm),
		valueAdam:      actorcritic.NewAdamWithGradClip(network.Actor.ValueParameters(), settings.LearningRate, settings.MaxGradNorm),
	}, nil
}

// BenchmarkRecurrentChunkTrainingFullWorkerSplit is
// BenchmarkRecurrentChunkTraining's sequel: instead of a single
// goroutine running trainNetworkOnChunks directly, it drives the real
// trainOnChunkMinibatch entry point, so updateWorkerCount(settings)
// goroutines (4 on this benchmark's host: runtime.NumCPU() minus the
// default UpdateReservedCPUs=2 — see updateWorkerCount) each train their
// own share of the minibatch concurrently, exactly as RecurrentTrainer.
// RunEpoch does during real training. "sequential" forces every one of
// those workers to batchSize=1 (today's worker-count, yesterday's
// per-worker behavior); "batched" uses maxUpdateBatchSize(settings) —
// 16 here, the sweet spot BenchmarkRecurrentChunkTraining found on a
// single core — so the only thing this isolates is whether batching
// still helps once it's compounding with the parallelism that already
// existed.
//
// Network size (256 wide, 3 hidden layers) matches current real
// training, not configs/config.json's shipped 128/2 default — ask
// before trusting these numbers for a differently-sized network.
//
//	go test ./pkg/ppo/ -run '^$' -bench BenchmarkRecurrentChunkTrainingFullWorkerSplit -benchtime=5x
func BenchmarkRecurrentChunkTrainingFullWorkerSplit(b *testing.B) {
	const (
		gridSize        = 36 // configs/config.json: grid_size
		hiddenSize      = 256
		numHiddenLayers = 3
		episodeLen      = 32
		chunkLen        = 16
		numChunks       = 64 // configs/config.json: minibatch_size
	)

	settings := config.Default()
	settings.HiddenSize = hiddenSize
	settings.RecurrentChunkLen = chunkLen
	settings.MinibatchSize = numChunks

	rng := rand.New(rand.NewPCG(17, 19))
	params := actorcritic.NewRecurrentParams(rng, snakeenv.StateVectorSize(gridSize), hiddenSize, snakeenv.NumActions, numHiddenLayers)
	chunks := benchmarkRecurrentChunks(b, params, gridSize, episodeLen, chunkLen, numChunks)

	workers := updateWorkerCount(settings)
	sweetSpot := maxUpdateBatchSize(settings)
	b.Logf("updateWorkerCount=%d maxUpdateBatchSize=%d", workers, sweetSpot)

	cases := []struct {
		name      string
		batchSize int
	}{
		{"sequential", 1},
		{"batched", sweetSpot},
	}
	for _, c := range cases {
		b.Run(fmt.Sprintf("%s(batch=%d)x%dworkers", c.name, c.batchSize, workers), func(b *testing.B) {
			tr, err := newRecurrentTrainerForBench(params, settings, c.batchSize)
			if err != nil {
				b.Fatalf("building trainer: %v", err)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				tr.trainOnChunkMinibatch(chunks)
			}
		})
	}
}
