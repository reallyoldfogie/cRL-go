package ppo

import (
	"context"
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
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
