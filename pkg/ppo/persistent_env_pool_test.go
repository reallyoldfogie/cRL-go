package ppo

import (
	"context"
	"errors"
	"math/rand/v2"
	"sync/atomic"
	"testing"
	"time"

	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// poolTrackingEnv tracks how many goroutines are inside Reset/Step at
// once (current) and the maximum ever observed (maxSeen), plus a plain
// reset count. A brief sleep inside both methods widens the race window
// so collectRolloutsPool sharing one env across two goroutines would
// reliably be caught by maxSeen > 1, not just theoretically possible.
type poolTrackingEnv struct {
	current    *int32
	maxSeen    *int32
	resetCount *int32
}

func newPoolTrackingEnv() *poolTrackingEnv {
	return &poolTrackingEnv{current: new(int32), maxSeen: new(int32), resetCount: new(int32)}
}

func (e *poolTrackingEnv) enter() {
	n := atomic.AddInt32(e.current, 1)
	for {
		old := atomic.LoadInt32(e.maxSeen)
		if n <= old || atomic.CompareAndSwapInt32(e.maxSeen, old, n) {
			break
		}
	}
}

func (e *poolTrackingEnv) exit() {
	atomic.AddInt32(e.current, -1)
}

func (e *poolTrackingEnv) Reset(ctx context.Context) (rl.Observation, error) {
	e.enter()
	defer e.exit()
	atomic.AddInt32(e.resetCount, 1)
	time.Sleep(time.Millisecond)
	return rl.Observation{Values: []float32{0, 0}}, nil
}

func (e *poolTrackingEnv) Step(ctx context.Context, action rl.Action) (rl.StepResult, error) {
	e.enter()
	defer e.exit()
	time.Sleep(time.Millisecond)
	return rl.StepResult{Observation: rl.Observation{Values: []float32{0, 0}}, Reward: 1, Done: true}, nil
}

func (*poolTrackingEnv) ObservationSize() int { return 2 }
func (*poolTrackingEnv) ActionSpace() int     { return 2 }

// erroringEnv always fails its Step call, used to confirm a pooled
// worker's own error surfaces through RunEpoch rather than being
// swallowed by sync.WaitGroup.
type erroringEnv struct{}

func (erroringEnv) Reset(ctx context.Context) (rl.Observation, error) {
	return rl.Observation{Values: []float32{0, 0}}, nil
}

func (erroringEnv) Step(ctx context.Context, action rl.Action) (rl.StepResult, error) {
	return rl.StepResult{}, errors.New("erroringEnv: simulated failure")
}

func (erroringEnv) ObservationSize() int { return 2 }
func (erroringEnv) ActionSpace() int     { return 2 }

func TestNewWithPersistentEnvPoolRejectsEmptyEnvs(t *testing.T) {
	settings := persistentTestSettings()

	_, err := NewWithPersistentEnvPool(settings, nil, nil)
	require.Error(t, err)
}

// TestCollectRolloutsPoolDistributesAcrossAllEnvs confirms
// RolloutSize episodes are striped across every pooled env even when
// RolloutSize doesn't divide evenly by the pool size.
func TestCollectRolloutsPoolDistributesAcrossAllEnvs(t *testing.T) {
	settings := persistentTestSettings()
	settings.RolloutSize = 7

	const n = 3
	envs := make([]rl.Environment, n)
	tracked := make([]*poolTrackingEnv, n)
	for i := range n {
		tracked[i] = newPoolTrackingEnv()
		envs[i] = tracked[i]
	}

	trainer, err := NewWithPersistentEnvPool(settings, envs, nil)
	require.NoError(t, err)

	_, err = trainer.RunEpoch(context.Background(), 0)
	require.NoError(t, err)

	var total int32
	for i, env := range tracked {
		count := atomic.LoadInt32(env.resetCount)
		total += count
		assert.InDeltaf(t, float64(settings.RolloutSize)/n, float64(count), 1,
			"env %d got %d episodes, expected close to RolloutSize/n", i, count)
	}
	assert.Equal(t, int32(settings.RolloutSize), total, "every episode must be collected exactly once across the pool")
}

// TestCollectRolloutsPoolNeverSharesOneEnvAcrossGoroutines is the actual
// correctness property NewWithPersistentEnvPool depends on: no two
// goroutines may ever be inside the same envs[i] concurrently.
func TestCollectRolloutsPoolNeverSharesOneEnvAcrossGoroutines(t *testing.T) {
	settings := persistentTestSettings()
	settings.RolloutSize = 12

	const n = 4
	envs := make([]rl.Environment, n)
	tracked := make([]*poolTrackingEnv, n)
	for i := range n {
		tracked[i] = newPoolTrackingEnv()
		envs[i] = tracked[i]
	}

	trainer, err := NewWithPersistentEnvPool(settings, envs, nil)
	require.NoError(t, err)

	for epoch := range settings.Epochs {
		_, err := trainer.RunEpoch(context.Background(), epoch)
		require.NoError(t, err)
	}

	for i, env := range tracked {
		assert.LessOrEqualf(t, atomic.LoadInt32(env.maxSeen), int32(1),
			"env %d was entered by more than one goroutine at once", i)
	}
}

// TestNewWithPersistentEnvPoolPropagatesContextCancellation mirrors
// TestNewWithPersistentEnvPropagatesContextCancellation for the pool
// path: a canceled context must surface as an error, not be swallowed.
func TestNewWithPersistentEnvPoolPropagatesContextCancellation(t *testing.T) {
	settings := persistentTestSettings()

	envs := []rl.Environment{cancelAwareEnv{}, cancelAwareEnv{}}
	trainer, err := NewWithPersistentEnvPool(settings, envs, nil)
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = trainer.RunEpoch(ctx, 0)
	require.Error(t, err)
	assert.True(t, errors.Is(err, context.Canceled), "expected error to wrap context.Canceled, got: %v", err)
}

// TestNewWithPersistentEnvPoolSurfacesPerEnvError confirms one pooled
// worker's error reaches the caller through RunEpoch rather than being
// dropped by sync.WaitGroup.
func TestNewWithPersistentEnvPoolSurfacesPerEnvError(t *testing.T) {
	settings := persistentTestSettings()

	envs := []rl.Environment{erroringEnv{}, countingEnv{resetCount: new(int)}}
	trainer, err := NewWithPersistentEnvPool(settings, envs, nil)
	require.NoError(t, err)

	_, err = trainer.RunEpoch(context.Background(), 0)
	require.Error(t, err)
}

// TestNewWithPersistentEnvPoolProducesSameShapeAsPersistentEnv sanity
// checks that an N=1 pool degenerates to the same result as
// NewWithPersistentEnv given the same seed and a deterministic
// environment, even though mc-rsi-trainer never routes a single
// environment through this path itself (see pkg/leapfrog.trainStudent).
func TestNewWithPersistentEnvPoolProducesSameShapeAsPersistentEnv(t *testing.T) {
	settings := persistentTestSettings()

	seqEnv := countingEnv{resetCount: new(int)}
	seqTrainer, err := NewWithPersistentEnv(settings, func(*rand.Rand) (rl.Environment, error) { return seqEnv, nil }, nil)
	require.NoError(t, err)
	seqStats, err := seqTrainer.RunEpoch(context.Background(), 0)
	require.NoError(t, err)

	poolEnv := countingEnv{resetCount: new(int)}
	poolTrainer, err := NewWithPersistentEnvPool(settings, []rl.Environment{poolEnv}, nil)
	require.NoError(t, err)
	poolStats, err := poolTrainer.RunEpoch(context.Background(), 0)
	require.NoError(t, err)

	assert.Equal(t, seqStats.SampleCount, poolStats.SampleCount)
	assert.Equal(t, seqStats.AverageReturn, poolStats.AverageReturn)
}
