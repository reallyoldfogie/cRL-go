package ppo

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// newRecurrentPoolTrainer builds a RecurrentTrainer over n poolTrackingEnv
// instances with RolloutSize == n (one slot per env, so resume/progress
// behavior can be asserted per env unambiguously).
func newRecurrentPoolTrainer(t *testing.T, n int) (*RecurrentTrainer, []*poolTrackingEnv) {
	t.Helper()
	settings := recurrentPersistentTestSettings()
	settings.RolloutSize = n

	envs := make([]rl.Environment, n)
	tracked := make([]*poolTrackingEnv, n)
	for i := range n {
		tracked[i] = newPoolTrackingEnv()
		envs[i] = tracked[i]
	}

	trainer, err := NewRecurrentWithPersistentEnvPool(settings, envs, nil)
	require.NoError(t, err)
	return trainer, tracked
}

// TestRecurrentResumeRolloutsSkipsAlreadyCollectedSlots covers the
// 2026-11-02 feature: a crash mid-epoch shouldn't force re-collecting
// trajectories the epoch had already finished before it died.
// ResumeRollouts' pre-filled slots must be left untouched (returned
// exactly as supplied, their owning env never re-reset) while a nil
// slot still gets collected normally - by whichever env's goroutine
// happens to claim it, not necessarily the env at that same numeric
// index (collectRecurrentRolloutsPool's work-stealing no longer
// guarantees env[i] claims slot[i] - see its own doc comment), so this
// checks the pool's total reset count rather than pinning the
// collection to one specific env.
func TestRecurrentResumeRolloutsSkipsAlreadyCollectedSlots(t *testing.T) {
	trainer, tracked := newRecurrentPoolTrainer(t, 3)

	collected, err := trainer.collectRollouts(context.Background(), 0)
	require.NoError(t, err)
	require.Len(t, collected, 3)
	for _, r := range collected {
		require.NotNil(t, r)
	}

	for _, tr := range tracked {
		atomic.StoreInt32(tr.resetCount, 0)
	}

	resumed := make([]*RecurrentRollout, 3)
	resumed[0] = collected[0]
	resumed[2] = collected[2]
	// resumed[1] stays nil - that slot still needs collecting.
	trainer.ResumeRollouts(resumed)

	result, err := trainer.collectRollouts(context.Background(), 1)
	require.NoError(t, err)

	assert.Same(t, resumed[0], result[0], "slot 0 was supplied via ResumeRollouts - it must be returned untouched")
	assert.Same(t, resumed[2], result[2], "slot 2 was supplied via ResumeRollouts - it must be returned untouched")
	require.NotNil(t, result[1], "slot 1 was nil - it must have been freshly collected")

	var totalResets int32
	for _, tr := range tracked {
		totalResets += atomic.LoadInt32(tr.resetCount)
	}
	assert.EqualValues(t, 1, totalResets, "exactly one fresh collection should have happened, for the one un-resumed slot")
}

// TestRecurrentSetRolloutProgressFiresOnlyForFreshCollections covers the
// other half of the 2026-11-02 feature: the progress callback a caller
// would use to persist each trajectory as it arrives must fire exactly
// once per freshly collected slot, and never for one supplied via
// ResumeRollouts (it was already persisted before the crash that's being
// recovered from - re-reporting it would just re-do redundant work).
func TestRecurrentSetRolloutProgressFiresOnlyForFreshCollections(t *testing.T) {
	trainer, _ := newRecurrentPoolTrainer(t, 3)

	collected, err := trainer.collectRollouts(context.Background(), 0)
	require.NoError(t, err)

	var mu sync.Mutex
	var seen []int
	trainer.SetRolloutProgress(func(epoch, index int, rollout *RecurrentRollout) {
		mu.Lock()
		defer mu.Unlock()
		seen = append(seen, index)
		assert.Equal(t, 1, epoch)
		assert.NotNil(t, rollout)
	})

	resumed := make([]*RecurrentRollout, 3)
	resumed[0] = collected[0]
	resumed[2] = collected[2]
	trainer.ResumeRollouts(resumed)

	_, err = trainer.collectRollouts(context.Background(), 1)
	require.NoError(t, err)

	assert.Equal(t, []int{1}, seen, "the progress callback must fire only for the freshly-collected slot, never a resumed one")
}

// TestRecurrentResumeRolloutsConsumedExactlyOnce covers ResumeRollouts'
// own stated contract: it must apply to the very next collectRollouts
// call and no other, so a later epoch never silently reuses a stale
// epoch's partial rollouts.
func TestRecurrentResumeRolloutsConsumedExactlyOnce(t *testing.T) {
	trainer, tracked := newRecurrentPoolTrainer(t, 3)

	collected, err := trainer.collectRollouts(context.Background(), 0)
	require.NoError(t, err)

	resumed := make([]*RecurrentRollout, 3)
	resumed[0] = collected[0]
	trainer.ResumeRollouts(resumed)

	_, err = trainer.collectRollouts(context.Background(), 1)
	require.NoError(t, err)

	for _, tr := range tracked {
		atomic.StoreInt32(tr.resetCount, 0)
	}

	// No ResumeRollouts call this time - every slot must be freshly
	// collected, even the one epoch 1 got from ResumeRollouts (only ever
	// supplied for that one call, not this one). Checks the pool's total
	// reset count rather than each env individually - work-stealing no
	// longer guarantees any particular env claims any particular slot
	// (see collectRecurrentRolloutsPool's own doc comment), so a single
	// fast env could legitimately claim more than one of the 3 slots
	// while another claims none, with the total still correctly at 3.
	_, err = trainer.collectRollouts(context.Background(), 2)
	require.NoError(t, err)

	var totalResets int32
	for _, tr := range tracked {
		totalResets += atomic.LoadInt32(tr.resetCount)
	}
	assert.EqualValues(t, 3, totalResets, "every slot must be freshly collected once ResumeRollouts' one-shot supply has been consumed")
}

// TestRecurrentResumeRolloutsPanicsOnWrongLength covers ResumeRollouts'
// documented contract: the slice it's handed must be exactly
// settings.RolloutSize long (nil for not-yet-collected), matching what
// collectRollouts itself always produces - anything else is a caller
// bug (e.g. reading back a cache written under a different RolloutSize)
// that must fail loudly, not silently collect the wrong episodes into
// the wrong slots.
func TestRecurrentResumeRolloutsPanicsOnWrongLength(t *testing.T) {
	trainer, _ := newRecurrentPoolTrainer(t, 3)

	assert.Panics(t, func() {
		trainer.ResumeRollouts(make([]*RecurrentRollout, 2))
	})
}
