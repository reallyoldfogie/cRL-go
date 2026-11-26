package ppo

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestRecurrentSetUpdateProgressReportsEveryMinibatch covers the
// 2026-11-25 feature backing a live spectating progress bar for the
// PPO-update phase: SetUpdateProgress's callback must fire once per
// minibatch Adam update, with completed counting up 1..total in order
// and total already correct (matching the epoch's own, separately
// tracked UpdateCount) from the very first call - not just known in
// hindsight once the epoch is over.
func TestRecurrentSetUpdateProgressReportsEveryMinibatch(t *testing.T) {
	trainer, _ := newRecurrentPoolTrainer(t, 3)

	var completedSeq []int
	var totals []int
	trainer.SetUpdateProgress(func(epoch, completed, total int) {
		assert.Equal(t, 0, epoch)
		completedSeq = append(completedSeq, completed)
		totals = append(totals, total)
	})

	stats, err := trainer.RunEpoch(context.Background(), 0)
	require.NoError(t, err)
	require.NotEmpty(t, completedSeq, "expected SetUpdateProgress's callback to fire at least once")

	for i, completed := range completedSeq {
		assert.Equal(t, i+1, completed, "completed must count up 1..total in order")
	}
	assert.Equal(t, stats.UpdateCount, completedSeq[len(completedSeq)-1], "the final completed value must equal the epoch's own UpdateCount")
	for _, total := range totals {
		assert.Equal(t, stats.UpdateCount, total, "total must already be correct on every call, not just in hindsight")
	}
}

// TestRecurrentUpdateProgressNilIsSafe confirms RunEpoch never assumes a
// progress callback is registered - the common case, every epoch before
// this feature existed.
func TestRecurrentUpdateProgressNilIsSafe(t *testing.T) {
	trainer, _ := newRecurrentPoolTrainer(t, 3)
	_, err := trainer.RunEpoch(context.Background(), 0)
	require.NoError(t, err)
}
