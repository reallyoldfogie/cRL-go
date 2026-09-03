package main

import (
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"
	"github.com/reallyoldfogie/cRL-go/pkg/config"
	"github.com/reallyoldfogie/cRL-go/pkg/decisionlog"
	"github.com/reallyoldfogie/cRL-go/pkg/gridworldenv"
	"github.com/reallyoldfogie/cRL-go/pkg/hierarchical"
	"github.com/reallyoldfogie/cRL-go/pkg/hierarchicalgridworld"
	"github.com/reallyoldfogie/cRL-go/pkg/policy"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestRunWithDecisionLogOutWritesOneRecordPerStep drives one short
// episode via run (the same entry point cmd/watch's main uses) against
// a freshly-initialized REINFORCE checkpoint, with -decision-log-out
// set, then reads the resulting log back and checks it matches what the
// episode actually did — see
// docs/plans/18-shared-decision-logging-format.md.
func TestRunWithDecisionLogOutWritesOneRecordPerStep(t *testing.T) {
	const gridSize = 9
	rng := rand.New(rand.NewPCG(1, 2))

	params := policy.NewParams(rng, gridworldenv.ObservationSize(gridSize), 8, gridworldenv.NumActions)
	checkpointPath := filepath.Join(t.TempDir(), "policy.json")
	environmentID := fmt.Sprintf("gridworld:%d", gridSize)
	require.NoError(t, policy.SaveFile(checkpointPath, params, environmentID, checkpoint.Metadata{}))

	decisionLogPath := filepath.Join(t.TempDir(), "decisions.jsonl")

	err := run([]string{
		"-env=gridworld",
		"-algo=reinforce",
		fmt.Sprintf("-grid-size=%d", gridSize),
		"-checkpoint=" + checkpointPath,
		"-episode-len=5",
		"-delay=0s",
		"-decision-log-out=" + decisionLogPath,
	})
	require.NoError(t, err)

	file, err := os.Open(decisionLogPath)
	require.NoError(t, err)
	defer file.Close()

	records, err := decisionlog.ReadAll(file)
	require.NoError(t, err)
	require.NotEmpty(t, records, "at least one step must have been recorded")
	assert.LessOrEqual(t, len(records), 5, "-episode-len=5 bounds the number of steps taken")

	for i, rec := range records {
		assert.Equal(t, i, rec.Step)
		assert.Len(t, rec.Observation, gridworldenv.ObservationSize(gridSize))
		assert.Len(t, rec.Decision.Probabilities, gridworldenv.NumActions)
		assert.False(t, rec.Decision.HasValue, "pkg/policy.Actor has no critic")
		assert.NotEmpty(t, rec.Render, "gridworldenv.Env.Render() must produce a non-empty snapshot")
		assert.Nil(t, rec.Extra, "the reinforce case reports no algorithm-specific extra fields")
	}
}

// TestRunWithDecisionLogOutRecordsHierarchicalExtraFields confirms the
// hierarchical case populates Record.Extra with the active subgoal on
// every step, and with the meta-controller's own rl.Decision exactly on
// the steps where a new subgoal was actually chosen — see
// docs/plans/18-shared-decision-logging-format.md.
func TestRunWithDecisionLogOutRecordsHierarchicalExtraFields(t *testing.T) {
	const (
		gridSize        = 36
		numSubgoals     = 3
		subgoalInterval = 3
		episodeLen      = 6
	)
	settings := config.Default()
	settings.GridSize = gridSize

	envFactory := func(rng *rand.Rand) (rl.Environment, error) {
		env, err := hierarchicalgridworld.New(gridSize, rng)
		if err != nil {
			return nil, err
		}
		return hierarchicalgridworld.NewAdapter(env), nil
	}

	trainer, err := hierarchical.New(settings, hierarchical.Config{
		NumSubgoals:      numSubgoals,
		SubgoalInterval:  subgoalInterval,
		MetaHiddenSize:   8,
		SubHiddenSize:    8,
		MetaLearningRate: 0.01,
		SubLearningRate:  0.01,
	}, envFactory, nil)
	require.NoError(t, err)

	checkpointPath := filepath.Join(t.TempDir(), "hierarchical.json")
	environmentID := fmt.Sprintf("hierarchicalgridworld:%d", gridSize)
	require.NoError(t, trainer.SaveFile(checkpointPath, environmentID, checkpoint.Metadata{}))

	decisionLogPath := filepath.Join(t.TempDir(), "decisions.jsonl")

	require.NoError(t, run([]string{
		"-env=hierarchicalgridworld",
		"-algo=hierarchical",
		fmt.Sprintf("-grid-size=%d", gridSize),
		"-checkpoint=" + checkpointPath,
		fmt.Sprintf("-num-subgoals=%d", numSubgoals),
		fmt.Sprintf("-subgoal-interval=%d", subgoalInterval),
		fmt.Sprintf("-episode-len=%d", episodeLen),
		"-delay=0s",
		"-decision-log-out=" + decisionLogPath,
	}))

	file, err := os.Open(decisionLogPath)
	require.NoError(t, err)
	defer file.Close()

	records, err := decisionlog.ReadAll(file)
	require.NoError(t, err)
	require.NotEmpty(t, records)

	metaDecisionSteps := 0
	for i, rec := range records {
		require.NotNil(t, rec.Extra, "step %d: hierarchical must always report Extra", i)
		assert.Contains(t, rec.Extra, "active_subgoal", "step %d", i)

		if i%subgoalInterval == 0 {
			assert.Equal(t, true, rec.Extra["meta_decision_made"], "step %d: a meta-decision is made at episode start and every %d steps", i, subgoalInterval)
			assert.Contains(t, rec.Extra, "meta_decision", "step %d", i)
			metaDecisionSteps++
		} else {
			assert.NotContains(t, rec.Extra, "meta_decision_made", "step %d: no new subgoal was chosen this step", i)
		}
	}
	assert.Positive(t, metaDecisionSteps, "at least the episode's first step must be a meta-decision step")
}
