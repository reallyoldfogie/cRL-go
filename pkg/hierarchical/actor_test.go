package hierarchical

import (
	"math/rand/v2"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	testActorNumSubgoals     = 3
	testActorSubgoalInterval = 4
	testActorObsSize         = 5
	testActorActionSpace     = 2
)

// newTestActorParams builds a fresh, deterministic meta/sub Params set
// for testActorNumSubgoals subgoals, so tests can construct
// independent Actors that behave identically when built from the same
// seed.
func newTestActorParams(seed uint64) (*actorcritic.Params, map[Subgoal]*actorcritic.Params) {
	rng := rand.New(rand.NewPCG(seed, seed+1))
	meta := actorcritic.NewParams(rng, testActorObsSize, 4, testActorNumSubgoals)
	subs := make(map[Subgoal]*actorcritic.Params, testActorNumSubgoals)
	for i := range testActorNumSubgoals {
		subs[Subgoal(i)] = actorcritic.NewParams(rng, testActorObsSize+testActorNumSubgoals, 4, testActorActionSpace)
	}
	return meta, subs
}

func newTestActor(t *testing.T, seed uint64) *Actor {
	t.Helper()
	meta, subs := newTestActorParams(seed)
	actor, err := NewActor(meta, subs, testActorNumSubgoals, testActorSubgoalInterval)
	require.NoError(t, err)
	return actor
}

func TestActorMakesMetaDecisionOnFirstCallAndEveryInterval(t *testing.T) {
	actor := newTestActor(t, 1)
	obs := rl.Observation{Values: make([]float32, testActorObsSize)}
	rng := rand.New(rand.NewPCG(10, 20))

	for step := range testActorSubgoalInterval * 3 {
		decision, err := actor.ActWithInfo(obs, rng)
		require.NoError(t, err)

		wantMetaDecision := step%testActorSubgoalInterval == 0
		assert.Equal(t, wantMetaDecision, decision.MetaDecisionMade, "step %d", step)
		assert.GreaterOrEqual(t, int(decision.ActiveSubgoal), 0)
		assert.Less(t, int(decision.ActiveSubgoal), testActorNumSubgoals)
		assert.GreaterOrEqual(t, int(decision.SubDecision.Action), 0)
		assert.Less(t, int(decision.SubDecision.Action), testActorActionSpace)
	}
}

func TestActorResetForcesFreshMetaDecision(t *testing.T) {
	actor := newTestActor(t, 2)
	obs := rl.Observation{Values: make([]float32, testActorObsSize)}
	rng := rand.New(rand.NewPCG(10, 20))

	_, err := actor.ActWithInfo(obs, rng)
	require.NoError(t, err)
	midInterval, err := actor.ActWithInfo(obs, rng)
	require.NoError(t, err)
	require.False(t, midInterval.MetaDecisionMade, "second call within the interval should not re-decide")

	actor.Reset()

	decision, err := actor.ActWithInfo(obs, rng)
	require.NoError(t, err)
	assert.True(t, decision.MetaDecisionMade, "Reset must force a fresh meta-decision on the next call")
}

// TestActMatchesActWithInfo confirms Act introduces no drift of its
// own: two independently-constructed but identically-seeded Actors
// must sample the same action for the same rng draw sequence.
func TestActMatchesActWithInfo(t *testing.T) {
	actorForAct := newTestActor(t, 3)
	actorForInfo := newTestActor(t, 3)
	obs := rl.Observation{Values: make([]float32, testActorObsSize)}

	for step := range testActorSubgoalInterval * 2 {
		action, err := actorForAct.Act(obs, rand.New(rand.NewPCG(uint64(step), uint64(step+1))))
		require.NoError(t, err)

		decision, err := actorForInfo.ActWithInfo(obs, rand.New(rand.NewPCG(uint64(step), uint64(step+1))))
		require.NoError(t, err)

		assert.Equal(t, action, decision.SubDecision.Action)
	}
}

func TestNewActorRejectsNonPositiveNumSubgoals(t *testing.T) {
	meta, subs := newTestActorParams(4)
	_, err := NewActor(meta, subs, 0, testActorSubgoalInterval)
	assert.Error(t, err)
}

func TestNewActorRejectsNonPositiveSubgoalInterval(t *testing.T) {
	meta, subs := newTestActorParams(5)
	_, err := NewActor(meta, subs, testActorNumSubgoals, 0)
	assert.Error(t, err)
}

func TestNewActorRejectsMissingSubPolicyParams(t *testing.T) {
	meta, subs := newTestActorParams(6)
	delete(subs, Subgoal(0))

	_, err := NewActor(meta, subs, testActorNumSubgoals, testActorSubgoalInterval)
	assert.Error(t, err)
}
