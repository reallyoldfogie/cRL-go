package policy

import (
	"math/rand/v2"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func testObservation() rl.Observation {
	return rl.Observation{Values: []float32{0.1, -0.2, 0.3, 0.4, -0.5, 0.6}}
}

// TestActorActMatchesManualInferenceAndSampling confirms Act introduces
// no randomness or drift of its own: given the same Params, obs, and
// rng state, it returns exactly the action a caller hand-driving
// NewInferenceNetwork/Forward/sampleAction would.
func TestActorActMatchesManualInferenceAndSampling(t *testing.T) {
	params := NewParams(rand.New(rand.NewPCG(21, 22)), 6, 4, 3)
	obs := testObservation()

	actor, err := NewActor(params)
	require.NoError(t, err)

	action, err := actor.Act(obs, nil, rand.New(rand.NewPCG(1, 2)))
	require.NoError(t, err)

	net, err := NewInferenceNetwork(params)
	require.NoError(t, err)
	copy(net.Input.Val.Data, obs.Values)
	net.Graph.Forward()

	wantAction := sampleAction(net.Output.Val, rand.New(rand.NewPCG(1, 2)))

	assert.Equal(t, wantAction, action)
}

// TestActorActWithMaskExcludesDisallowedActions confirms a mask that
// allows only one action always yields that action, regardless of the
// rng draw.
func TestActorActWithMaskExcludesDisallowedActions(t *testing.T) {
	params := NewParams(rand.New(rand.NewPCG(23, 24)), 6, 4, 3)
	obs := testObservation()
	mask := []bool{false, true, false}

	actor, err := NewActor(params)
	require.NoError(t, err)

	for seed := range 20 {
		action, err := actor.Act(obs, mask, rand.New(rand.NewPCG(uint64(seed), uint64(seed+1))))
		require.NoError(t, err)
		assert.Equal(t, rl.Action(1), action)
	}
}

func TestActorActRejectsAllFalseMask(t *testing.T) {
	params := NewParams(rand.New(rand.NewPCG(25, 26)), 6, 4, 3)

	actor, err := NewActor(params)
	require.NoError(t, err)

	_, err = actor.Act(testObservation(), []bool{false, false, false}, rand.New(rand.NewPCG(1, 2)))
	assert.Error(t, err)
}

func TestNewActorRejectsNilParams(t *testing.T) {
	_, err := NewActor(nil)
	assert.Error(t, err)
}

// TestActorActWithInfoMatchesAct confirms ActWithInfo samples the same
// action as Act for the same Params, obs, and rng state — the richer
// return value must not introduce any drift of its own.
func TestActorActWithInfoMatchesAct(t *testing.T) {
	params := NewParams(rand.New(rand.NewPCG(41, 42)), 6, 4, 3)
	obs := testObservation()

	actor, err := NewActor(params)
	require.NoError(t, err)

	wantAction, err := actor.Act(obs, nil, rand.New(rand.NewPCG(1, 2)))
	require.NoError(t, err)

	decision, err := actor.ActWithInfo(obs, nil, rand.New(rand.NewPCG(1, 2)))
	require.NoError(t, err)
	assert.Equal(t, wantAction, decision.Action)
}

// TestActorActWithInfoReportsNoValue confirms pkg/policy's Actor, which
// has no critic, always reports HasValue=false.
func TestActorActWithInfoReportsNoValue(t *testing.T) {
	params := NewParams(rand.New(rand.NewPCG(43, 44)), 6, 4, 3)
	actor, err := NewActor(params)
	require.NoError(t, err)

	decision, err := actor.ActWithInfo(testObservation(), nil, rand.New(rand.NewPCG(1, 2)))
	require.NoError(t, err)
	assert.False(t, decision.HasValue)
	assert.Equal(t, float32(0), decision.Value)
}

// TestActorActWithInfoRenormalizesOverMaskedActions confirms
// Probabilities is renormalized over only the allowed actions, while
// RawProbabilities preserves the policy's unmasked output.
func TestActorActWithInfoRenormalizesOverMaskedActions(t *testing.T) {
	params := NewParams(rand.New(rand.NewPCG(45, 46)), 6, 4, 3)
	mask := []bool{false, true, false}

	actor, err := NewActor(params)
	require.NoError(t, err)

	decision, err := actor.ActWithInfo(testObservation(), mask, rand.New(rand.NewPCG(1, 2)))
	require.NoError(t, err)

	require.Len(t, decision.Probabilities, 3)
	require.Len(t, decision.RawProbabilities, 3)
	assert.Equal(t, rl.Action(1), decision.Action)
	assert.Equal(t, float32(1), decision.Probabilities[1], "the only allowed action must renormalize to probability 1")
	assert.Equal(t, float32(0), decision.Probabilities[0])
	assert.Equal(t, float32(0), decision.Probabilities[2])
	assert.NotEqual(t, decision.Probabilities, decision.RawProbabilities, "raw must reflect the unmasked distribution")
}
