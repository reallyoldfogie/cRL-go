package actorcritic

import (
	"math/rand/v2"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/reinforce"
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
// NewInferenceNetwork/Forward/reinforce.SampleAction would.
func TestActorActMatchesManualInferenceAndSampling(t *testing.T) {
	params := NewParams(rand.New(rand.NewPCG(31, 32)), 6, 4, 3)
	obs := testObservation()

	actor, err := NewActor(params)
	require.NoError(t, err)

	action, err := actor.Act(obs, nil, rand.New(rand.NewPCG(1, 2)))
	require.NoError(t, err)

	net, err := NewInferenceNetwork(params)
	require.NoError(t, err)
	copy(net.Input.Val.Data, obs.Values)
	net.Graph.Forward()

	wantAction := reinforce.SampleAction(net.PolicyOutput.Val, rand.New(rand.NewPCG(1, 2)))

	assert.Equal(t, wantAction, action)
}

// TestActorActWithMaskExcludesDisallowedActions confirms a mask that
// allows only one action always yields that action, regardless of the
// rng draw.
func TestActorActWithMaskExcludesDisallowedActions(t *testing.T) {
	params := NewParams(rand.New(rand.NewPCG(33, 34)), 6, 4, 3)
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
	params := NewParams(rand.New(rand.NewPCG(35, 36)), 6, 4, 3)

	actor, err := NewActor(params)
	require.NoError(t, err)

	_, err = actor.Act(testObservation(), []bool{false, false, false}, rand.New(rand.NewPCG(1, 2)))
	assert.Error(t, err)
}

func TestNewActorRejectsNilParams(t *testing.T) {
	_, err := NewActor(nil)
	assert.Error(t, err)
}
