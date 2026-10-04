package actorcritic

import (
	"math/rand/v2"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/reinforce"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func recurrentActorTestObservation() rl.Observation {
	return rl.Observation{Values: []float32{0.1, -0.2, 0.3, 0.4, -0.5}}
}

func TestNewRecurrentActorRejectsNilParams(t *testing.T) {
	_, err := NewRecurrentActor(nil)
	assert.Error(t, err)
}

// TestRecurrentActorActCarriesStateAcrossCalls confirms the whole reason
// RecurrentActor can't rebuild a fresh network per call the way Actor
// does: feeding the same observation twice in a row must give a
// different result the second time, since the carried (h, c) differs.
func TestRecurrentActorActCarriesStateAcrossCalls(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	params := NewRecurrentParams(rng, 5, 4, 3, 1)

	actor, err := NewRecurrentActor(params)
	require.NoError(t, err)

	decision1, err := actor.ActWithInfo(recurrentActorTestObservation(), nil, rand.New(rand.NewPCG(9, 9)))
	require.NoError(t, err)
	decision2, err := actor.ActWithInfo(recurrentActorTestObservation(), nil, rand.New(rand.NewPCG(9, 9)))
	require.NoError(t, err)

	assert.NotEqual(t, decision1.RawProbabilities, decision2.RawProbabilities, "the second call must see different carried state than the first")
}

// TestRecurrentActorResetZeroesCarriedState confirms Reset actually
// restores the "fresh episode" behavior: after Reset, the very next Act
// call must reproduce the SAME output NewRecurrentActor's own first
// call produced (both start from zeroed (h, c)).
func TestRecurrentActorResetZeroesCarriedState(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	params := NewRecurrentParams(rng, 5, 4, 3, 1)
	obs := recurrentActorTestObservation()

	fresh, err := NewRecurrentActor(params)
	require.NoError(t, err)
	want, err := fresh.ActWithInfo(obs, nil, rand.New(rand.NewPCG(9, 9)))
	require.NoError(t, err)

	reused, err := NewRecurrentActor(params)
	require.NoError(t, err)
	_, err = reused.ActWithInfo(obs, nil, rand.New(rand.NewPCG(1, 1))) // advance state with an unrelated step
	require.NoError(t, err)
	reused.Reset()

	got, err := reused.ActWithInfo(obs, nil, rand.New(rand.NewPCG(9, 9)))
	require.NoError(t, err)
	assert.Equal(t, want.RawProbabilities, got.RawProbabilities, "Reset must restore zeroed carried state exactly")
	assert.Equal(t, want.Value, got.Value)
}

// TestRecurrentActorActMatchesManualRecurrentInferenceNetwork confirms
// Act/ActWithInfo introduce no drift of their own relative to hand-
// driving a RecurrentInferenceNetwork the same way pkg/ppo's rollout
// collection does — mirroring the non-recurrent
// TestActorActMatchesManualInferenceAndSampling.
func TestRecurrentActorActMatchesManualRecurrentInferenceNetwork(t *testing.T) {
	params := NewRecurrentParams(rand.New(rand.NewPCG(31, 32)), 5, 4, 3, 1)
	obs := recurrentActorTestObservation()

	actor, err := NewRecurrentActor(params)
	require.NoError(t, err)
	action, err := actor.Act(obs, nil, rand.New(rand.NewPCG(1, 2)))
	require.NoError(t, err)

	net, err := NewRecurrentInferenceNetwork(params)
	require.NoError(t, err)
	copy(net.Input.Val.Data, obs.Values)
	net.Graph.Forward()
	wantAction := reinforce.SampleAction(net.PolicyOutput.Val, rand.New(rand.NewPCG(1, 2)))

	assert.Equal(t, wantAction, action)
}

func TestRecurrentActorRejectsAllFalseMask(t *testing.T) {
	params := NewRecurrentParams(rand.New(rand.NewPCG(35, 36)), 5, 4, 3, 1)
	actor, err := NewRecurrentActor(params)
	require.NoError(t, err)

	_, err = actor.Act(recurrentActorTestObservation(), []bool{false, false, false}, rand.New(rand.NewPCG(1, 2)))
	assert.Error(t, err)
}

func TestRecurrentActorRefreshRejectsNilParams(t *testing.T) {
	params := NewRecurrentParams(rand.New(rand.NewPCG(37, 38)), 5, 4, 3, 1)
	actor, err := NewRecurrentActor(params)
	require.NoError(t, err)

	assert.Error(t, actor.Refresh(nil))
}
