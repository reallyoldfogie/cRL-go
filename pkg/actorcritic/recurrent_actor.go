package actorcritic

import (
	"fmt"
	"math/rand/v2"
	"sync/atomic"

	"github.com/reallyoldfogie/cRL-go/pkg/reinforce"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

// RecurrentActor mirrors Actor (see actor.go) for a RecurrentParams, with
// one structural difference forced by recurrence itself: Actor rebuilds
// a fresh InferenceNetwork on every Act call (cheap, and correct, since
// it's stateless) — a RecurrentActor cannot do that, since doing so
// would reset the LSTM's (h, c) to zero every single step, destroying
// the whole point of carrying memory across an episode. So
// RecurrentActor holds one persistent *RecurrentInferenceNetwork
// instead, built once and reused for every Act/ActWithInfo call across
// an episode, with Reset explicitly zeroing its carried state at the
// start of the next one.
//
// Not safe for concurrent use by multiple goroutines evaluating
// different episodes at once — unlike Actor (stateless per call, so
// trivially shareable), a RecurrentActor's carried (h, c) belongs to
// exactly one in-progress episode. A caller evaluating several episodes
// concurrently needs one RecurrentActor per concurrent episode, mirroring
// how pkg/ppo's own recurrent rollout collection builds one
// RecurrentInferenceNetwork per trajectory rather than sharing one.
type RecurrentActor struct {
	params atomic.Pointer[RecurrentParams]
	net    *RecurrentInferenceNetwork
}

// NewRecurrentActor wraps a snapshot of params in a RecurrentActor (see
// Params.Snapshot and Actor's own doc comment for why a snapshot, not
// params itself, is stored), building its one persistent
// RecurrentInferenceNetwork immediately, with (h, c) starting at zero —
// equivalent to calling Reset right after construction.
func NewRecurrentActor(params *RecurrentParams) (*RecurrentActor, error) {
	if params == nil {
		return nil, fmt.Errorf("actorcritic: params must not be nil")
	}
	a := &RecurrentActor{}
	a.params.Store(params.Snapshot())
	net, err := NewRecurrentInferenceNetwork(a.params.Load())
	if err != nil {
		return nil, fmt.Errorf("actorcritic: building recurrent inference network: %w", err)
	}
	a.net = net
	return a, nil
}

// Refresh replaces the Actor's snapshot with a fresh copy of live's
// current weights, mirroring Actor.Refresh — but note a RecurrentActor-
// specific wrinkle Actor doesn't have: the carried (h, c) was computed
// under whatever weights were live at the time of each prior Act call,
// not necessarily live's weights as of this Refresh call. Refreshing
// mid-episode means the next step's forward pass mixes old-weight-
// computed state with newly-refreshed weights — a narrow version of the
// same old-params-vs-current-params staleness PPO already tolerates
// elsewhere in this codebase (e.g. OldLogProb), not resolved further
// here. Rebuilds the persistent network over the new snapshot, but
// deliberately does not touch HPrevIn/CPrevIn — only Reset clears
// carried state.
func (a *RecurrentActor) Refresh(live *RecurrentParams) error {
	if live == nil {
		return fmt.Errorf("actorcritic: live params must not be nil")
	}
	snapshot := live.Snapshot()
	net, err := NewRecurrentInferenceNetwork(snapshot)
	if err != nil {
		return fmt.Errorf("actorcritic: building recurrent inference network: %w", err)
	}
	copy(net.HPrevIn.Val.Data, a.net.HPrevIn.Val.Data)
	copy(net.CPrevIn.Val.Data, a.net.CPrevIn.Val.Data)

	a.params.Store(snapshot)
	a.net = net
	return nil
}

// Reset zeroes the Actor's carried (h, c) — call this once at the start
// of every new episode. An Actor used across multiple episodes without
// calling Reset between them would incorrectly carry the previous
// episode's final state into the next one.
func (a *RecurrentActor) Reset() {
	a.net.HPrevIn.Val.Clear()
	a.net.CPrevIn.Val.Clear()
}

// Act builds on the Actor's persistent RecurrentInferenceNetwork, runs
// one forward pass over obs using whatever (h, c) is currently carried,
// samples an action from the resulting policy distribution, and threads
// the step's HOut/COut into the next call's HPrevIn/CPrevIn — see
// Actor.Act's own doc comment for the masking/sampling behavior this
// otherwise matches exactly.
func (a *RecurrentActor) Act(obs rl.Observation, mask []bool, rng *rand.Rand) (rl.Action, error) {
	net := a.net
	copy(net.Input.Val.Data, obs.Values)
	net.Graph.Forward()

	action, err := reinforce.SampleMaskedAction(net.PolicyOutput.Val, mask, rng)
	if err != nil {
		return 0, err
	}
	copy(net.HPrevIn.Val.Data, net.HOut.Val.Data)
	copy(net.CPrevIn.Val.Data, net.COut.Val.Data)
	return action, nil
}

// ActWithInfo behaves exactly like Act, but returns a full rl.Decision —
// see Actor.ActWithInfo's own doc comment, which this otherwise matches
// exactly.
func (a *RecurrentActor) ActWithInfo(obs rl.Observation, mask []bool, rng *rand.Rand) (rl.Decision, error) {
	net := a.net
	copy(net.Input.Val.Data, obs.Values)
	net.Graph.Forward()

	action, raw, renormalized, err := reinforce.SampleMaskedActionWithProbabilities(net.PolicyOutput.Val, mask, rng)
	if err != nil {
		return rl.Decision{}, err
	}
	decision := rl.Decision{
		Action:           action,
		Probabilities:    renormalized,
		RawProbabilities: raw,
		Value:            net.ValueOutput.Val.Data[0],
		HasValue:         true,
	}
	copy(net.HPrevIn.Val.Data, net.HOut.Val.Data)
	copy(net.CPrevIn.Val.Data, net.COut.Val.Data)
	return decision, nil
}
