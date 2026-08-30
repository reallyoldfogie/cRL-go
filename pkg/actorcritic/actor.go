package actorcritic

import (
	"fmt"
	"math/rand/v2"
	"sync/atomic"

	"github.com/reallyoldfogie/cRL-go/pkg/reinforce"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

// Actor wraps a snapshot of a trained Params to provide a single entry
// point from an observation straight to a sampled action, mirroring
// policy.Actor over pkg/policy.Params. Unlike policy.Actor, Act here can
// call reinforce.SampleMaskedAction directly instead of duplicating it,
// since pkg/reinforce does not depend on this package (only on
// pkg/policy), so there is no import cycle to avoid. See
// docs/plans/07-inference-api-and-action-masking.md.
//
// Act always infers against Actor's own private snapshot (see
// Params.Snapshot), never against a live Params directly — even if the
// Params passed to NewActor/Refresh is concurrently being trained (see
// Params.Lock), Act never blocks on or races with that training.
// Weight updates only become visible to Act once Refresh is called
// again; see docs/plans/09-concurrency-safe-live-inference.md for why
// this snapshot-and-explicit-refresh design was chosen over reading the
// live Params on every call.
type Actor struct {
	params atomic.Pointer[Params]
}

// NewActor wraps a snapshot of params in an Actor (see Params.Snapshot
// and the Actor doc comment for why a snapshot, not params itself, is
// stored).
func NewActor(params *Params) (*Actor, error) {
	if params == nil {
		return nil, fmt.Errorf("actorcritic: params must not be nil")
	}
	actor := &Actor{}
	actor.params.Store(params.Snapshot())
	return actor, nil
}

// Refresh replaces the Actor's snapshot with a fresh copy of live's
// current weights (see Params.Snapshot), so subsequent Act calls reflect
// any training applied to live since the last NewActor/Refresh call.
// Safe to call concurrently with Act and with a trainer applying
// gradient updates to live (guarded by live's own lock — see
// Params.Lock).
func (a *Actor) Refresh(live *Params) error {
	if live == nil {
		return fmt.Errorf("actorcritic: live params must not be nil")
	}
	a.params.Store(live.Snapshot())
	return nil
}

// Act builds a fresh InferenceNetwork over the Actor's Params, runs one
// forward pass over obs, and samples an action from the policy head's
// resulting distribution via reinforce.SampleMaskedAction. ValueOutput
// is not consulted here, matching how a value estimate isn't itself
// part of an action decision. mask is nil (unmasked) or a []bool of
// length ActionSpace(); see reinforce.SampleMaskedAction for its exact
// semantics.
//
// Building a fresh InferenceNetwork per call matches the "cheap to
// build" assumption already documented on InferenceNetwork; see
// policy.Actor.Act's doc comment for the same rationale.
func (a *Actor) Act(obs rl.Observation, mask []bool, rng *rand.Rand) (rl.Action, error) {
	net, err := NewInferenceNetwork(a.params.Load())
	if err != nil {
		return 0, fmt.Errorf("actorcritic: building inference network: %w", err)
	}

	copy(net.Input.Val.Data, obs.Values)
	net.Graph.Forward()

	return reinforce.SampleMaskedAction(net.PolicyOutput.Val, mask, rng)
}
