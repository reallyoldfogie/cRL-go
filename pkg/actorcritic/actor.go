package actorcritic

import (
	"fmt"
	"math/rand/v2"

	"github.com/reallyoldfogie/cRL-go/pkg/reinforce"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

// Actor wraps a Params to provide a single entry point from an
// observation straight to a sampled action, mirroring policy.Actor over
// pkg/policy.Params. Unlike policy.Actor, Act here can call
// reinforce.SampleMaskedAction directly instead of duplicating it,
// since pkg/reinforce does not depend on this package (only on
// pkg/policy), so there is no import cycle to avoid. See
// docs/plans/07-inference-api-and-action-masking.md.
type Actor struct {
	params *Params
}

// NewActor wraps params in an Actor. params is not copied: Act always
// reads its current values, so weight updates applied by a
// concurrently-running trainer are visible to later Act calls exactly
// as they already are to a freshly-built InferenceNetwork.
func NewActor(params *Params) (*Actor, error) {
	if params == nil {
		return nil, fmt.Errorf("actorcritic: params must not be nil")
	}
	return &Actor{params: params}, nil
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
	net, err := NewInferenceNetwork(a.params)
	if err != nil {
		return 0, fmt.Errorf("actorcritic: building inference network: %w", err)
	}

	copy(net.Input.Val.Data, obs.Values)
	net.Graph.Forward()

	return reinforce.SampleMaskedAction(net.PolicyOutput.Val, mask, rng)
}
