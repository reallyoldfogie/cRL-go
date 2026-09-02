package policy

import (
	"fmt"
	"math/rand/v2"
	"sync/atomic"

	"github.com/reallyoldfogie/cRL-go/pkg/mat"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

// Actor wraps a snapshot of a trained Params to provide a single entry
// point from an observation straight to a sampled action: NewActor
// once, then Act repeatedly, instead of a caller hand-driving
// NewInferenceNetwork, copying an observation into its Input, calling
// Graph.Forward, and sampling the result itself. See
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
		return nil, fmt.Errorf("policy: params must not be nil")
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
		return fmt.Errorf("policy: live params must not be nil")
	}
	a.params.Store(live.Snapshot())
	return nil
}

// Act builds a fresh InferenceNetwork over the Actor's Params, runs one
// forward pass over obs, and samples an action from the resulting
// distribution via sampleMaskedAction. mask is nil (unmasked) or a
// []bool of length ActionSpace(): a non-nil mask excludes disallowed
// actions and renormalizes over the remainder before sampling,
// returning an error if every action is masked out.
//
// Building a fresh InferenceNetwork per call matches the "cheap to
// build" assumption already documented on InferenceNetwork (a fresh one
// is already built per rollout-collection goroutine per epoch); if
// per-call allocation ever matters for a live decision loop's latency,
// that's a narrower, separately-justified optimization on top of this
// API, not a reason to avoid it here.
func (a *Actor) Act(obs rl.Observation, mask []bool, rng *rand.Rand) (rl.Action, error) {
	net, err := NewInferenceNetwork(a.params.Load())
	if err != nil {
		return 0, fmt.Errorf("policy: building inference network: %w", err)
	}

	copy(net.Input.Val.Data, obs.Values)
	net.Graph.Forward()

	return sampleMaskedAction(net.Output.Val, mask, rng)
}

// ActWithInfo behaves exactly like Act, but returns a full rl.Decision
// instead of only the sampled rl.Action, exposing the distribution
// that produced it (both before and after mask) for a caller that
// wants to audit why a decision was made, not only observe the
// outcome. Decision.HasValue is always false here: pkg/policy has no
// critic to report a value estimate from (see pkg/actorcritic.Actor's
// version of this method for that). See
// docs/plans/16-decision-auditing-and-explainability.md.
func (a *Actor) ActWithInfo(obs rl.Observation, mask []bool, rng *rand.Rand) (rl.Decision, error) {
	net, err := NewInferenceNetwork(a.params.Load())
	if err != nil {
		return rl.Decision{}, fmt.Errorf("policy: building inference network: %w", err)
	}

	copy(net.Input.Val.Data, obs.Values)
	net.Graph.Forward()

	action, raw, renormalized, err := sampleMaskedActionWithProbabilities(net.Output.Val, mask, rng)
	if err != nil {
		return rl.Decision{}, err
	}
	return rl.Decision{Action: action, Probabilities: renormalized, RawProbabilities: raw}, nil
}

// sampleAction and sampleMaskedAction mirror
// reinforce.SampleAction/reinforce.SampleMaskedAction exactly; see
// pkg/reinforce/episode.go for the canonical, directly-tested versions
// and their documentation. They are duplicated here, rather than
// imported, because pkg/reinforce already depends on this package
// (reinforce.collectTrajectory uses policy.Params and
// policy.NewInferenceNetwork), so importing pkg/reinforce back from
// here would create an import cycle — the same reason
// pkg/actorcritic/network.go's chain type is duplicated rather than
// shared with this package's.

func sampleAction(probs *mat.Matrix, rng *rand.Rand) rl.Action {
	sample := rng.Float32()

	var cumulative float32
	size := len(probs.Data)
	for i, p := range probs.Data {
		cumulative += p
		if sample <= cumulative {
			return rl.Action(i)
		}
	}
	return rl.Action(size - 1)
}

func sampleMaskedAction(probs *mat.Matrix, mask []bool, rng *rand.Rand) (rl.Action, error) {
	action, _, _, err := sampleMaskedActionWithProbabilities(probs, mask, rng)
	return action, err
}

// sampleMaskedActionWithProbabilities mirrors
// reinforce.SampleMaskedActionWithProbabilities exactly; see
// pkg/reinforce/episode.go for the canonical, directly-tested version
// and its documentation. Duplicated here for the same import-cycle
// reason sampleAction/sampleMaskedAction are (see above): raw is
// probs's own distribution; renormalized is the distribution actually
// sampled from. sampleMaskedAction is implemented in terms of this
// function.
func sampleMaskedActionWithProbabilities(probs *mat.Matrix, mask []bool, rng *rand.Rand) (action rl.Action, raw []float32, renormalized []float32, err error) {
	raw = append([]float32(nil), probs.Data...)

	if mask == nil {
		return sampleAction(probs, rng), raw, raw, nil
	}
	if len(mask) != len(probs.Data) {
		return 0, nil, nil, fmt.Errorf("policy: mask length %d does not match action space %d", len(mask), len(probs.Data))
	}

	allAllowed := true
	var maskedSum float32
	for i, allowed := range mask {
		if allowed {
			maskedSum += probs.Data[i]
		} else {
			allAllowed = false
		}
	}
	if allAllowed {
		return sampleAction(probs, rng), raw, raw, nil
	}
	if maskedSum <= 0 {
		return 0, nil, nil, fmt.Errorf("policy: no legal action: mask excludes every action with nonzero probability")
	}

	renormalized = make([]float32, len(probs.Data))
	sample := rng.Float32()

	var cumulative float32
	lastAllowed := -1
	chosen := -1
	for i, allowed := range mask {
		if !allowed {
			continue
		}
		lastAllowed = i
		p := probs.Data[i] / maskedSum
		renormalized[i] = p
		cumulative += p
		if chosen == -1 && sample <= cumulative {
			chosen = i
		}
	}
	if chosen == -1 {
		chosen = lastAllowed
	}
	return rl.Action(chosen), raw, renormalized, nil
}
