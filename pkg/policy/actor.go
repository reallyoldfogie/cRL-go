package policy

import (
	"fmt"
	"math/rand/v2"

	"github.com/reallyoldfogie/cRL-go/pkg/mat"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

// Actor wraps a Params to provide a single entry point from an
// observation straight to a sampled action: NewActor once, then Act
// repeatedly, instead of a caller hand-driving NewInferenceNetwork,
// copying an observation into its Input, calling Graph.Forward, and
// sampling the result itself. See docs/plans/07-inference-api-and-action-masking.md.
type Actor struct {
	params *Params
}

// NewActor wraps params in an Actor. params is not copied: Act always
// reads its current values, so weight updates applied by a
// concurrently-running trainer are visible to later Act calls exactly
// as they already are to a freshly-built InferenceNetwork.
func NewActor(params *Params) (*Actor, error) {
	if params == nil {
		return nil, fmt.Errorf("policy: params must not be nil")
	}
	return &Actor{params: params}, nil
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
	net, err := NewInferenceNetwork(a.params)
	if err != nil {
		return 0, fmt.Errorf("policy: building inference network: %w", err)
	}

	copy(net.Input.Val.Data, obs.Values)
	net.Graph.Forward()

	return sampleMaskedAction(net.Output.Val, mask, rng)
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
	if mask == nil {
		return sampleAction(probs, rng), nil
	}
	if len(mask) != len(probs.Data) {
		return 0, fmt.Errorf("policy: mask length %d does not match action space %d", len(mask), len(probs.Data))
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
		return sampleAction(probs, rng), nil
	}
	if maskedSum <= 0 {
		return 0, fmt.Errorf("policy: no legal action: mask excludes every action with nonzero probability")
	}

	sample := rng.Float32()

	var cumulative float32
	lastAllowed := -1
	for i, allowed := range mask {
		if !allowed {
			continue
		}
		lastAllowed = i
		cumulative += probs.Data[i] / maskedSum
		if sample <= cumulative {
			return rl.Action(i), nil
		}
	}
	return rl.Action(lastAllowed), nil
}
