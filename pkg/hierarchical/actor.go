package hierarchical

import (
	"fmt"
	"math/rand/v2"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

// Decision is the result of one Actor.ActWithInfo call: the sub-policy
// decision that actually determined the environment action (always
// present), plus the meta-controller's own decision on any step it was
// consulted. MetaDecisionMade is true only on the step a new subgoal
// was actually chosen (episode start, or every SubgoalInterval steps
// thereafter — see Actor.Act); MetaDecision is the zero rl.Decision on
// every other step, since the meta-controller wasn't run then.
type Decision struct {
	SubDecision      rl.Decision
	ActiveSubgoal    Subgoal
	MetaDecision     rl.Decision
	MetaDecisionMade bool
}

// Actor drives live inference for a hierarchical policy: it wraps a
// meta-controller actorcritic.Actor and one sub-policy actorcritic.Actor
// per Subgoal, reproducing collectHierarchicalTrajectory's decision
// process (pkg/hierarchical/rollout.go) one step at a time instead of
// for a whole pre-collected episode — see
// docs/plans/16-decision-auditing-and-explainability.md and
// docs/plans/15-agent-and-training-visualization.md, both of which
// motivated this.
//
// Unlike policy.Actor/actorcritic.Actor, Actor is not a pure function
// of an Observation: it carries per-episode state (which subgoal is
// currently active, and how many steps remain until the next
// meta-decision), since "the active subgoal" only makes sense across a
// sequence of steps, not for one observation in isolation. Call Reset
// once at the start of every episode, before the first Act/ActWithInfo
// call of that episode.
//
// Actor takes NumSubgoals and SubgoalInterval directly, rather than a
// full Config, deliberately: Config's other fields (hidden
// sizes/learning rates) describe how to train fresh networks and are
// irrelevant to driving already-trained ones, so requiring a full,
// separately-validated Config here would force a caller to supply
// meaningless values just to satisfy Config.Validate.
type Actor struct {
	metaActor *actorcritic.Actor
	subActors map[Subgoal]*actorcritic.Actor

	numSubgoals     int
	subgoalInterval int

	activeSubgoal      Subgoal
	stepsSinceDecision int
}

// NewActor builds an Actor from a previously-trained meta-controller
// and every sub-policy's actorcritic.Params (e.g. from
// Trainer.Params() or LoadFile), snapshotting each into its own
// actorcritic.Actor (see actorcritic.NewActor). subParams must have
// exactly one entry for every Subgoal in [0, numSubgoals).
func NewActor(metaParams *actorcritic.Params, subParams map[Subgoal]*actorcritic.Params, numSubgoals, subgoalInterval int) (*Actor, error) {
	if numSubgoals <= 0 {
		return nil, fmt.Errorf("hierarchical: num_subgoals must be positive, got %d", numSubgoals)
	}
	if subgoalInterval <= 0 {
		return nil, fmt.Errorf("hierarchical: subgoal_interval must be positive, got %d", subgoalInterval)
	}

	metaActor, err := actorcritic.NewActor(metaParams)
	if err != nil {
		return nil, fmt.Errorf("hierarchical: building meta-controller actor: %w", err)
	}

	subActors := make(map[Subgoal]*actorcritic.Actor, numSubgoals)
	for i := range numSubgoals {
		subgoal := Subgoal(i)
		params, ok := subParams[subgoal]
		if !ok {
			return nil, fmt.Errorf("hierarchical: missing sub-policy params for subgoal %d", subgoal)
		}
		subActor, err := actorcritic.NewActor(params)
		if err != nil {
			return nil, fmt.Errorf("hierarchical: building sub-policy actor for subgoal %d: %w", subgoal, err)
		}
		subActors[subgoal] = subActor
	}

	return &Actor{
		metaActor:       metaActor,
		subActors:       subActors,
		numSubgoals:     numSubgoals,
		subgoalInterval: subgoalInterval,
	}, nil
}

// Reset clears Actor's per-episode state, so the next Act/ActWithInfo
// call always makes a fresh meta-decision instead of continuing to use
// whatever subgoal was active at the end of a previous episode.
func (a *Actor) Reset() {
	a.stepsSinceDecision = 0
}

// Act behaves like ActWithInfo, returning only the sampled rl.Action
// that should be applied to the environment (the sub-policy's, since
// that's what actually acts) for callers that don't need the full
// Decision.
func (a *Actor) Act(obs rl.Observation, rng *rand.Rand) (rl.Action, error) {
	decision, err := a.ActWithInfo(obs, rng)
	if err != nil {
		return 0, err
	}
	return decision.SubDecision.Action, nil
}

// ActWithInfo mirrors collectHierarchicalTrajectory's live decision
// process: obs is the environment's raw, un-augmented observation. If
// this is the first call since Reset, or SubgoalInterval steps have
// elapsed since the last meta-decision, the meta-controller chooses a
// new subgoal from obs first (MetaDecisionMade=true, MetaDecision set).
// The currently-active subgoal's sub-policy then chooses the actual
// rl.Action from obs augmented with that subgoal (see
// augmentObservation) — this happens every call, regardless of whether
// a new subgoal was just chosen this step or several steps ago.
func (a *Actor) ActWithInfo(obs rl.Observation, rng *rand.Rand) (Decision, error) {
	var metaDecision rl.Decision
	metaDecisionMade := false

	if a.stepsSinceDecision == 0 {
		decision, err := a.metaActor.ActWithInfo(obs, nil, rng)
		if err != nil {
			return Decision{}, fmt.Errorf("hierarchical: selecting subgoal: %w", err)
		}
		a.activeSubgoal = Subgoal(decision.Action)
		metaDecision = decision
		metaDecisionMade = true
	}

	subActor, ok := a.subActors[a.activeSubgoal]
	if !ok {
		return Decision{}, fmt.Errorf("hierarchical: no sub-policy actor for subgoal %d", a.activeSubgoal)
	}

	subDecision, err := subActor.ActWithInfo(augmentObservation(obs, a.activeSubgoal, a.numSubgoals), nil, rng)
	if err != nil {
		return Decision{}, err
	}

	a.stepsSinceDecision++
	if a.stepsSinceDecision >= a.subgoalInterval {
		a.stepsSinceDecision = 0
	}

	return Decision{
		SubDecision:      subDecision,
		ActiveSubgoal:    a.activeSubgoal,
		MetaDecision:     metaDecision,
		MetaDecisionMade: metaDecisionMade,
	}, nil
}
