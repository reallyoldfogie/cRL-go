package ppo

import (
	"fmt"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/autograd"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

// LossConfig holds the PPO objective's fixed hyperparameters: how far
// the probability ratio is allowed to move before it's clipped
// (ClipEpsilon), and the relative weight of the value loss and entropy
// bonus against the clipped-surrogate policy loss.
type LossConfig struct {
	ClipEpsilon float32
	EntropyCoef float32
	ValueCoef   float32
}

// TrainingNetwork composes two independent actorcritic.TrainingNetworks
// — PolicyActor (whose PolicyOutput feeds the policy+entropy terms) and
// ValueActor (whose ValueOutput feeds the value term) — with the PPO
// clipped-surrogate + value + entropy objective (see loss.go), built
// once per training run and driven by overwriting its placeholder Vars
// (via SetInput/SetActionMask/SetStep) before each Forward/Backward call
// — exactly how pkg/policy.TrainingNetwork's Advantage placeholder is
// overwritten once per step rather than rebuilt.
//
// PolicyActor and ValueActor must never be built from the same
// *actorcritic.Params instance (see NewTrainingNetwork): every Var an
// actorcritic.TrainingNetwork wraps aliases its Params' own matrices
// directly (autograd.Parameter never copies), so sharing one Params
// between both would silently merge the two "independent" branches back
// together the moment either one's Adam step mutated a shared weight.
//
// Splitting the policy and value branches into disjoint parameter sets
// this way — mirroring RecurrentChunkTrainingNetwork's own
// PolicyParameters/ValueParameters split (see
// pkg/actorcritic/recurrent_network.go) — is what lets Trainer recover
// an exact policy-only/value-only gradient norm from a single combined
// Forward/Backward pass over Graph, rather than needing three separate
// passes (see Trainer.trainOnMinibatch's doc comment).
type TrainingNetwork struct {
	PolicyActor *actorcritic.TrainingNetwork
	ValueActor  *actorcritic.TrainingNetwork

	// ActionMask, OldLogProb, and Advantage are all shaped like the
	// policy head's output ((outputSize, 1)) and are cleared and given
	// a single nonzero entry at the sampled action's index by SetStep;
	// see loss.go's doc comment for why the loss composition relies on
	// exactly one action contributing per step.
	ActionMask   *autograd.Var
	OldLogProb   *autograd.Var
	Advantage    *autograd.Var
	ReturnTarget *autograd.Var

	Loss  *autograd.Var
	Graph *autograd.Graph
}

// NewTrainingNetwork builds a TrainingNetwork over policyParams (feeding
// PolicyActor) and valueParams (feeding ValueActor) — two distinct
// *actorcritic.Params instances; see TrainingNetwork's own doc comment
// for why passing the same instance for both would be incorrect. cfg's
// hyperparameters are fixed as constants baked into the graph (they
// don't change during training, unlike the per-step placeholder Vars
// SetStep overwrites).
func NewTrainingNetwork(policyParams, valueParams *actorcritic.Params, cfg LossConfig) (*TrainingNetwork, error) {
	if policyParams == valueParams {
		return nil, fmt.Errorf("ppo: building actor-critic training network: policyParams and valueParams must be distinct instances")
	}

	policyActor, err := actorcritic.NewTrainingNetwork(policyParams)
	if err != nil {
		return nil, fmt.Errorf("ppo: building policy actor-critic training network: %w", err)
	}
	valueActor, err := actorcritic.NewTrainingNetwork(valueParams)
	if err != nil {
		return nil, fmt.Errorf("ppo: building value actor-critic training network: %w", err)
	}

	outputSize := policyParams.OutputSize()
	actionMask := autograd.NewVar(outputSize, 1, autograd.FlagNone)
	oldLogProb := autograd.NewVar(outputSize, 1, autograd.FlagNone)
	advantage := autograd.NewVar(outputSize, 1, autograd.FlagNone)
	returnTarget := autograd.NewVar(1, 1, autograd.FlagNone)

	policyLoss, valueLoss, entropyBonus, err := buildLossComponents(policyActor.PolicyOutput, valueActor.ValueOutput, actionMask, oldLogProb, advantage, returnTarget, cfg)
	if err != nil {
		return nil, err
	}
	loss, err := combineObjectiveLosses(policyLoss, valueLoss, entropyBonus)
	if err != nil {
		return nil, err
	}

	return &TrainingNetwork{
		PolicyActor:  policyActor,
		ValueActor:   valueActor,
		ActionMask:   actionMask,
		OldLogProb:   oldLogProb,
		Advantage:    advantage,
		ReturnTarget: returnTarget,
		Loss:         loss,
		Graph:        autograd.BuildGraph(loss),
	}, nil
}

// SetInput copies observation into both PolicyActor's and ValueActor's
// Input — both branches must see the exact same observation — and sets
// the action-legality mask on PolicyActor only (the value head's output
// never depends on which actions are legal).
func (n *TrainingNetwork) SetInput(observation []float32, mask []bool) {
	copy(n.PolicyActor.Input.Val.Data, observation)
	copy(n.ValueActor.Input.Val.Data, observation)
	n.PolicyActor.SetActionMask(mask)
}

// SetStep overwrites the per-step placeholder Vars ahead of one
// Forward/Backward call: ActionMask, OldLogProb, and Advantage are
// cleared and then given a single nonzero entry at action's index
// (oldLogProb and advantage respectively), and ReturnTarget is set to
// returnTarget.
func (n *TrainingNetwork) SetStep(action rl.Action, oldLogProb, advantage, returnTarget float32) {
	n.ActionMask.Val.Clear()
	n.ActionMask.Val.Data[action] = 1

	n.OldLogProb.Val.Clear()
	n.OldLogProb.Val.Data[action] = oldLogProb

	n.Advantage.Val.Clear()
	n.Advantage.Val.Data[action] = advantage

	n.ReturnTarget.Val.Data[0] = returnTarget
}

// Parameters returns every trainable Var of both branches — PolicyActor's
// first, then ValueActor's — the set ZeroGrad clears and the set a
// caller's optimizer (see Trainer.adam) should step.
func (n *TrainingNetwork) Parameters() []*autograd.Var {
	return append(n.PolicyActor.Parameters(), n.ValueActor.Parameters()...)
}

// ZeroGrad clears both branches' accumulated gradients.
func (n *TrainingNetwork) ZeroGrad() {
	n.PolicyActor.ZeroGrad()
	n.ValueActor.ZeroGrad()
}

// SharedTrunkTrainingNetwork is TrainingNetwork's predecessor: a single
// actorcritic.TrainingNetwork (one shared trunk feeding both heads)
// wrapped in the same PPO clipped-surrogate + value + entropy objective.
// It exists only because pkg/hierarchical.Trainer already builds and
// drives this shape directly as a library (one network per
// meta-controller/sub-policy, with its own single-pass training loop —
// see pkg/hierarchical/trainer.go's trainOnMinibatch), and migrating it
// to TrainingNetwork's split policy/value branches would also require
// migrating its rollout collection and checkpoint format to match (its
// InferenceNetwork-based action selection reads both outputs from one
// shared-trunk forward pass, not two independent ones) — a materially
// larger, separate change than the PPO-trainer-only redundant-pass fix
// this type was split off to make possible. Unlike TrainingNetwork, a
// single combined Forward/Backward pass here cannot separate the
// trunk's gradient into a pure policy-only/value-only contribution,
// which is exactly why pkg/ppo.Trainer no longer uses this shape.
type SharedTrunkTrainingNetwork struct {
	Actor *actorcritic.TrainingNetwork

	ActionMask   *autograd.Var
	OldLogProb   *autograd.Var
	Advantage    *autograd.Var
	ReturnTarget *autograd.Var

	Loss  *autograd.Var
	Graph *autograd.Graph
}

// NewSharedTrunkTrainingNetwork builds a SharedTrunkTrainingNetwork over
// actorParams; see TrainingNetwork's doc comment for why pkg/ppo.Trainer
// itself no longer uses this shape.
func NewSharedTrunkTrainingNetwork(actorParams *actorcritic.Params, cfg LossConfig) (*SharedTrunkTrainingNetwork, error) {
	actor, err := actorcritic.NewTrainingNetwork(actorParams)
	if err != nil {
		return nil, fmt.Errorf("ppo: building actor-critic training network: %w", err)
	}

	outputSize := actorParams.OutputSize()
	actionMask := autograd.NewVar(outputSize, 1, autograd.FlagNone)
	oldLogProb := autograd.NewVar(outputSize, 1, autograd.FlagNone)
	advantage := autograd.NewVar(outputSize, 1, autograd.FlagNone)
	returnTarget := autograd.NewVar(1, 1, autograd.FlagNone)

	policyLoss, valueLoss, entropyBonus, err := buildLossComponents(actor.PolicyOutput, actor.ValueOutput, actionMask, oldLogProb, advantage, returnTarget, cfg)
	if err != nil {
		return nil, err
	}
	loss, err := combineObjectiveLosses(policyLoss, valueLoss, entropyBonus)
	if err != nil {
		return nil, err
	}

	return &SharedTrunkTrainingNetwork{
		Actor:        actor,
		ActionMask:   actionMask,
		OldLogProb:   oldLogProb,
		Advantage:    advantage,
		ReturnTarget: returnTarget,
		Loss:         loss,
		Graph:        autograd.BuildGraph(loss),
	}, nil
}

// SetStep is SharedTrunkTrainingNetwork's counterpart to
// TrainingNetwork.SetStep; see its doc comment.
func (n *SharedTrunkTrainingNetwork) SetStep(action rl.Action, oldLogProb, advantage, returnTarget float32) {
	n.ActionMask.Val.Clear()
	n.ActionMask.Val.Data[action] = 1

	n.OldLogProb.Val.Clear()
	n.OldLogProb.Val.Data[action] = oldLogProb

	n.Advantage.Val.Clear()
	n.Advantage.Val.Data[action] = advantage

	n.ReturnTarget.Val.Data[0] = returnTarget
}
