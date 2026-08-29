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

// TrainingNetwork composes an actorcritic.TrainingNetwork with the PPO
// clipped-surrogate + value + entropy objective (see loss.go), built
// once per training run and driven by overwriting its placeholder Vars
// (via SetStep) before each Forward/Backward call — exactly how
// pkg/policy.TrainingNetwork's Advantage placeholder is overwritten once
// per step rather than rebuilt.
type TrainingNetwork struct {
	Actor *actorcritic.TrainingNetwork

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

// NewTrainingNetwork builds a TrainingNetwork over actorParams, fixing
// cfg's hyperparameters as constants baked into the graph (they don't
// change during training, unlike the per-step placeholder Vars SetStep
// overwrites).
func NewTrainingNetwork(actorParams *actorcritic.Params, cfg LossConfig) (*TrainingNetwork, error) {
	actor, err := actorcritic.NewTrainingNetwork(actorParams)
	if err != nil {
		return nil, fmt.Errorf("ppo: building actor-critic training network: %w", err)
	}

	outputSize := actorParams.OutputSize()
	actionMask := autograd.NewVar(outputSize, 1, autograd.FlagNone)
	oldLogProb := autograd.NewVar(outputSize, 1, autograd.FlagNone)
	advantage := autograd.NewVar(outputSize, 1, autograd.FlagNone)
	returnTarget := autograd.NewVar(1, 1, autograd.FlagNone)

	loss, err := buildLoss(actor, actionMask, oldLogProb, advantage, returnTarget, cfg)
	if err != nil {
		return nil, err
	}

	return &TrainingNetwork{
		Actor:        actor,
		ActionMask:   actionMask,
		OldLogProb:   oldLogProb,
		Advantage:    advantage,
		ReturnTarget: returnTarget,
		Loss:         loss,
		Graph:        autograd.BuildGraph(loss),
	}, nil
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
