package ppo

import (
	"fmt"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/autograd"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

// RecurrentTrainingNetwork composes an
// actorcritic.RecurrentChunkTrainingNetwork with the PPO objective,
// once per timestep in the chunk, summed into one combined Loss over
// the whole chunk — see
// docs/plans/20-configurable-depth-step-memory-and-dynamic-architecture.md,
// Part B. Unlike the non-recurrent TrainingNetwork (one shared set of
// per-step placeholders, one Forward/Backward call per individual
// step), this holds ChunkLen separate sets of placeholders and builds
// its Graph/Loss once, spanning every timestep, since correct BPTT
// requires the whole chunk's forward computation in one graph.
type RecurrentTrainingNetwork struct {
	Actor *actorcritic.RecurrentChunkTrainingNetwork

	ChunkLen int

	// ActionMask, OldLogProb, Advantage, ReturnTarget mirror
	// TrainingNetwork's own placeholders (network.go) — one-hot at the
	// sampled action's index for loss indexing, not to be confused with
	// Actor.MaskBias's action-*legality* masking — but one set per
	// timestep rather than one shared set.
	ActionMask   []*autograd.Var
	OldLogProb   []*autograd.Var
	Advantage    []*autograd.Var
	ReturnTarget []*autograd.Var
	// StepMask is an (OutputSize, 1) Var filled entirely with 1.0 for a
	// real step or 0.0 for a padded one (a short final chunk — see
	// SetChunk), multiplying that whole timestep's loss vector to zero
	// uniformly across every term. An all-zero ActionMask alone would
	// already zero buildPolicyLoss's and buildValueLoss's contributions
	// (see loss.go's own masking math), but not buildEntropyBonus's — a
	// genuine unmasked per-action sum by design — so a padded step's
	// otherwise-meaningless policy output would still leak a nonzero
	// entropy term into the combined loss without this.
	StepMask []*autograd.Var

	Loss  *autograd.Var
	Graph *autograd.Graph
}

// NewRecurrentTrainingNetwork builds a RecurrentTrainingNetwork over
// actorParams, unrolled chunkLen timesteps.
func NewRecurrentTrainingNetwork(actorParams *actorcritic.RecurrentParams, chunkLen int, cfg LossConfig) (*RecurrentTrainingNetwork, error) {
	actor, err := actorcritic.NewRecurrentChunkTrainingNetwork(actorParams, chunkLen)
	if err != nil {
		return nil, fmt.Errorf("ppo: building recurrent actor-critic training network: %w", err)
	}

	outputSize := actorParams.OutputSize()

	actionMask := make([]*autograd.Var, chunkLen)
	oldLogProb := make([]*autograd.Var, chunkLen)
	advantage := make([]*autograd.Var, chunkLen)
	returnTarget := make([]*autograd.Var, chunkLen)
	stepMask := make([]*autograd.Var, chunkLen)

	var combinedLoss *autograd.Var
	for t := range chunkLen {
		actionMask[t] = autograd.NewVar(outputSize, 1, autograd.FlagNone)
		oldLogProb[t] = autograd.NewVar(outputSize, 1, autograd.FlagNone)
		advantage[t] = autograd.NewVar(outputSize, 1, autograd.FlagNone)
		returnTarget[t] = autograd.NewVar(1, 1, autograd.FlagNone)
		stepMask[t] = autograd.NewVar(outputSize, 1, autograd.FlagNone)

		stepLoss, err := buildLoss(actor.PolicyOutput[t], actor.ValueOutput[t], actionMask[t], oldLogProb[t], advantage[t], returnTarget[t], cfg)
		if err != nil {
			return nil, err
		}

		maskedStepLoss, err := autograd.Mul(stepLoss, stepMask[t])
		if err != nil {
			return nil, err
		}

		if combinedLoss == nil {
			combinedLoss = maskedStepLoss
		} else {
			combinedLoss, err = autograd.Add(combinedLoss, maskedStepLoss)
			if err != nil {
				return nil, err
			}
		}
	}

	return &RecurrentTrainingNetwork{
		Actor:        actor,
		ChunkLen:     chunkLen,
		ActionMask:   actionMask,
		OldLogProb:   oldLogProb,
		Advantage:    advantage,
		ReturnTarget: returnTarget,
		StepMask:     stepMask,
		Loss:         combinedLoss,
		Graph:        autograd.BuildGraph(combinedLoss),
	}, nil
}

// recurrentTrainingStep is one chunk position's flattened (observation,
// action, ...) tuple, mirroring trainingStep (trainer.go) but with an
// extra Real flag: a short final chunk (an episode length not a
// multiple of ChunkLen) is padded out to ChunkLen with Real=false
// entries rather than built as a variable-length graph — see
// RecurrentTrainer's own chunking logic for how these get constructed.
type recurrentTrainingStep struct {
	Observation rl.Observation
	Action      rl.Action
	Mask        []bool
	OldLogProb  float32
	Advantage   float32
	Return      float32
	Real        bool
}

// SetChunk overwrites every per-timestep placeholder ahead of one
// Forward/Backward call: steps must have exactly n.ChunkLen entries.
// hPrevStart/cPrevStart seed the chunk's starting hidden/cell state
// (recorded at rollout time — see RecurrentRollout).
func (n *RecurrentTrainingNetwork) SetChunk(steps []recurrentTrainingStep, hPrevStart, cPrevStart []float32) {
	n.Actor.SetChunkState(hPrevStart, cPrevStart)

	for t, step := range steps {
		copy(n.Actor.Input[t].Val.Data, step.Observation.Values)
		n.Actor.SetActionMask(t, step.Mask)

		n.ActionMask[t].Val.Clear()
		n.OldLogProb[t].Val.Clear()
		n.Advantage[t].Val.Clear()
		n.ReturnTarget[t].Val.Data[0] = step.Return

		if step.Real {
			n.ActionMask[t].Val.Data[step.Action] = 1
			n.OldLogProb[t].Val.Data[step.Action] = step.OldLogProb
			n.Advantage[t].Val.Data[step.Action] = step.Advantage
			n.StepMask[t].Val.Fill(1)
		} else {
			n.StepMask[t].Val.Fill(0)
		}
	}
}
