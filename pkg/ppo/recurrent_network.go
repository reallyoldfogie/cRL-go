package ppo

import (
	"fmt"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/autograd"
	"github.com/reallyoldfogie/cRL-go/pkg/mat"
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
//
// BatchSize chunks are trained together in one such Forward/Backward
// call (see Actor's own BatchSize): every placeholder below is
// (OutputSize, BatchSize)/(1, BatchSize) rather than (OutputSize,
// 1)/(1, 1), column j holding batch member j's own data, so the whole
// Loss sums over every real (timestep, batch column) pair in the
// chunk-batch at once — see loss.go's own doc comment for why summing
// over columns this way is exactly equivalent to what BatchSize
// separate Forward/Backward calls, one per chunk, would have
// accumulated.
type RecurrentTrainingNetwork struct {
	Actor *actorcritic.RecurrentChunkTrainingNetwork

	ChunkLen  int
	BatchSize int

	// ActionMask, OldLogProb, Advantage, ReturnTarget mirror
	// TrainingNetwork's own placeholders (network.go) — one-hot per
	// column at that column's sampled action index for loss indexing,
	// not to be confused with Actor.MaskBias's action-*legality*
	// masking — but one set per timestep rather than one shared set.
	ActionMask   []*autograd.Var
	OldLogProb   []*autograd.Var
	Advantage    []*autograd.Var
	ReturnTarget []*autograd.Var
	// StepMask is an (OutputSize, BatchSize) Var whose column j is
	// filled entirely with 1.0 if timestep t is a real step for batch
	// member j, or 0.0 if it's padding (a short final chunk, or an
	// unused batch column when fewer than BatchSize chunks were
	// supplied — see SetChunkColumn), multiplying that whole column's
	// loss vector to zero uniformly across every term. An all-zero
	// ActionMask alone would already zero buildPolicyLoss's and
	// buildValueLoss's contributions (see loss.go's own masking math),
	// but not buildEntropyBonus's — a genuine unmasked per-action sum by
	// design — so a padded (timestep, column)'s otherwise-meaningless
	// policy output would still leak a nonzero entropy term into the
	// combined loss without this.
	StepMask []*autograd.Var

	Loss  *autograd.Var
	Graph *autograd.Graph
}

// NewRecurrentTrainingNetwork builds a RecurrentTrainingNetwork over
// actorParams, unrolled chunkLen timesteps, training batchSize chunks
// together — see RecurrentTrainingNetwork's own doc comment.
func NewRecurrentTrainingNetwork(actorParams *actorcritic.RecurrentParams, chunkLen, batchSize int, cfg LossConfig) (*RecurrentTrainingNetwork, error) {
	actor, err := actorcritic.NewRecurrentChunkTrainingNetwork(actorParams, chunkLen, batchSize)
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
		actionMask[t] = autograd.NewVar(outputSize, batchSize, autograd.FlagNone)
		oldLogProb[t] = autograd.NewVar(outputSize, batchSize, autograd.FlagNone)
		advantage[t] = autograd.NewVar(outputSize, batchSize, autograd.FlagNone)
		returnTarget[t] = autograd.NewVar(1, batchSize, autograd.FlagNone)
		stepMask[t] = autograd.NewVar(outputSize, batchSize, autograd.FlagNone)

		policyLoss, valueLoss, entropyBonus, err := buildLossComponents(actor.PolicyOutput[t], actor.ValueOutput[t], actionMask[t], oldLogProb[t], advantage[t], returnTarget[t], cfg)
		if err != nil {
			return nil, err
		}
		stepLoss, err := combineObjectiveLosses(policyLoss, valueLoss, entropyBonus)
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
		BatchSize:    batchSize,
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

// SetChunkColumn overwrites every per-timestep placeholder's batch
// column batchIndex ahead of one Forward/Backward call: steps must have
// exactly n.ChunkLen entries. hPrevStart/cPrevStart seed that column's
// chunk's starting hidden/cell state (recorded at rollout time — see
// RecurrentRollout). Call once per batchIndex in [0, n.BatchSize)
// before Forward — this network is reused across many Forward/Backward
// cycles, so every column (including any left over as padding when
// fewer than BatchSize real chunks are available) must be set every
// time; see RecurrentTrainer.trainNetworkOnChunks, which always does
// both (real chunks, then padding) before each Forward call.
func (n *RecurrentTrainingNetwork) SetChunkColumn(batchIndex int, steps []recurrentTrainingStep, hPrevStart, cPrevStart []float32) {
	n.Actor.SetChunkState(batchIndex, hPrevStart, cPrevStart)
	n.setChunkColumnData(batchIndex, steps)
}

// SetSplitChunkColumn is SetChunkColumn for a chunk whose policy and
// value branches started from independently-recorded recurrent state
// (see RecurrentRollout.ValueHPrevs/ValueCPrevs).
func (n *RecurrentTrainingNetwork) SetSplitChunkColumn(batchIndex int, steps []recurrentTrainingStep, policyH, policyC, valueH, valueC []float32) {
	n.Actor.SetSplitChunkState(batchIndex, policyH, policyC, valueH, valueC)
	n.setChunkColumnData(batchIndex, steps)
}

func (n *RecurrentTrainingNetwork) setChunkColumnData(batchIndex int, steps []recurrentTrainingStep) {
	for t, step := range steps {
		n.Actor.Input[t].Val.SetColumn(batchIndex, step.Observation.Values)
		n.Actor.SetActionMask(t, batchIndex, step.Mask)

		n.ActionMask[t].Val.ClearColumn(batchIndex)
		n.OldLogProb[t].Val.ClearColumn(batchIndex)
		n.Advantage[t].Val.ClearColumn(batchIndex)
		n.ReturnTarget[t].Val.SetColumn(batchIndex, []float32{step.Return})

		if step.Real {
			setOneHotColumn(n.ActionMask[t].Val, batchIndex, step.Action, 1)
			setOneHotColumn(n.OldLogProb[t].Val, batchIndex, step.Action, step.OldLogProb)
			setOneHotColumn(n.Advantage[t].Val, batchIndex, step.Action, step.Advantage)
			n.StepMask[t].Val.FillColumn(batchIndex, 1)
		} else {
			n.StepMask[t].Val.FillColumn(batchIndex, 0)
		}
	}
}

// setOneHotColumn sets column batchIndex's entry at row int(action) of
// m to value — the strided-write counterpart, for a single sparse
// entry, to mat.Matrix.SetColumn's whole-column copy.
func setOneHotColumn(m *mat.Matrix, batchIndex int, action rl.Action, value float32) {
	m.Data[int(action)*m.Cols+batchIndex] = value
}
