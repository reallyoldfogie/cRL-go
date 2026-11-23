package ppo

import (
	"github.com/reallyoldfogie/cRL-go/pkg/autograd"
	"github.com/reallyoldfogie/cRL-go/pkg/mat"
)

// buildLoss composes the PPO clipped-surrogate policy objective, a
// value-function squared-error loss, and an entropy bonus into a single
// combined loss Var, following docs/plans/03-gae-and-ppo-objective.md.
//
// Every intermediate Var here keeps the same shape as the policy head's
// output ((outputSize, BatchSize) — (outputSize, 1) for every
// non-recurrent caller, which never batches), or is broadcast into it,
// so the whole loss stays one (outputSize, BatchSize) matrix whose
// entries sum — via Graph.Backward's all-ones gradient seeding, see
// pkg/autograd/graph.go's doc comment and pkg/autograd/ops.go's
// reinforceLossOp, which relies on the same property — to the true
// scalar objective, summed over every batch column too (exactly what a
// caller training several samples/chunks together in one combined pass
// wants: the gradient of their total loss, equal to what separately
// accumulating each sample's own backward pass would have summed to):
//
//   - The policy surrogate terms are masked to the sampled action's
//     index by actionMask (one-hot per column: 1 at that column's
//     sampled action, 0 elsewhere), so only that entry is nonzero in
//     each column; every other entry algebraically reduces to 0 (ratio
//     1 there, multiplied by a 0 advantage).
//   - The entropy bonus is a genuine per-action sum (that's the
//     definition of entropy), so every action contributes and no
//     masking is needed.
//   - The value loss is a single (1, BatchSize) row, one scalar per
//     column; broadcasting it evenly into every entry of an
//     (outputSize, BatchSize) matrix would multiply it by outputSize
//     once summed, so it's instead broadcast into each column's own
//     sampled action's index only (via actionMask, through
//     MulBroadcastCols), contributing exactly once per column.
func buildLoss(
	policyOutput, valueOutput *autograd.Var,
	actionMask, oldLogProb, advantage, returnTarget *autograd.Var,
	cfg LossConfig,
) (*autograd.Var, error) {
	policyLoss, valueLoss, entropyBonus, err := buildLossComponents(policyOutput, valueOutput, actionMask, oldLogProb, advantage, returnTarget, cfg)
	if err != nil {
		return nil, err
	}
	return combineObjectiveLosses(policyLoss, valueLoss, entropyBonus)
}

func buildLossComponents(
	policyOutput, valueOutput *autograd.Var,
	actionMask, oldLogProb, advantage, returnTarget *autograd.Var,
	cfg LossConfig,
) (policyLoss, valueLoss, entropyBonus *autograd.Var, err error) {
	probs := policyOutput
	value := valueOutput

	logProbsNew, err := autograd.Log(probs)
	if err != nil {
		return nil, nil, nil, err
	}

	policyLoss, err = buildPolicyLoss(logProbsNew, actionMask, oldLogProb, advantage, cfg.ClipEpsilon)
	if err != nil {
		return nil, nil, nil, err
	}

	valueLoss, err = buildValueLoss(value, returnTarget, actionMask, cfg.ValueCoef)
	if err != nil {
		return nil, nil, nil, err
	}

	entropyBonus, err = buildEntropyBonus(probs, logProbsNew, cfg.EntropyCoef)
	if err != nil {
		return nil, nil, nil, err
	}
	return policyLoss, valueLoss, entropyBonus, nil
}

func combineObjectiveLosses(policyLoss, valueLoss, entropyBonus *autograd.Var) (*autograd.Var, error) {
	lossWithValue, err := autograd.Add(policyLoss, valueLoss)
	if err != nil {
		return nil, err
	}
	// Subtract (not add) the entropy bonus: higher entropy should
	// *reduce* the loss, rewarding exploration, matching the standard
	// PPO objective L = L_clip - c1*L_VF + c2*S[pi] rewritten as a loss
	// to minimize (L_clip and L_VF are already framed as losses above,
	// i.e. already negated/squared appropriately).
	negatedEntropyBonus, err := autograd.Neg(entropyBonus)
	if err != nil {
		return nil, err
	}
	return autograd.Add(lossWithValue, negatedEntropyBonus)
}

// buildPolicyLoss computes the PPO clipped-surrogate policy loss:
//
//	ratio = exp(logProbsNew[a] - oldLogProb[a])
//	policyLoss = -min(ratio*advantage, clamp(ratio, 1-eps, 1+eps)*advantage)
//
// where a is the sampled action (identified by actionMask). Every term
// here is computed across the full (outputSize, 1) shape rather than
// indexing out a single scalar: at non-selected positions,
// maskedLogProbNew and oldLogProb are both 0 (actionMask/oldLogProb are
// zero there), so logRatio is 0, ratio is exp(0)=1, and both surrogates
// reduce to 1*0=0 once multiplied by advantage (which is also 0 at
// non-selected positions) — leaving only the selected action's terms
// nonzero.
func buildPolicyLoss(logProbsNew, actionMask, oldLogProb, advantage *autograd.Var, clipEpsilon float32) (*autograd.Var, error) {
	maskedLogProbNew, err := autograd.Mul(logProbsNew, actionMask)
	if err != nil {
		return nil, err
	}
	logRatio, err := autograd.Sub(maskedLogProbNew, oldLogProb)
	if err != nil {
		return nil, err
	}
	ratio, err := autograd.Exp(logRatio)
	if err != nil {
		return nil, err
	}

	surrogate1, err := autograd.Mul(ratio, advantage)
	if err != nil {
		return nil, err
	}
	clippedRatio, err := autograd.Clamp(ratio, 1-clipEpsilon, 1+clipEpsilon)
	if err != nil {
		return nil, err
	}
	surrogate2, err := autograd.Mul(clippedRatio, advantage)
	if err != nil {
		return nil, err
	}

	minSurrogate, err := autograd.Min(surrogate1, surrogate2)
	if err != nil {
		return nil, err
	}
	return autograd.Neg(minSurrogate)
}

// buildValueLoss computes valueCoef*(value - returnTarget)^2 — each a
// (1, BatchSize) row, one scalar per batch column — and broadcasts the
// result into each column's own sampled action's index of an
// (outputSize, BatchSize) matrix (via actionMask, see this file's doc
// comment), so it contributes to the combined loss exactly once per
// column.
func buildValueLoss(value, returnTarget, actionMask *autograd.Var, valueCoef float32) (*autograd.Var, error) {
	diff, err := autograd.Sub(value, returnTarget)
	if err != nil {
		return nil, err
	}
	squaredError, err := autograd.Mul(diff, diff)
	if err != nil {
		return nil, err
	}

	valueCoefVar := autograd.Constant(filledMatrix(value.Val.Rows, value.Val.Cols, valueCoef))
	scaledSquaredError, err := autograd.Mul(squaredError, valueCoefVar)
	if err != nil {
		return nil, err
	}

	// actionMask is (outputSize, BatchSize), scaledSquaredError is
	// (1, BatchSize): MulBroadcastCols(actionMask, scaledSquaredError)
	// [i][j] = actionMask[i][j] * scaledSquaredError[0][j], i.e. column
	// j's scalar value loss placed at that column's own sampled action's
	// index and 0 everywhere else in that column — a true MatMul can't
	// express this once BatchSize > 1 (its shared contraction dimension
	// would need actionMask.Cols == scaledSquaredError.Rows, forcing
	// BatchSize == 1), so this would otherwise silently mix one column's
	// value loss into every other column's gradient.
	return autograd.MulBroadcastCols(actionMask, scaledSquaredError)
}

// buildEntropyBonus computes entropyCoef * sum_a(-probs[a]*log(probs[a])),
// i.e. entropyCoef times the policy's Shannon entropy, expressed as a
// per-action (outputSize, BatchSize) matrix whose entries sum (via
// Backward's implicit-sum trick) to that scalar, per column.
// logProbsNew is accepted as a parameter (rather than recomputed) since
// buildLoss's caller already computed log(probs) for the policy loss.
func buildEntropyBonus(probs, logProbsNew *autograd.Var, entropyCoef float32) (*autograd.Var, error) {
	perActionEntropy, err := autograd.Mul(probs, logProbsNew)
	if err != nil {
		return nil, err
	}
	positiveEntropy, err := autograd.Neg(perActionEntropy)
	if err != nil {
		return nil, err
	}

	entropyCoefVar := autograd.Constant(filledMatrix(probs.Val.Rows, probs.Val.Cols, entropyCoef))
	return autograd.Mul(positiveEntropy, entropyCoefVar)
}

// filledMatrix returns a (rows, cols) matrix with every entry set to
// value — e.g. a scalar hyperparameter (entropyCoef, valueCoef) that
// doesn't vary by batch column, broadcast up front to whatever shape
// the Var it's combined with actually has, rather than needing its own
// broadcasting autograd op (it never needs a gradient, so there's
// nothing Backward would need to do with it beyond this).
func filledMatrix(rows, cols int, value float32) *mat.Matrix {
	m := mat.New(rows, cols)
	m.Fill(value)
	return m
}
