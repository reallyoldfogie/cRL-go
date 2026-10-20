package ppo

import (
	"math"

	"github.com/reallyoldfogie/cRL-go/pkg/autograd"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

// epochDiagnostics contains objective and optimizer signals accumulated while
// replaying one PPO epoch. The loss components are reported per real timestep;
// gradient norms are reported per Adam update after its sample-count
// averaging, before and after configured clipping.
type epochDiagnostics struct {
	policyLossSum     float64
	valueLossSum      float64
	entropySum        float64
	steps             int
	gradientNormSum   float64
	appliedNormSum    float64
	policyGradNormSum float64
	valueGradNormSum  float64
	updates           int
}

func (d *epochDiagnostics) observeStep(policy []float32, value, oldLogProb, advantage, returnTarget float32, action rl.Action, cfg LossConfig) {
	if int(action) < 0 || int(action) >= len(policy) {
		return
	}
	prob := policy[action]
	if prob < logProbEpsilon {
		prob = logProbEpsilon
	}
	ratio := float32(math.Exp(float64(float32(math.Log(float64(prob))) - oldLogProb)))
	clipped := ratio
	if clipped < 1-cfg.ClipEpsilon {
		clipped = 1 - cfg.ClipEpsilon
	}
	if clipped > 1+cfg.ClipEpsilon {
		clipped = 1 + cfg.ClipEpsilon
	}
	surrogate1 := ratio * advantage
	surrogate2 := clipped * advantage
	policyLoss := -surrogate1
	if surrogate2 < surrogate1 {
		policyLoss = -surrogate2
	}
	valueDiff := value - returnTarget
	valueLoss := cfg.ValueCoef * valueDiff * valueDiff
	var entropy float32
	for _, p := range policy {
		if p > logProbEpsilon {
			entropy -= p * float32(math.Log(float64(p)))
		}
	}

	d.policyLossSum += float64(policyLoss)
	d.valueLossSum += float64(valueLoss)
	d.entropySum += float64(entropy)
	d.steps++
}

func (d *epochDiagnostics) observeUpdate(gradientNorm, appliedNorm, policyGradientNorm, valueGradientNorm float32) {
	d.gradientNormSum += float64(gradientNorm)
	d.appliedNormSum += float64(appliedNorm)
	d.policyGradNormSum += float64(policyGradientNorm)
	d.valueGradNormSum += float64(valueGradientNorm)
	d.updates++
}

func (d epochDiagnostics) averages() (policyLoss, valueLoss, entropy, gradientNorm, appliedNorm, policyGradientNorm, valueGradientNorm float32) {
	if d.steps > 0 {
		policyLoss = float32(d.policyLossSum / float64(d.steps))
		valueLoss = float32(d.valueLossSum / float64(d.steps))
		entropy = float32(d.entropySum / float64(d.steps))
	}
	if d.updates > 0 {
		gradientNorm = float32(d.gradientNormSum / float64(d.updates))
		appliedNorm = float32(d.appliedNormSum / float64(d.updates))
		policyGradientNorm = float32(d.policyGradNormSum / float64(d.updates))
		valueGradientNorm = float32(d.valueGradNormSum / float64(d.updates))
	}
	return
}

func objectiveGradientNorm(parameters []*autograd.Var, sampleCount int) float32 {
	if sampleCount <= 0 {
		return 0
	}
	scale := 1 / float32(sampleCount)
	var sumSquares float64
	for _, parameter := range parameters {
		for _, gradient := range parameter.Grad.Data {
			scaled := float64(gradient * scale)
			sumSquares += scaled * scaled
		}
	}
	return float32(math.Sqrt(sumSquares))
}

func finishAdvantageStats(values []float32) (mean, std, min, max float32) {
	if len(values) == 0 {
		return
	}
	min, max = values[0], values[0]
	var sum, sumSquares float64
	for _, value := range values {
		if value < min {
			min = value
		}
		if value > max {
			max = value
		}
		sum += float64(value)
		sumSquares += float64(value) * float64(value)
	}
	mean = float32(sum / float64(len(values)))
	variance := sumSquares/float64(len(values)) - float64(mean)*float64(mean)
	if variance > 0 {
		std = float32(math.Sqrt(variance))
	}
	return
}
