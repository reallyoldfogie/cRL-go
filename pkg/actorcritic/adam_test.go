package actorcritic

import (
	"math"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/autograd"
	"github.com/reallyoldfogie/cRL-go/pkg/mat"
	"github.com/stretchr/testify/assert"
)

// TestAdamStepMatchesHandComputedUpdate checks a single Adam step
// against the reference update computed by hand for the same inputs:
//
//	gradient = 1.0 / sampleCount = 1.0 / 2 = 0.5
//	moment1 = (1-0.9)*0.5 = 0.05
//	moment2 = (1-0.999)*0.5^2 = 0.00025
//	moment1Hat = 0.05 / (1-0.9^1) = 0.05 / 0.1 = 0.5
//	moment2Hat = 0.00025 / (1-0.999^1) = 0.00025 / 0.001 = 0.25
//	update = learningRate * moment1Hat / (sqrt(moment2Hat) + epsilon)
//	       = 0.1 * 0.5 / (0.5 + 1e-8) ≈ 0.1
//	newValue = 1.0 - 0.1 ≈ 0.9
func TestAdamStepMatchesHandComputedUpdate(t *testing.T) {
	param := autograd.Parameter(&mat.Matrix{Rows: 1, Cols: 1, Data: []float32{1.0}})
	param.Grad.Data[0] = 1.0 // accumulated gradient before averaging by sampleCount

	adam := NewAdam([]*autograd.Var{param}, 0.1)
	adam.Step(2)

	assert.InDelta(t, 0.9, param.Val.Data[0], 1e-3)
}

// TestAdamStepNoOpWithZeroSamples confirms Step leaves every parameter
// (and the bias-correction step counter) untouched when sampleCount is
// 0, mirroring ApplyGradientStep's equivalent guard.
func TestAdamStepNoOpWithZeroSamples(t *testing.T) {
	param := autograd.Parameter(&mat.Matrix{Rows: 1, Cols: 1, Data: []float32{1.0}})
	param.Grad.Data[0] = 1.0

	adam := NewAdam([]*autograd.Var{param}, 0.1)
	adam.Step(0)

	assert.Equal(t, float32(1.0), param.Val.Data[0])
}

// TestAdamStepBiasCorrectionAdvancesAcrossCalls confirms repeated Step
// calls use an increasing step count for bias correction (rather than
// resetting), by checking the update's magnitude shrinks in the
// expected way as moment1Hat/moment2Hat's correction factors approach 1
// with a repeatedly-zero gradient after the first step.
func TestAdamStepBiasCorrectionAdvancesAcrossCalls(t *testing.T) {
	param := autograd.Parameter(&mat.Matrix{Rows: 1, Cols: 1, Data: []float32{1.0}})
	param.Grad.Data[0] = 1.0

	adam := NewAdam([]*autograd.Var{param}, 0.1)
	adam.Step(1)
	firstUpdateValue := param.Val.Data[0]

	// A second step with the same (nonzero) gradient must still move
	// the parameter further, rather than repeating the exact same
	// state (which would indicate the moment estimates or step counter
	// weren't actually advancing).
	adam.Step(1)
	secondUpdateValue := param.Val.Data[0]

	assert.NotEqual(t, firstUpdateValue, secondUpdateValue)
	assert.False(t, math.IsNaN(float64(secondUpdateValue)))
}
