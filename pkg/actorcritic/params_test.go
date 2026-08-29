package actorcritic

import (
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewParamsShapesAndBoundedWeights(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	params := NewParams(rng, 12, 8, 5)

	assert.Equal(t, 12, params.InputSize())
	assert.Equal(t, 8, params.HiddenSize())
	assert.Equal(t, 5, params.OutputSize())

	assert.Equal(t, 8, params.W0.Rows)
	assert.Equal(t, 12, params.W0.Cols)
	assert.Equal(t, 8, params.W1.Rows)
	assert.Equal(t, 8, params.W1.Cols)
	assert.Equal(t, 5, params.Wpi.Rows)
	assert.Equal(t, 8, params.Wpi.Cols)

	// The value head always has width 1, regardless of the policy
	// head's output size.
	assert.Equal(t, 1, params.Wv.Rows)
	assert.Equal(t, 8, params.Wv.Cols)
	assert.Equal(t, 1, params.Bv.Rows)
	assert.Equal(t, 1, params.Bv.Cols)

	// Biases start at zero.
	for _, v := range params.B0.Data {
		assert.Equal(t, float32(0), v)
	}
	for _, v := range params.Bv.Data {
		assert.Equal(t, float32(0), v)
	}
}

func TestNewParamsProducesDistinctIndependentInstances(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	a := NewParams(rng, 6, 4, 3)
	b := NewParams(rng, 6, 4, 3)

	assert.NotSame(t, a.W0, b.W0)
	assert.NotSame(t, a.Wv, b.Wv)
}
