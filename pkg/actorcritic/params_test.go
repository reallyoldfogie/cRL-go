package actorcritic

import (
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewParamsShapesAndBoundedWeights(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	params := NewParams(rng, 12, 8, 5, 2)

	assert.Equal(t, 12, params.InputSize())
	assert.Equal(t, 8, params.HiddenSize())
	assert.Equal(t, 5, params.OutputSize())
	assert.Equal(t, 2, params.NumHiddenLayers())

	assert.Equal(t, 8, params.Hidden[0].W.Rows)
	assert.Equal(t, 12, params.Hidden[0].W.Cols)
	assert.Equal(t, 8, params.Hidden[1].W.Rows)
	assert.Equal(t, 8, params.Hidden[1].W.Cols)
	assert.Equal(t, 5, params.Wpi.Rows)
	assert.Equal(t, 8, params.Wpi.Cols)

	// The value head always has width 1, regardless of the policy
	// head's output size.
	assert.Equal(t, 1, params.Wv.Rows)
	assert.Equal(t, 8, params.Wv.Cols)
	assert.Equal(t, 1, params.Bv.Rows)
	assert.Equal(t, 1, params.Bv.Cols)

	// Biases start at zero.
	for _, v := range params.Hidden[0].B.Data {
		assert.Equal(t, float32(0), v)
	}
	for _, v := range params.Bv.Data {
		assert.Equal(t, float32(0), v)
	}
}

func TestNewParamsSupportsArbitraryDepth(t *testing.T) {
	rng := rand.New(rand.NewPCG(13, 14))
	params := NewParams(rng, 12, 8, 5, 4)

	require.Equal(t, 4, params.NumHiddenLayers())
	assert.Equal(t, 12, params.Hidden[0].W.Cols, "only the first hidden layer consumes the raw input width")
	for i := 1; i < 4; i++ {
		assert.Equal(t, 8, params.Hidden[i].W.Cols, "every later hidden layer consumes the previous layer's HiddenSize-wide output")
	}
	for i := 0; i < 4; i++ {
		assert.Equal(t, 8, params.Hidden[i].W.Rows)
	}
}

func TestNewParamsRejectsFewerThanOneHiddenLayer(t *testing.T) {
	rng := rand.New(rand.NewPCG(15, 16))
	assert.Panics(t, func() { NewParams(rng, 6, 4, 3, 0) })
}

func TestNewParamsProducesDistinctIndependentInstances(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	a := NewParams(rng, 6, 4, 3, 2)
	b := NewParams(rng, 6, 4, 3, 2)

	assert.NotSame(t, a.Hidden[0].W, b.Hidden[0].W)
	assert.NotSame(t, a.Wv, b.Wv)
}
