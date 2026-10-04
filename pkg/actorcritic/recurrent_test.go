package actorcritic

import (
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWarmStartRecurrentParamsReusesTrunkWeightsAndAddsAFreshLSTM(t *testing.T) {
	rng := rand.New(rand.NewPCG(31, 32))
	trunk := NewParams(rng, 12, 8, 5, 2)

	warm := WarmStartRecurrentParams(trunk, rng)

	assert.Equal(t, trunk.Hidden[0].W.Data, warm.Trunk.Hidden[0].W.Data, "the trunk's learned weights must be reused exactly")
	assert.Equal(t, trunk.Wpi.Data, warm.Trunk.Wpi.Data)
	require.NotSame(t, trunk.Hidden[0].W, warm.Trunk.Hidden[0].W, "the trunk must be cloned (Snapshot), not aliased")
	assert.Equal(t, 8, warm.LSTM.HiddenSize())
	assert.Equal(t, 8, warm.LSTM.InputSize())
	assert.Equal(t, float32(1.0), warm.LSTM.BF.Data[0], "the fresh LSTM still gets the forget-gate-bias-1.0 init")

	// Mutating the source trunk afterward must not affect the warm-started copy.
	trunk.Hidden[0].W.Data[0] += 100
	assert.NotEqual(t, trunk.Hidden[0].W.Data, warm.Trunk.Hidden[0].W.Data)
}

func TestNewRecurrentParamsShapes(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	p := NewRecurrentParams(rng, 12, 8, 5, 2)

	assert.Equal(t, 12, p.InputSize())
	assert.Equal(t, 8, p.HiddenSize())
	assert.Equal(t, 5, p.OutputSize())
	assert.Equal(t, 2, p.NumHiddenLayers())

	assert.Equal(t, 8, p.LSTM.HiddenSize())
	assert.Equal(t, 8, p.LSTM.InputSize(), "the LSTM consumes the trunk's HiddenSize-wide output")

	assert.Equal(t, 8, p.LSTM.WxF.Rows)
	assert.Equal(t, 8, p.LSTM.WxF.Cols)
	assert.Equal(t, 8, p.LSTM.WhF.Rows)
	assert.Equal(t, 8, p.LSTM.WhF.Cols)
	assert.Equal(t, 8, p.LSTM.BF.Rows)
	assert.Equal(t, 1, p.LSTM.BF.Cols)
}

// TestNewLSTMLayerForgetGateBiasStartsAtOne pins the Jozefowicz et al.
// initialization trick this package relies on: the forget gate's bias
// starts biased toward retaining memory (1.0), not 0 like every other
// bias, since a freshly-initialized LSTM with a 0 forget bias forgets
// roughly half its state every step and is empirically much harder to
// train from.
func TestNewLSTMLayerForgetGateBiasStartsAtOne(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	l := NewLSTMLayer(rng, 6, 4)

	for _, v := range l.BF.Data {
		assert.Equal(t, float32(1.0), v)
	}
	// Every other bias starts at zero, same as NewParams' biases.
	for _, v := range l.BI.Data {
		assert.Equal(t, float32(0), v)
	}
	for _, v := range l.BO.Data {
		assert.Equal(t, float32(0), v)
	}
	for _, v := range l.BG.Data {
		assert.Equal(t, float32(0), v)
	}
}

func TestNewRecurrentParamsProducesDistinctIndependentInstances(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	a := NewRecurrentParams(rng, 6, 4, 3, 2)
	b := NewRecurrentParams(rng, 6, 4, 3, 2)

	assert.NotSame(t, a.Trunk.Hidden[0].W, b.Trunk.Hidden[0].W)
	assert.NotSame(t, a.LSTM.WxF, b.LSTM.WxF)
}

func TestRecurrentParamsSnapshotIsIndependentDeepCopy(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	p := NewRecurrentParams(rng, 6, 4, 3, 2)

	snapshot := p.Snapshot()
	require.NotSame(t, p.Trunk.Hidden[0].W, snapshot.Trunk.Hidden[0].W)
	require.NotSame(t, p.LSTM.WxF, snapshot.LSTM.WxF)
	assert.Equal(t, p.Trunk.Hidden[0].W.Data, snapshot.Trunk.Hidden[0].W.Data)
	assert.Equal(t, p.LSTM.WxF.Data, snapshot.LSTM.WxF.Data)

	// Mutating the live params must not affect the snapshot.
	p.Trunk.Hidden[0].W.Data[0] += 100
	p.LSTM.WxF.Data[0] += 100
	assert.NotEqual(t, p.Trunk.Hidden[0].W.Data, snapshot.Trunk.Hidden[0].W.Data)
	assert.NotEqual(t, p.LSTM.WxF.Data, snapshot.LSTM.WxF.Data)
}
