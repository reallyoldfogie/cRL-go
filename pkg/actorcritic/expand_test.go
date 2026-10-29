package actorcritic

import (
	"bytes"
	"math/rand/v2"
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/checkpoint"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestExpandParamsPreservesOldLogitsExactly is ExpandParams' central
// claim: feeding the SAME old-shape values into the expanded network
// (any new input columns multiplied by zero, so whatever garbage sits
// in them is irrelevant) leaves the value head's output byte-for-byte
// unchanged (Wv/Bv are untouched, and every hidden layer's own
// activation is unchanged - new input columns are zero, so the extra
// observation features contribute nothing), and preserves the exact
// probability RATIO between any two old actions - the one softmax-
// invariant property that holds regardless of what a new action's own
// starting logit is (new_logit=0 does NOT mean 0 post-softmax
// probability, since softmax's denominator still includes it - see
// ExpandParams' own doc comment for why that's harmless once action
// masking, not a bare unmasked forward pass, is what actually decides
// what gets sampled).
func TestExpandParamsPreservesOldLogitsExactly(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	oldInputSize, hiddenSize, oldOutputSize := 5, 6, 3
	p := NewParams(rng, oldInputSize, hiddenSize, oldOutputSize, 2)

	newInputSize, newOutputSize := 8, 5
	expanded, err := ExpandParams(p, newInputSize, newOutputSize)
	require.NoError(t, err)
	assert.Equal(t, newInputSize, expanded.InputSize())
	assert.Equal(t, newOutputSize, expanded.OutputSize())
	assert.Equal(t, hiddenSize, expanded.HiddenSize())
	assert.Equal(t, p.NumHiddenLayers(), expanded.NumHiddenLayers())

	for trial := 0; trial < 20; trial++ {
		oldObs := randomVector(rng, oldInputSize)
		// The new input columns are garbage on purpose - they must not
		// affect any output.
		newObs := append(append([]float32{}, oldObs...), randomVector(rng, newInputSize-oldInputSize)...)

		oldResult, err := forward(p, rl.Observation{Values: oldObs})
		require.NoError(t, err)
		newResult, err := forward(expanded, rl.Observation{Values: newObs})
		require.NoError(t, err)

		assert.InDelta(t, oldResult.value, newResult.value, 1e-5, "trial %d: value output changed", trial)
		for i := 1; i < oldOutputSize; i++ {
			oldRatio := oldResult.policy[i] / oldResult.policy[0]
			newRatio := newResult.policy[i] / newResult.policy[0]
			assert.InDelta(t, oldRatio, newRatio, 1e-4, "trial %d: probability ratio between old actions 0 and %d changed", trial, i)
		}
	}
}

// TestExpandParamsLeavesSourceUnmodified mirrors
// TestNet2WiderNetDoesNotModifyTheSourceParams' own check: ExpandParams
// must not mutate p.
func TestExpandParamsLeavesSourceUnmodified(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	p := NewParams(rng, 4, 5, 2, 2)
	before := p.Snapshot()

	_, err := ExpandParams(p, 9, 6)
	require.NoError(t, err)

	assert.Equal(t, before.Hidden[0].W.Data, p.Hidden[0].W.Data)
	assert.Equal(t, before.Wpi.Data, p.Wpi.Data)
	assert.Equal(t, 4, p.InputSize(), "the original must keep its original input size")
	assert.Equal(t, 2, p.OutputSize(), "the original must keep its original output size")
}

// TestExpandParamsRejectsShrinking is the plan's own explicit
// requirement: a smaller requested size is a hard error, never guessed
// at.
func TestExpandParamsRejectsShrinking(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	p := NewParams(rng, 5, 4, 3, 1)

	_, err := ExpandParams(p, 4, 3)
	assert.Error(t, err, "smaller input size must be rejected")

	_, err = ExpandParams(p, 5, 2)
	assert.Error(t, err, "smaller output size must be rejected")
}

// TestExpandParamsAllowsExactSizeAsANoOpClone confirms newInputSize ==
// InputSize() (and likewise for output) is accepted, not rejected as a
// "shrink" - ExpandRecurrentParams relies on this for the value trunk's
// own output size, which never changes.
func TestExpandParamsAllowsExactSizeAsANoOpClone(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	p := NewParams(rng, 5, 4, 3, 1)

	same, err := ExpandParams(p, 5, 3)
	require.NoError(t, err)
	assert.Equal(t, 5, same.InputSize())
	assert.Equal(t, 3, same.OutputSize())
	assert.Equal(t, p.Wpi.Data, same.Wpi.Data)
}

// TestExpandRecurrentParamsPreservesOldLogitsAndLeavesLSTMUntouched
// checks ExpandRecurrentParams' two central claims: the trunks expand
// exactly like ExpandParams on its own (reusing that function
// internally), and the LSTM layers - which never depended on
// observation/action size in the first place - come through as
// unmodified clones.
func TestExpandRecurrentParamsPreservesOldLogitsAndLeavesLSTMUntouched(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 10))
	oldInputSize, hiddenSize, oldOutputSize := 6, 5, 4
	rp := NewRecurrentParams(rng, oldInputSize, hiddenSize, oldOutputSize, 2)

	newInputSize, newOutputSize := 10, 7
	expanded, err := ExpandRecurrentParams(rp, newInputSize, newOutputSize)
	require.NoError(t, err)

	assert.Equal(t, newInputSize, expanded.PolicyTrunk.InputSize())
	assert.Equal(t, newOutputSize, expanded.PolicyTrunk.OutputSize())
	assert.Equal(t, newInputSize, expanded.ValueTrunk.InputSize())
	// ValueTrunk.OutputSize() is its own Wpi.Rows, not Wv.Rows (the
	// actual scalar value head, always width 1 regardless) - NewParams
	// always allocates a full policy-shaped Wpi/Bpi even for a trunk
	// only ever used for its Wv/Bv, and that allocation is simply
	// unused dead weight here. ExpandRecurrentParams deliberately keeps
	// it at its own prior size (passing the value trunk's own current
	// OutputSize back in as "new"), since nothing reads it either way.
	assert.Equal(t, oldOutputSize, expanded.ValueTrunk.OutputSize())

	assert.Equal(t, rp.PolicyLSTM.WxF.Data, expanded.PolicyLSTM.WxF.Data)
	assert.Equal(t, rp.PolicyLSTM.WhF.Data, expanded.PolicyLSTM.WhF.Data)
	assert.Equal(t, rp.ValueLSTM.WxI.Data, expanded.ValueLSTM.WxI.Data)
	assert.Same(t, expanded.Trunk, expanded.PolicyTrunk, "Trunk must alias PolicyTrunk, matching NewRecurrentParams' own convention")
	assert.Same(t, expanded.LSTM, expanded.PolicyLSTM, "LSTM must alias PolicyLSTM, matching NewRecurrentParams' own convention")
}

// --- EnvironmentID expansion detection ---

func TestParseActionObsID(t *testing.T) {
	actions, obs, template, ok := parseActionObsID("mc-agent-rlenv:actions=5:obs=25:norm=1")
	require.True(t, ok)
	assert.Equal(t, 5, actions)
	assert.Equal(t, 25, obs)
	assert.Equal(t, "mc-agent-rlenv:actions=#:obs=#:norm=1", template)

	_, _, _, ok = parseActionObsID("snake:36")
	assert.False(t, ok, "an ID with no actions=/obs= fragment must report ok=false")
}

func TestExpandTarget(t *testing.T) {
	newActions, newObs, ok := ExpandTarget(
		"mc-agent-rlenv:actions=5:obs=25:norm=1",
		"mc-agent-rlenv:actions=19:obs=76:norm=1",
	)
	require.True(t, ok)
	assert.Equal(t, 19, newActions)
	assert.Equal(t, 76, newObs)

	_, _, ok = ExpandTarget(
		"mc-agent-rlenv:actions=5:obs=25:norm=1",
		"mc-agent-rlenv:actions=5:obs=25:norm=1",
	)
	assert.False(t, ok, "an exact match is not an expansion target - Load/RecurrentLoad already handle it")

	_, _, ok = ExpandTarget(
		"mc-agent-rlenv:actions=19:obs=76:norm=1",
		"mc-agent-rlenv:actions=5:obs=25:norm=1",
	)
	assert.False(t, ok, "shrinking must not be reported as expandable")

	_, _, ok = ExpandTarget(
		"mc-agent-rlenv:actions=5:obs=25",
		"mc-agent-rlenv:actions=19:obs=76:norm=1",
	)
	assert.False(t, ok, "a mismatch outside the actions=/obs= fragment (here, the norm=1 suffix) must not be treated as expandable")

	_, _, ok = ExpandTarget("snake:36", "mc-agent-rlenv:actions=19:obs=76")
	assert.False(t, ok, "an ID with no actions=/obs= fragment at all is never expandable")
}

// --- LoadExpanding / RecurrentLoadExpanding ---

func TestLoadExpandingUpgradesAnOlderShapeAutomatically(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	p := NewParams(rng, 25, 8, 5, 2)

	var buf bytes.Buffer
	require.NoError(t, p.Save(&buf, "mc-agent-rlenv:actions=5:obs=25", checkpoint.Metadata{Epoch: 3}))

	loaded, metadata, err := LoadExpanding(bytes.NewReader(buf.Bytes()), "mc-agent-rlenv:actions=19:obs=76")
	require.NoError(t, err)
	assert.Equal(t, 76, loaded.InputSize())
	assert.Equal(t, 19, loaded.OutputSize())
	assert.Equal(t, 3, metadata.Epoch, "run-progress metadata must survive the upgrade")
}

func TestLoadExpandingStillRejectsAnIncompatibleShape(t *testing.T) {
	rng := rand.New(rand.NewPCG(13, 14))
	p := NewParams(rng, 25, 8, 5, 2)

	var buf bytes.Buffer
	require.NoError(t, p.Save(&buf, "mc-agent-rlenv:actions=5:obs=25", checkpoint.Metadata{}))

	// A different hidden size isn't an actions=/obs=-only difference -
	// rather than guess, this must fail exactly like Load would.
	_, _, err := LoadExpanding(bytes.NewReader(buf.Bytes()), "different-env:actions=19:obs=76")
	assert.Error(t, err)
}

func TestLoadExpandingRejectsShrinking(t *testing.T) {
	rng := rand.New(rand.NewPCG(15, 16))
	p := NewParams(rng, 76, 8, 19, 2)

	var buf bytes.Buffer
	require.NoError(t, p.Save(&buf, "mc-agent-rlenv:actions=19:obs=76", checkpoint.Metadata{}))

	_, _, err := LoadExpanding(bytes.NewReader(buf.Bytes()), "mc-agent-rlenv:actions=5:obs=25")
	assert.Error(t, err, "a checkpoint bigger than what's expected must not be silently truncated")
}

func TestLoadExpandingMatchesPlainLoadWhenShapeAlreadyMatches(t *testing.T) {
	rng := rand.New(rand.NewPCG(17, 18))
	p := NewParams(rng, 25, 8, 5, 2)

	var buf bytes.Buffer
	require.NoError(t, p.Save(&buf, "mc-agent-rlenv:actions=5:obs=25", checkpoint.Metadata{}))

	loaded, _, err := LoadExpanding(bytes.NewReader(buf.Bytes()), "mc-agent-rlenv:actions=5:obs=25")
	require.NoError(t, err)
	assert.Equal(t, p.Wpi.Data, loaded.Wpi.Data)
}

func TestRecurrentLoadExpandingUpgradesAnOlderShapeAutomatically(t *testing.T) {
	rng := rand.New(rand.NewPCG(19, 20))
	rp := NewRecurrentParams(rng, 25, 8, 5, 2)

	var buf bytes.Buffer
	require.NoError(t, rp.Save(&buf, "mc-agent-rlenv:actions=5:obs=25", checkpoint.Metadata{Epoch: 7}))

	loaded, metadata, err := RecurrentLoadExpanding(bytes.NewReader(buf.Bytes()), "mc-agent-rlenv:actions=19:obs=76")
	require.NoError(t, err)
	assert.Equal(t, 76, loaded.PolicyTrunk.InputSize())
	assert.Equal(t, 19, loaded.PolicyTrunk.OutputSize())
	assert.Equal(t, 76, loaded.ValueTrunk.InputSize())
	assert.Equal(t, 7, metadata.Epoch)
}
