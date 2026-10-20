package actorcritic

import (
	"fmt"

	"github.com/reallyoldfogie/cRL-go/pkg/autograd"
	"github.com/reallyoldfogie/cRL-go/pkg/mat"
)

// sub, mul, sigmoid, and tanh extend chain (network.go) with the ops
// buildRecurrentForward needs beyond the plain MLP's matMul/add/relu/
// softmax.
func (c *chain) sub(a, b *autograd.Var) *autograd.Var {
	if c.err != nil {
		return nil
	}
	v, err := autograd.Sub(a, b)
	c.err = err
	return v
}

func (c *chain) mul(a, b *autograd.Var) *autograd.Var {
	if c.err != nil {
		return nil
	}
	v, err := autograd.Mul(a, b)
	c.err = err
	return v
}

func (c *chain) sigmoid(a *autograd.Var) *autograd.Var {
	if c.err != nil {
		return nil
	}
	v, err := autograd.Sigmoid(a)
	c.err = err
	return v
}

func (c *chain) tanh(a *autograd.Var) *autograd.Var {
	if c.err != nil {
		return nil
	}
	v, err := autograd.Tanh(a)
	c.err = err
	return v
}

// lstmVars wraps an LSTMLayer's twelve matrices as autograd.Var leaves.
type lstmVars struct {
	WxF, WhF, BF *autograd.Var
	WxI, WhI, BI *autograd.Var
	WxO, WhO, BO *autograd.Var
	WxG, WhG, BG *autograd.Var
}

func (l lstmVars) all() []*autograd.Var {
	return []*autograd.Var{
		l.WxF, l.WhF, l.BF,
		l.WxI, l.WhI, l.BI,
		l.WxO, l.WhO, l.BO,
		l.WxG, l.WhG, l.BG,
	}
}

// buildLSTMVars wraps layer's matrices as autograd.Var leaves, using wrap
// (autograd.Constant for inference, autograd.Parameter for training) for
// every matrix — mirrors buildParamVars (network.go) for the trunk.
func buildLSTMVars(layer *LSTMLayer, wrap func(*mat.Matrix) *autograd.Var) lstmVars {
	return lstmVars{
		WxF: wrap(layer.WxF), WhF: wrap(layer.WhF), BF: wrap(layer.BF),
		WxI: wrap(layer.WxI), WhI: wrap(layer.WhI), BI: wrap(layer.BI),
		WxO: wrap(layer.WxO), WhO: wrap(layer.WhO), BO: wrap(layer.BO),
		WxG: wrap(layer.WxG), WhG: wrap(layer.WhG), BG: wrap(layer.BG),
	}
}

// buildRecurrentForward wires up the MLP trunk (identical to
// buildForward's own loop over pv.Hidden), an LSTM step consuming the
// trunk's output alongside hPrev/cPrev, and both heads computed from the
// LSTM's new hidden state hOut (rather than from the trunk's own output
// directly, as buildForward's heads are):
//
//	f = sigmoid(WxF*a + WhF*hPrev + BF)   // forget gate
//	i = sigmoid(WxI*a + WhI*hPrev + BI)   // input gate
//	o = sigmoid(WxO*a + WhO*hPrev + BO)   // output gate
//	g = tanh(WxG*a + WhG*hPrev + BG)      // candidate memory content
//	cOut = f*cPrev + i*g
//	hOut = o*tanh(cOut)
//
// policyMaskBias, when non-nil, is added to the policy head's
// pre-softmax logits before Softmax, exactly as buildForward's own
// parameter of the same name does — see
// docs/plans/19-training-time-action-masking.md.
func buildRecurrentForward(trunkInput *autograd.Var, pv paramVars, hPrev, cPrev *autograd.Var, lv lstmVars, policyMaskBias *autograd.Var) (policyOutput, valueOutput, hOut, cOut *autograd.Var, err error) {
	c := &chain{}

	a := trunkInput
	for _, layer := range pv.Hidden {
		z := c.matMul(layer.W, a)
		zb := c.add(z, layer.B)
		a = c.relu(zb)
	}

	gate := func(wx, wh, b *autograd.Var) *autograd.Var {
		z := c.add(c.add(c.matMul(wx, a), c.matMul(wh, hPrev)), b)
		return z
	}

	f := c.sigmoid(gate(lv.WxF, lv.WhF, lv.BF))
	i := c.sigmoid(gate(lv.WxI, lv.WhI, lv.BI))
	o := c.sigmoid(gate(lv.WxO, lv.WhO, lv.BO))
	g := c.tanh(gate(lv.WxG, lv.WhG, lv.BG))

	cOut = c.add(c.mul(f, cPrev), c.mul(i, g))
	hOut = c.mul(o, c.tanh(cOut))

	zpi := c.matMul(pv.Wpi, hOut)
	zpib := c.add(zpi, pv.Bpi)
	policyLogits := zpib
	if policyMaskBias != nil {
		policyLogits = c.add(zpib, policyMaskBias)
	}
	policyOutput = c.softmax(policyLogits)

	zv := c.matMul(pv.Wv, hOut)
	valueOutput = c.add(zv, pv.Bv)

	return policyOutput, valueOutput, hOut, cOut, c.err
}

// RecurrentInferenceNetwork is a forward-only, single-step computation
// graph over a shared, read-only RecurrentParams (mirroring
// InferenceNetwork, see network.go). Unlike InferenceNetwork, it also
// takes HPrevIn/CPrevIn as explicit leaf inputs and exposes HOut/COut as
// outputs: the hidden/cell state is per-episode runtime state owned by
// the caller (see RecurrentParams' own doc comment), not something this
// network tracks itself. A caller drives an episode by feeding last
// step's HOut/COut into the next step's HPrevIn/CPrevIn (both zeroed at
// episode start) between Forward calls.
type RecurrentInferenceNetwork struct {
	Input                        *autograd.Var
	PolicyHPrevIn, PolicyCPrevIn *autograd.Var
	ValueHPrevIn, ValueCPrevIn   *autograd.Var
	HPrevIn                      *autograd.Var
	CPrevIn                      *autograd.Var

	PolicyOutput           *autograd.Var
	ValueOutput            *autograd.Var
	HOut                   *autograd.Var
	COut                   *autograd.Var
	PolicyHOut, PolicyCOut *autograd.Var
	ValueHOut, ValueCOut   *autograd.Var

	Graph *autograd.Graph
}

// NewRecurrentInferenceNetwork builds a RecurrentInferenceNetwork over
// params.
func NewRecurrentInferenceNetwork(params *RecurrentParams) (*RecurrentInferenceNetwork, error) {
	input := autograd.NewVar(params.InputSize(), 1, autograd.FlagNone)
	policyHPrevIn := autograd.NewVar(params.HiddenSize(), 1, autograd.FlagNone)
	policyCPrevIn := autograd.NewVar(params.HiddenSize(), 1, autograd.FlagNone)
	valueHPrevIn := autograd.NewVar(params.HiddenSize(), 1, autograd.FlagNone)
	valueCPrevIn := autograd.NewVar(params.HiddenSize(), 1, autograd.FlagNone)

	policyOutput, _, policyHOut, policyCOut, err := buildRecurrentForward(input, buildParamVars(params.policyTrunk(), autograd.Constant), policyHPrevIn, policyCPrevIn, buildLSTMVars(params.policyLSTM(), autograd.Constant), nil)
	if err != nil {
		return nil, err
	}
	_, valueOutput, valueHOut, valueCOut, err := buildRecurrentForward(input, buildParamVars(params.valueTrunk(), autograd.Constant), valueHPrevIn, valueCPrevIn, buildLSTMVars(params.valueLSTM(), autograd.Constant), nil)
	if err != nil {
		return nil, err
	}

	return &RecurrentInferenceNetwork{
		Input:         input,
		PolicyHPrevIn: policyHPrevIn, PolicyCPrevIn: policyCPrevIn,
		ValueHPrevIn: valueHPrevIn, ValueCPrevIn: valueCPrevIn,
		HPrevIn: policyHPrevIn, CPrevIn: policyCPrevIn,
		PolicyOutput: policyOutput,
		ValueOutput:  valueOutput,
		HOut:         policyHOut, COut: policyCOut,
		PolicyHOut: policyHOut, PolicyCOut: policyCOut,
		ValueHOut: valueHOut, ValueCOut: valueCOut,
		Graph: autograd.BuildGraphMulti(policyOutput, valueOutput, policyHOut, policyCOut, valueHOut, valueCOut),
	}, nil
}

// RecurrentChunkTrainingNetwork wraps a RecurrentParams' trunk and LSTM
// matrices as gradient-accumulating (autograd.Parameter) leaves, with the
// forward computation unrolled ChunkLen timesteps — the number of
// consecutive steps of one rollout trained on in a single
// Forward/Backward call (see
// docs/plans/20-configurable-depth-step-memory-and-dynamic-architecture.md,
// Part B, for why recurrence forces this rather than training one step
// at a time the way the non-recurrent TrainingNetwork does).
//
// Exactly one copy of every weight Var is built (ChunkLen unrolled
// applications all reference the same Parameter Vars — this is standard
// weight-sharing across time, and the existing autograd engine already
// accumulates gradients correctly across however many times a Var is
// used as input within one graph; verified directly against
// pkg/autograd's matMulOp.Backward and mat.Matrix.MatMul's zeroOut
// parameter before this type was written).
//
// Like TrainingNetwork, this does not build a Graph or own a Loss: PPO's
// objective depends on rollout data (advantages, old log-probs, return
// targets) this package doesn't produce, so pkg/ppo composes the actual
// per-timestep loss and the one Graph spanning the whole chunk on top of
// ChunkLen PolicyOutput/ValueOutput pairs exposed here.
type RecurrentChunkTrainingNetwork struct {
	ChunkLen int

	// Input is the per-timestep observation input, length ChunkLen.
	Input []*autograd.Var
	// MaskBias is the per-timestep action-legality mask bias (see
	// TrainingNetwork.SetActionMask's doc comment for what this means
	// and why — same mechanism, one copy per timestep), length ChunkLen.
	MaskBias []*autograd.Var
	// HPrevIn, CPrevIn are the chunk's starting hidden/cell state — fed
	// in once, not per-timestep, since every later timestep's state
	// flows through the unrolled graph's own edges.
	HPrevIn, CPrevIn             *autograd.Var
	PolicyHPrevIn, PolicyCPrevIn *autograd.Var
	ValueHPrevIn, ValueCPrevIn   *autograd.Var

	// PolicyOutput, ValueOutput are the per-timestep head outputs a
	// caller composes a loss on top of, length ChunkLen.
	PolicyOutput           []*autograd.Var
	ValueOutput            []*autograd.Var
	PolicyHOut, PolicyCOut []*autograd.Var
	ValueHOut, ValueCOut   []*autograd.Var

	params      paramVars
	lstm        lstmVars
	valueParams paramVars
	valueLSTM   lstmVars
}

// NewRecurrentChunkTrainingNetwork builds a RecurrentChunkTrainingNetwork
// over params, unrolled chunkLen timesteps. chunkLen must be at least 1.
func NewRecurrentChunkTrainingNetwork(params *RecurrentParams, chunkLen int) (*RecurrentChunkTrainingNetwork, error) {
	if chunkLen < 1 {
		return nil, fmt.Errorf("actorcritic: NewRecurrentChunkTrainingNetwork: chunkLen must be at least 1, got %d", chunkLen)
	}

	policyParams := buildParamVars(params.policyTrunk(), autograd.Parameter)
	valueParams := buildParamVars(params.valueTrunk(), autograd.Parameter)
	policyLSTM := buildLSTMVars(params.policyLSTM(), autograd.Parameter)
	valueLSTM := buildLSTMVars(params.valueLSTM(), autograd.Parameter)

	policyHPrevIn := autograd.NewVar(params.HiddenSize(), 1, autograd.FlagNone)
	policyCPrevIn := autograd.NewVar(params.HiddenSize(), 1, autograd.FlagNone)
	valueHPrevIn := autograd.NewVar(params.HiddenSize(), 1, autograd.FlagNone)
	valueCPrevIn := autograd.NewVar(params.HiddenSize(), 1, autograd.FlagNone)

	n := &RecurrentChunkTrainingNetwork{
		ChunkLen: chunkLen,
		Input:    make([]*autograd.Var, chunkLen),
		MaskBias: make([]*autograd.Var, chunkLen),
		HPrevIn:  policyHPrevIn, CPrevIn: policyCPrevIn,
		PolicyHPrevIn: policyHPrevIn, PolicyCPrevIn: policyCPrevIn,
		ValueHPrevIn: valueHPrevIn, ValueCPrevIn: valueCPrevIn,
		PolicyOutput: make([]*autograd.Var, chunkLen),
		ValueOutput:  make([]*autograd.Var, chunkLen),
		PolicyHOut:   make([]*autograd.Var, chunkLen), PolicyCOut: make([]*autograd.Var, chunkLen),
		ValueHOut: make([]*autograd.Var, chunkLen), ValueCOut: make([]*autograd.Var, chunkLen),
		params: policyParams, lstm: policyLSTM,
		valueParams: valueParams, valueLSTM: valueLSTM,
	}

	policyH, policyC := policyHPrevIn, policyCPrevIn
	valueH, valueC := valueHPrevIn, valueCPrevIn
	for t := range chunkLen {
		input := autograd.NewVar(params.InputSize(), 1, autograd.FlagNone)
		maskBias := autograd.NewVar(params.OutputSize(), 1, autograd.FlagNone)

		policyOutput, _, policyHOut, policyCOut, err := buildRecurrentForward(input, policyParams, policyH, policyC, policyLSTM, maskBias)
		if err != nil {
			return nil, err
		}
		_, valueOutput, valueHOut, valueCOut, err := buildRecurrentForward(input, valueParams, valueH, valueC, valueLSTM, nil)
		if err != nil {
			return nil, err
		}

		n.Input[t] = input
		n.MaskBias[t] = maskBias
		n.PolicyOutput[t] = policyOutput
		n.ValueOutput[t] = valueOutput
		n.PolicyHOut[t], n.PolicyCOut[t] = policyHOut, policyCOut
		n.ValueHOut[t], n.ValueCOut[t] = valueHOut, valueCOut
		policyH, policyC = policyHOut, policyCOut
		valueH, valueC = valueHOut, valueCOut
	}

	return n, nil
}

// SetActionMask overwrites timestep t's MaskBias ahead of one Forward/
// Backward call, exactly like TrainingNetwork.SetActionMask (see its own
// doc comment).
func (n *RecurrentChunkTrainingNetwork) SetActionMask(t int, mask []bool) {
	n.MaskBias[t].Val.Clear()
	for i, allowed := range mask {
		if !allowed {
			n.MaskBias[t].Val.Data[i] = maskedActionLogitBias
		}
	}
}

// SetChunkState overwrites the chunk's starting hidden/cell state ahead
// of one Forward/Backward call.
func (n *RecurrentChunkTrainingNetwork) SetChunkState(hPrev, cPrev []float32) {
	copy(n.HPrevIn.Val.Data, hPrev)
	copy(n.CPrevIn.Val.Data, cPrev)
	copy(n.ValueHPrevIn.Val.Data, hPrev)
	copy(n.ValueCPrevIn.Val.Data, cPrev)
}

// SetSplitChunkState seeds the policy and value recurrent branches independently.
func (n *RecurrentChunkTrainingNetwork) SetSplitChunkState(policyH, policyC, valueH, valueC []float32) {
	copy(n.PolicyHPrevIn.Val.Data, policyH)
	copy(n.PolicyCPrevIn.Val.Data, policyC)
	copy(n.ValueHPrevIn.Val.Data, valueH)
	copy(n.ValueCPrevIn.Val.Data, valueC)
}

// Parameters returns every one of the network's trainable weight and
// bias Vars (the trunk's and the LSTM's, the same set ZeroGrad and
// ApplyGradientStep iterate internally) — fixed in number regardless of
// ChunkLen, since weights are shared/reused across timesteps rather than
// duplicated per timestep.
func (n *RecurrentChunkTrainingNetwork) Parameters() []*autograd.Var {
	params := append(n.params.all(), n.lstm.all()...)
	return append(params, append(n.valueParams.all(), n.valueLSTM.all()...)...)
}

// ZeroGrad clears every parameter's accumulated gradient.
func (n *RecurrentChunkTrainingNetwork) ZeroGrad() {
	for _, p := range n.Parameters() {
		p.Grad.Clear()
	}
}

// ApplyGradientStep performs one manual (vanilla) SGD step, identical in
// shape to TrainingNetwork.ApplyGradientStep.
func (n *RecurrentChunkTrainingNetwork) ApplyGradientStep(learningRate float32, sampleCount int) {
	if sampleCount == 0 {
		return
	}

	scale := learningRate / float32(sampleCount)
	for _, p := range n.Parameters() {
		p.Grad.Scale(scale)
		_ = p.Val.Sub(p.Val, p.Grad)
	}
}
