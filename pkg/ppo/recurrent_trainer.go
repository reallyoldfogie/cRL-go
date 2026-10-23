package ppo

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"sync"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/config"
	"github.com/reallyoldfogie/cRL-go/pkg/reinforce"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

// RecurrentTrainer runs the same PPO training loop as Trainer (rollout
// collection, GAE, shuffled-minibatch Adam updates) but over an
// actorcritic.RecurrentParams network: it carries an LSTM hidden/cell
// state across an episode's steps, which forces chunked,
// backprop-through-time training instead of per-step training — see
// docs/plans/20-configurable-depth-step-memory-and-dynamic-architecture.md,
// Part B, and RecurrentTrainingNetwork's own doc comment for why.
//
// Supports all three of Trainer's own rollout-collection strategies
// (plain EnvFactory, one persistent environment, a persistent
// environment pool) — see NewRecurrent/NewRecurrentWithPersistentEnv/
// NewRecurrentWithPersistentEnvPool and collectRecurrentRollouts'
// dispatch, mirroring Trainer's own exactly. Built on
// collectRecurrentTrajectoryFromEnv's existing "takes an
// already-constructed env" split (recurrent_rollout.go), which was
// deliberately shaped that way in Part B specifically so these
// persistent-env variants could be added later without touching rollout
// collection itself.
type RecurrentTrainer struct {
	settings   config.Settings
	envFactory reinforce.EnvFactory

	// persistentEnv/persistentEnvs mirror Trainer's own identically-named
	// fields exactly — see Trainer's own doc comment for what each means
	// and why at most one of envFactory/persistentEnv/persistentEnvs is
	// ever set.
	persistentEnv  rl.Environment
	persistentEnvs []rl.Environment

	params  *actorcritic.RecurrentParams
	network *RecurrentTrainingNetwork

	// policyAdam and valueAdam clip and step the policy and value
	// branches independently (each against settings.MaxGradNorm on only
	// its own gradient), rather than sharing one Adam/clip over both
	// branches' gradients combined — see trainOnChunkMinibatch. A value
	// loss spike no longer shrinks the policy's own gradient along with
	// it (and vice versa): the two branches never share a Var (see
	// actorcritic.RecurrentChunkTrainingNetwork.PolicyParameters/
	// ValueParameters), so there is nothing for a combined clip to have
	// ever legitimately coupled.
	policyAdam *actorcritic.Adam
	valueAdam  *actorcritic.Adam
}

// NewRecurrent constructs a RecurrentTrainer from settings and
// envFactory, mirroring New's shape (including the initialParams
// resume-from-checkpoint convention). settings.RecurrentChunkLen must be
// positive. Collects rollouts in parallel across settings.Workers
// goroutines, rebuilding an environment per episode; see
// NewRecurrentWithPersistentEnv for a long-lived-environment
// alternative.
func NewRecurrent(settings config.Settings, envFactory reinforce.EnvFactory, initialParams *actorcritic.RecurrentParams) (*RecurrentTrainer, error) {
	initRNG := reinforce.WorkerRNG(settings.Seed, 0, initWorkerIndex)

	env, err := envFactory(initRNG)
	if err != nil {
		return nil, fmt.Errorf("ppo: constructing environment: %w", err)
	}

	trainer, err := newRecurrentTrainer(settings, env, initRNG, initialParams)
	if err != nil {
		return nil, err
	}
	trainer.envFactory = envFactory
	return trainer, nil
}

// NewRecurrentWithPersistentEnv constructs a RecurrentTrainer exactly
// like NewRecurrent, except it builds one environment, once, via
// persistentFactory, and reuses it across every episode of every epoch
// (collectRecurrentRolloutsSequential resets it between episodes instead
// of rebuilding it) — mirroring Trainer.NewWithPersistentEnv exactly.
func NewRecurrentWithPersistentEnv(settings config.Settings, persistentFactory reinforce.PersistentEnvFactory, initialParams *actorcritic.RecurrentParams) (*RecurrentTrainer, error) {
	initRNG := reinforce.WorkerRNG(settings.Seed, 0, initWorkerIndex)

	env, err := persistentFactory(initRNG)
	if err != nil {
		return nil, fmt.Errorf("ppo: constructing persistent environment: %w", err)
	}

	trainer, err := newRecurrentTrainer(settings, env, initRNG, initialParams)
	if err != nil {
		return nil, err
	}
	trainer.persistentEnv = env
	return trainer, nil
}

// NewRecurrentWithPersistentEnvPool constructs a RecurrentTrainer
// exactly like NewRecurrentWithPersistentEnv, except it drives N
// already-constructed, long-lived environments (envs) concurrently with
// each other instead of exactly one sequentially — mirroring
// Trainer.NewWithPersistentEnvPool exactly, including its own "callers
// own envs' lifecycle" and "concurrency bounded to len(envs), not
// settings.Workers" contracts.
func NewRecurrentWithPersistentEnvPool(settings config.Settings, envs []rl.Environment, initialParams *actorcritic.RecurrentParams) (*RecurrentTrainer, error) {
	if len(envs) == 0 {
		return nil, fmt.Errorf("ppo: envs must contain at least one environment")
	}

	initRNG := reinforce.WorkerRNG(settings.Seed, 0, initWorkerIndex)

	trainer, err := newRecurrentTrainer(settings, envs[0], initRNG, initialParams)
	if err != nil {
		return nil, err
	}
	trainer.persistentEnvs = envs
	return trainer, nil
}

// newRecurrentTrainer builds the params/network/optimizer shared by
// NewRecurrent/NewRecurrentWithPersistentEnv/
// NewRecurrentWithPersistentEnvPool from an already-constructed env,
// mirroring newTrainer exactly — not itself stored on the returned
// RecurrentTrainer; callers set envFactory/persistentEnv/persistentEnvs
// afterward depending on which constructor they came from.
func newRecurrentTrainer(settings config.Settings, env rl.Environment, initRNG *rand.Rand, initialParams *actorcritic.RecurrentParams) (*RecurrentTrainer, error) {
	if err := settings.Validate(); err != nil {
		return nil, err
	}
	if settings.RecurrentChunkLen <= 0 {
		return nil, fmt.Errorf("ppo: settings.RecurrentChunkLen must be positive, got %d", settings.RecurrentChunkLen)
	}

	params := initialParams
	if params == nil {
		params = actorcritic.NewRecurrentParams(initRNG, env.ObservationSize(), settings.HiddenSize, env.ActionSpace(), 2)
	} else if err := validateRecurrentParamsShape(params, env, settings); err != nil {
		return nil, err
	}

	network, err := NewRecurrentTrainingNetwork(params, settings.RecurrentChunkLen, LossConfig{
		ClipEpsilon: settings.ClipEpsilon,
		EntropyCoef: settings.EntropyCoef,
		ValueCoef:   settings.ValueCoef,
	})
	if err != nil {
		return nil, fmt.Errorf("ppo: building recurrent training network: %w", err)
	}

	return &RecurrentTrainer{
		settings:   settings,
		params:     params,
		network:    network,
		policyAdam: actorcritic.NewAdamWithGradClip(network.Actor.PolicyParameters(), settings.LearningRate, settings.MaxGradNorm),
		valueAdam:  actorcritic.NewAdamWithGradClip(network.Actor.ValueParameters(), settings.LearningRate, settings.MaxGradNorm),
	}, nil
}

// validateRecurrentParamsShape mirrors validateParamsShape for
// actorcritic.RecurrentParams.
func validateRecurrentParamsShape(params *actorcritic.RecurrentParams, env rl.Environment, settings config.Settings) error {
	if params.InputSize() != env.ObservationSize() {
		return fmt.Errorf(
			"ppo: initial recurrent params input size %d does not match environment observation size %d",
			params.InputSize(), env.ObservationSize(),
		)
	}
	if params.HiddenSize() != settings.HiddenSize {
		return fmt.Errorf(
			"ppo: initial recurrent params hidden size %d does not match settings hidden size %d",
			params.HiddenSize(), settings.HiddenSize,
		)
	}
	if params.OutputSize() != env.ActionSpace() {
		return fmt.Errorf(
			"ppo: initial recurrent params output size %d does not match environment action space %d",
			params.OutputSize(), env.ActionSpace(),
		)
	}
	return nil
}

// Params returns the RecurrentTrainer's current actor-critic parameters.
func (tr *RecurrentTrainer) Params() *actorcritic.RecurrentParams {
	return tr.params
}

// OptimizerState returns the Adam continuation state for checkpointing:
// the policy and value optimizers' moments concatenated policy-then-value,
// the same order Parameters() already lists them in, so this round-trips
// through RestoreOptimizerState (including an older checkpoint saved
// before the two branches had independent optimizers, since that single
// combined state was built from this exact parameter order already).
func (tr *RecurrentTrainer) OptimizerState() actorcritic.AdamState {
	policy := tr.policyAdam.State()
	value := tr.valueAdam.State()
	return actorcritic.AdamState{
		StepCount: policy.StepCount,
		Moment1:   append(append([][]float32{}, policy.Moment1...), value.Moment1...),
		Moment2:   append(append([][]float32{}, policy.Moment2...), value.Moment2...),
	}
}

// RestoreOptimizerState restores both optimizers' moments from a
// compatible checkpoint, splitting the concatenated state at the policy
// branch's parameter count — see OptimizerState.
func (tr *RecurrentTrainer) RestoreOptimizerState(state actorcritic.AdamState) error {
	policyCount := len(tr.network.Actor.PolicyParameters())
	if len(state.Moment1) < policyCount || len(state.Moment2) < policyCount {
		return fmt.Errorf("ppo: recurrent optimizer state has %d/%d tensors, need at least %d for the policy branch", len(state.Moment1), len(state.Moment2), policyCount)
	}
	policyState := actorcritic.AdamState{StepCount: state.StepCount, Moment1: state.Moment1[:policyCount], Moment2: state.Moment2[:policyCount]}
	valueState := actorcritic.AdamState{StepCount: state.StepCount, Moment1: state.Moment1[policyCount:], Moment2: state.Moment2[policyCount:]}
	if err := tr.policyAdam.RestoreState(policyState); err != nil {
		return fmt.Errorf("ppo: restoring policy optimizer state: %w", err)
	}
	if err := tr.valueAdam.RestoreState(valueState); err != nil {
		return fmt.Errorf("ppo: restoring value optimizer state: %w", err)
	}
	return nil
}

// RunEpoch runs one full recurrent-PPO training epoch and returns
// summary statistics, mirroring Trainer.RunEpoch's own doc comment.
func (tr *RecurrentTrainer) RunEpoch(ctx context.Context, epoch int) (EpochStats, error) {
	rollouts, err := tr.collectRollouts(ctx, epoch)
	if err != nil {
		return EpochStats{}, err
	}

	var chunks []recurrentChunk
	var returnSum float32
	var rawAdvantages []float32
	for _, rollout := range rollouts {
		advantages, returns := ComputeGAE(rollout.Rollout, tr.settings.Gamma, tr.settings.GAELambda)
		rawAdvantages = append(rawAdvantages, advantages...)
		chunks = append(chunks, chunkRollout(rollout, advantages, returns, tr.settings.RecurrentChunkLen, tr.params.InputSize())...)
		if len(returns) > 0 {
			returnSum += returns[0]
		}
	}

	normalizeRecurrentAdvantages(chunks)

	shuffleRNG := reinforce.WorkerRNG(tr.settings.Seed, epoch, shuffleWorkerIndex)
	updateCount := 0
	var diagnostics epochDiagnostics
	for range tr.settings.PPOEpochs {
		shuffleRNG.Shuffle(len(chunks), func(i, j int) {
			chunks[i], chunks[j] = chunks[j], chunks[i]
		})

		for start := 0; start < len(chunks); start += tr.settings.MinibatchSize {
			end := min(start+tr.settings.MinibatchSize, len(chunks))

			tr.params.Lock()
			diagnosticsForBatch := tr.trainOnChunkMinibatch(chunks[start:end])
			tr.params.Unlock()
			diagnostics.policyLossSum += diagnosticsForBatch.policyLossSum
			diagnostics.valueLossSum += diagnosticsForBatch.valueLossSum
			diagnostics.entropySum += diagnosticsForBatch.entropySum
			diagnostics.steps += diagnosticsForBatch.steps
			diagnostics.gradientNormSum += diagnosticsForBatch.gradientNormSum
			diagnostics.appliedNormSum += diagnosticsForBatch.appliedNormSum
			diagnostics.policyGradNormSum += diagnosticsForBatch.policyGradNormSum
			diagnostics.valueGradNormSum += diagnosticsForBatch.valueGradNormSum
			diagnostics.updates += diagnosticsForBatch.updates
			updateCount++
		}
	}

	policyLoss, valueLoss, entropy, gradientNorm, appliedNorm, policyGradientNorm, valueGradientNorm := diagnostics.averages()
	advantageMean, advantageStd, advantageMin, advantageMax := finishAdvantageStats(rawAdvantages)
	return EpochStats{
		Epoch:               epoch,
		AverageReturn:       returnSum / float32(len(rollouts)),
		SampleCount:         recurrentRealStepCount(chunks),
		UpdateCount:         updateCount,
		PolicyLoss:          policyLoss,
		ValueLoss:           valueLoss,
		Entropy:             entropy,
		AdvantageMean:       advantageMean,
		AdvantageStd:        advantageStd,
		AdvantageMin:        advantageMin,
		AdvantageMax:        advantageMax,
		GradientNorm:        gradientNorm,
		AppliedGradientNorm: appliedNorm,
		PolicyGradientNorm:  policyGradientNorm,
		ValueGradientNorm:   valueGradientNorm,
	}, nil
}

// collectRollouts collects settings.RolloutSize trajectories for this
// epoch, dispatching to collectRecurrentRolloutsPool if tr was built via
// NewRecurrentWithPersistentEnvPool, collectRecurrentRolloutsSequential
// if tr was built via NewRecurrentWithPersistentEnv, or
// collectRecurrentRolloutsParallel (the original, per-episode-
// construction path) otherwise — mirroring Trainer.collectRollouts'
// dispatch exactly.
func (tr *RecurrentTrainer) collectRollouts(ctx context.Context, epoch int) ([]*RecurrentRollout, error) {
	if len(tr.persistentEnvs) > 0 {
		return tr.collectRecurrentRolloutsPool(ctx, epoch)
	}
	if tr.persistentEnv != nil {
		return tr.collectRecurrentRolloutsSequential(ctx, epoch)
	}
	return tr.collectRecurrentRolloutsParallel(ctx, epoch)
}

// collectRecurrentRolloutsParallel collects settings.RolloutSize
// trajectories in parallel, bounded to settings.Workers concurrent
// goroutines, mirroring Trainer.collectRolloutsParallel exactly but
// calling collectRecurrentTrajectory.
func (tr *RecurrentTrainer) collectRecurrentRolloutsParallel(ctx context.Context, epoch int) ([]*RecurrentRollout, error) {
	rollouts := make([]*RecurrentRollout, tr.settings.RolloutSize)
	errs := make([]error, tr.settings.RolloutSize)

	var wg sync.WaitGroup
	semaphore := make(chan struct{}, tr.settings.Workers)

	for i := range tr.settings.RolloutSize {
		wg.Add(1)
		semaphore <- struct{}{}

		go func(index int) {
			defer wg.Done()
			defer func() { <-semaphore }()

			rng := reinforce.WorkerRNG(tr.settings.Seed, epoch, index)
			rollout, err := collectRecurrentTrajectory(ctx, tr.params, tr.envFactory, tr.settings.EpisodeLen, rng)
			rollouts[index] = rollout
			errs[index] = err
		}(i)
	}
	wg.Wait()

	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	return rollouts, nil
}

// collectRecurrentRolloutsSequential collects settings.RolloutSize
// trajectories one at a time, on the calling goroutine, against
// tr.persistentEnv — mirroring Trainer.collectRolloutsSequential
// exactly, calling collectRecurrentTrajectoryFromEnv instead.
func (tr *RecurrentTrainer) collectRecurrentRolloutsSequential(ctx context.Context, epoch int) ([]*RecurrentRollout, error) {
	rollouts := make([]*RecurrentRollout, tr.settings.RolloutSize)
	for i := range tr.settings.RolloutSize {
		rng := reinforce.WorkerRNG(tr.settings.Seed, epoch, i)
		rollout, err := collectRecurrentTrajectoryFromEnv(ctx, tr.params, tr.persistentEnv, tr.settings.EpisodeLen, rng)
		if err != nil {
			return nil, err
		}
		rollouts[i] = rollout
	}
	return rollouts, nil
}

// collectRecurrentRolloutsPool collects settings.RolloutSize
// trajectories across tr.persistentEnvs, bounded to exactly
// len(tr.persistentEnvs) concurrent goroutines — mirroring
// Trainer.collectRolloutsPool exactly, including its own
// episode-striping and early-return-on-error behavior.
func (tr *RecurrentTrainer) collectRecurrentRolloutsPool(ctx context.Context, epoch int) ([]*RecurrentRollout, error) {
	n := len(tr.persistentEnvs)
	rollouts := make([]*RecurrentRollout, tr.settings.RolloutSize)
	errs := make([]error, tr.settings.RolloutSize)

	var wg sync.WaitGroup
	for worker := 0; worker < n; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			env := tr.persistentEnvs[worker]
			for i := worker; i < tr.settings.RolloutSize; i += n {
				rng := reinforce.WorkerRNG(tr.settings.Seed, epoch, i)
				rollout, err := collectRecurrentTrajectoryFromEnv(ctx, tr.params, env, tr.settings.EpisodeLen, rng)
				if err != nil {
					errs[i] = err
					return
				}
				rollouts[i] = rollout
			}
		}(worker)
	}
	wg.Wait()

	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	return rollouts, nil
}

// trainOnChunkMinibatch replays every chunk in minibatch through the
// shared RecurrentTrainingNetwork sequentially, accumulating gradients
// from the one combined (policy + value + entropy) graph, then applies
// an independent Adam step to each branch, each averaged over the real
// timestep count and clipped against only its own gradient norm. Because
// the policy and value branches never share a Var (see
// actorcritic.RecurrentChunkTrainingNetwork.PolicyParameters/
// ValueParameters), this one combined backward pass already leaves each
// branch's .Grad holding exactly that branch's own gradient — reading
// policyAdam/valueAdam's norms separately is exact, not an
// approximation, so no separate objective-only backward pass is needed
// to measure them (contrast Trainer.trainOnMinibatch, whose plain,
// non-recurrent network does share its trunk between both heads and so
// still needs one).
func (tr *RecurrentTrainer) trainOnChunkMinibatch(minibatch []recurrentChunk) epochDiagnostics {
	var diagnostics epochDiagnostics
	cfg := LossConfig{ClipEpsilon: tr.settings.ClipEpsilon, EntropyCoef: tr.settings.EntropyCoef, ValueCoef: tr.settings.ValueCoef}
	realSteps := recurrentRealStepCount(minibatch)
	if realSteps == 0 {
		return diagnostics
	}

	tr.network.Actor.ZeroGrad()
	for _, chunk := range minibatch {
		tr.network.SetSplitChunk(chunk.Steps, chunk.PolicyHPrevStart, chunk.PolicyCPrevStart, chunk.ValueHPrevStart, chunk.ValueCPrevStart)
		tr.network.Graph.Forward()
		for t, step := range chunk.Steps {
			if !step.Real {
				continue
			}
			diagnostics.observeStep(tr.network.Actor.PolicyOutput[t].Val.Data, tr.network.Actor.ValueOutput[t].Val.Data[0], step.OldLogProb, step.Advantage, step.Return, step.Action, cfg)
		}
		tr.network.Graph.Backward()
	}

	policyGradientNorm := tr.policyAdam.GradientNorm(realSteps)
	valueGradientNorm := tr.valueAdam.GradientNorm(realSteps)
	policyAppliedNorm := tr.policyAdam.AppliedGradientNorm(realSteps)
	valueAppliedNorm := tr.valueAdam.AppliedGradientNorm(realSteps)
	diagnostics.observeUpdate(l2Combine(policyGradientNorm, valueGradientNorm), l2Combine(policyAppliedNorm, valueAppliedNorm), policyGradientNorm, valueGradientNorm)

	tr.policyAdam.Step(realSteps)
	tr.valueAdam.Step(realSteps)
	return diagnostics
}

// l2Combine returns the L2 norm of the concatenation of two
// independently-measured gradient norms (sqrt(a² + b²)) — exact, not an
// approximation, because the policy and value branches' parameters are
// disjoint, so this equals what a single optimizer spanning both
// branches together would have reported for the whole network's norm.
func l2Combine(a, b float32) float32 {
	return float32(math.Sqrt(float64(a)*float64(a) + float64(b)*float64(b)))
}

func recurrentRealStepCount(chunks []recurrentChunk) int {
	count := 0
	for _, chunk := range chunks {
		for _, step := range chunk.Steps {
			if step.Real {
				count++
			}
		}
	}
	return count
}

// recurrentChunk is one contiguous, fixed-length (ChunkLen) window of
// one rollout's steps, self-contained with its own starting hidden/cell
// state — see chunkRollout.
type recurrentChunk struct {
	Steps                              []recurrentTrainingStep
	HPrevStart, CPrevStart             []float32
	PolicyHPrevStart, PolicyCPrevStart []float32
	ValueHPrevStart, ValueCPrevStart   []float32
}

// chunkRollout slices rollout's transitions into contiguous,
// fixed-length (chunkLen) windows, each carrying its own starting
// (HPrevs[windowStart], CPrevs[windowStart]) — the recorded state that
// was fed into the chunk's first real step at rollout time. A trailing
// short window (the episode's length isn't a multiple of chunkLen) is
// padded with Real=false entries rather than dropped or built as a
// variable-length graph — see recurrentTrainingStep's own doc comment.
// inputSize sizes a padded entry's placeholder Observation.
func chunkRollout(rollout *RecurrentRollout, advantages, returns []float32, chunkLen, inputSize int) []recurrentChunk {
	stepCount := len(rollout.Episode.Transitions)

	var chunks []recurrentChunk
	for start := 0; start < stepCount; start += chunkLen {
		end := min(start+chunkLen, stepCount)

		steps := make([]recurrentTrainingStep, chunkLen)
		for i := range chunkLen {
			idx := start + i
			if idx < end {
				transition := rollout.Episode.Transitions[idx]
				steps[i] = recurrentTrainingStep{
					Observation: transition.Observation,
					Action:      transition.Action,
					Mask:        transition.Mask,
					OldLogProb:  rollout.LogProbs[idx],
					Advantage:   advantages[idx],
					Return:      returns[idx],
					Real:        true,
				}
			} else {
				steps[i] = recurrentTrainingStep{
					Observation: rl.Observation{Values: make([]float32, inputSize)},
					Real:        false,
				}
			}
		}

		valueHStart, valueCStart := rollout.HPrevs[start], rollout.CPrevs[start]
		if len(rollout.ValueHPrevs) == len(rollout.HPrevs) {
			valueHStart, valueCStart = rollout.ValueHPrevs[start], rollout.ValueCPrevs[start]
		}
		chunks = append(chunks, recurrentChunk{
			Steps:            steps,
			HPrevStart:       rollout.HPrevs[start],
			CPrevStart:       rollout.CPrevs[start],
			PolicyHPrevStart: rollout.HPrevs[start], PolicyCPrevStart: rollout.CPrevs[start],
			ValueHPrevStart: valueHStart, ValueCPrevStart: valueCStart,
		})
	}
	return chunks
}

// normalizeRecurrentAdvantages rescales every real step's Advantage
// in place to batch-wide (across every real step of every chunk) mean
// 0, standard deviation 1 — mirroring normalizeAdvantages (trainer.go),
// excluding padded steps (Real=false) from the statistics, since their
// Advantage is always 0 regardless and including them would skew the
// mean/std toward 0 for no real reason.
func normalizeRecurrentAdvantages(chunks []recurrentChunk) {
	const advantageEpsilon = 1e-8

	var sum, sumSquares float32
	var count int
	for _, chunk := range chunks {
		for _, step := range chunk.Steps {
			if !step.Real {
				continue
			}
			sum += step.Advantage
			sumSquares += step.Advantage * step.Advantage
			count++
		}
	}
	if count == 0 {
		return
	}

	mean := sum / float32(count)
	variance := sumSquares/float32(count) - mean*mean
	std := float32(math.Sqrt(float64(max(variance, 0))))

	for ci := range chunks {
		for si := range chunks[ci].Steps {
			if !chunks[ci].Steps[si].Real {
				continue
			}
			chunks[ci].Steps[si].Advantage = (chunks[ci].Steps[si].Advantage - mean) / (std + advantageEpsilon)
		}
	}
}
