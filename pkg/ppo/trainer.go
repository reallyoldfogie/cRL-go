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

// EpochStats summarizes the result of one PPO training epoch.
type EpochStats struct {
	Epoch         int
	AverageReturn float32
	SampleCount   int
	// UpdateCount is the number of Adam.Step calls RunEpoch applied
	// (PPOEpochs passes times however many minibatches each pass split
	// the collected steps into), so a caller accumulating it across
	// epochs can populate a checkpoint.Metadata.TotalUpdates.
	UpdateCount int
}

// Trainer runs the PPO training loop described in
// docs/plans/03-gae-and-ppo-objective.md and
// docs/plans/04-adam-optimizer-and-minibatch-trainer.md: each epoch
// collects a batch of rollouts from the current policy against
// envFactory (in parallel, mirroring pkg/reinforce.Trainer's
// collectRollouts), computes GAE(lambda) advantages and value-target
// returns per trajectory, flattens every trajectory's steps into one
// pool, and runs settings.PPOEpochs passes of shuffled-minibatch Adam
// updates over that pool.
//
// Trainer coexists with, and is entirely independent of,
// pkg/reinforce.Trainer: it trains a pkg/actorcritic network instead of
// a pkg/policy one, reusing only genuinely environment/algorithm-generic
// pieces of pkg/reinforce (EnvFactory, SampleAction, WorkerRNG) rather
// than any of its REINFORCE-specific training logic.
type Trainer struct {
	settings   config.Settings
	envFactory reinforce.EnvFactory

	// persistentEnv, when non-nil, is a long-lived environment built
	// once (via NewWithPersistentEnv) instead of per-episode via
	// envFactory; its presence is what collectRollouts uses to choose
	// collectRolloutsSequential over collectRolloutsParallel. Exactly
	// one of envFactory/persistentEnv/persistentEnvs is set, never more
	// than one.
	persistentEnv rl.Environment

	// persistentEnvs, when non-empty, is a pool of N long-lived
	// environments (built once, via NewWithPersistentEnvPool), each
	// reused across many episodes via Reset and driven concurrently with
	// each other — but never touched by more than one goroutine at a
	// time (see collectRolloutsPool). Its presence is what
	// collectRollouts uses to choose collectRolloutsPool over the other
	// two collection strategies. Exactly one of
	// envFactory/persistentEnv/persistentEnvs is set, never more than
	// one.
	persistentEnvs []rl.Environment

	params  *actorcritic.Params
	network *TrainingNetwork
	adam    *actorcritic.Adam
}

// New constructs a Trainer from settings and envFactory, validating
// settings and sizing the actor-critic network from envFactory's
// environment, mirroring pkg/reinforce.New's shape (including the
// initialParams resume-from-checkpoint convention). The resulting
// Trainer collects rollouts in parallel across settings.Workers
// goroutines, rebuilding an environment per episode; see
// NewWithPersistentEnv for a long-lived-environment alternative.
func New(settings config.Settings, envFactory reinforce.EnvFactory, initialParams *actorcritic.Params) (*Trainer, error) {
	if err := settings.Validate(); err != nil {
		return nil, err
	}

	initRNG := reinforce.WorkerRNG(settings.Seed, 0, initWorkerIndex)

	env, err := envFactory(initRNG)
	if err != nil {
		return nil, fmt.Errorf("ppo: constructing environment: %w", err)
	}

	trainer, err := newTrainer(settings, env, initRNG, initialParams)
	if err != nil {
		return nil, err
	}
	trainer.envFactory = envFactory
	return trainer, nil
}

// NewWithPersistentEnv constructs a Trainer exactly like New, except it
// builds one environment, once, via persistentFactory, and reuses it
// across every episode of every epoch (collectRolloutsSequential resets
// it between episodes instead of rebuilding it), rather than rebuilding
// a fresh one per episode. Rollout collection through this Trainer runs
// sequentially on the calling goroutine rather than across
// settings.Workers goroutines, since there is only one environment
// instance to drive; settings.Workers is simply unused in that case.
func NewWithPersistentEnv(settings config.Settings, persistentFactory reinforce.PersistentEnvFactory, initialParams *actorcritic.Params) (*Trainer, error) {
	if err := settings.Validate(); err != nil {
		return nil, err
	}

	initRNG := reinforce.WorkerRNG(settings.Seed, 0, initWorkerIndex)

	env, err := persistentFactory(initRNG)
	if err != nil {
		return nil, fmt.Errorf("ppo: constructing persistent environment: %w", err)
	}

	trainer, err := newTrainer(settings, env, initRNG, initialParams)
	if err != nil {
		return nil, err
	}
	trainer.persistentEnv = env
	return trainer, nil
}

// NewWithPersistentEnvPool constructs a Trainer exactly like
// NewWithPersistentEnv, except it drives N already-constructed,
// long-lived environments (envs) concurrently with each other instead
// of exactly one sequentially. Each envs[i] is Reset and stepped by
// exactly one goroutine for its entire lifetime — see
// collectRolloutsPool — so envs must not be shared with any other
// caller. Concurrency is bounded to len(envs), not settings.Workers:
// settings.Workers sizes a generic CPU-bound goroutine pool appropriate
// for cheap, freely-constructible environments (see
// collectRolloutsParallel), which has no relationship to how many real,
// expensive, already-connected sessions envs actually contains — using
// settings.Workers here would either under-use available envs
// (Workers < len(envs)) or require sharing one envs[i] across multiple
// concurrent goroutines (Workers > len(envs)), which rl.Environment's
// own doc comment does not promise is safe. settings.Workers is simply
// unused on this path, exactly as it already is for
// NewWithPersistentEnv.
//
// Callers own envs' lifecycle (construction and eventual teardown, if
// applicable) before and after this Trainer's use of them; this
// constructor and the Trainer it returns never construct or close an
// environment themselves.
func NewWithPersistentEnvPool(settings config.Settings, envs []rl.Environment, initialParams *actorcritic.Params) (*Trainer, error) {
	if len(envs) == 0 {
		return nil, fmt.Errorf("ppo: envs must contain at least one environment")
	}
	if err := settings.Validate(); err != nil {
		return nil, err
	}

	initRNG := reinforce.WorkerRNG(settings.Seed, 0, initWorkerIndex)

	trainer, err := newTrainer(settings, envs[0], initRNG, initialParams)
	if err != nil {
		return nil, err
	}
	trainer.persistentEnvs = envs
	return trainer, nil
}

// newTrainer builds the params/network/optimizer shared by New and
// NewWithPersistentEnv from an already-constructed env, used only here
// to read ObservationSize()/ActionSpace() (and, for a non-nil
// initialParams, to validate against it) — it is not itself stored on
// the returned Trainer; callers set either envFactory or persistentEnv
// afterward depending on which constructor they came from.
func newTrainer(settings config.Settings, env rl.Environment, initRNG *rand.Rand, initialParams *actorcritic.Params) (*Trainer, error) {
	params := initialParams
	if params == nil {
		params = actorcritic.NewParams(initRNG, env.ObservationSize(), settings.HiddenSize, env.ActionSpace())
	} else if err := validateParamsShape(params, env, settings); err != nil {
		return nil, err
	}

	network, err := NewTrainingNetwork(params, LossConfig{
		ClipEpsilon: settings.ClipEpsilon,
		EntropyCoef: settings.EntropyCoef,
		ValueCoef:   settings.ValueCoef,
	})
	if err != nil {
		return nil, fmt.Errorf("ppo: building training network: %w", err)
	}

	return &Trainer{
		settings: settings,
		params:   params,
		network:  network,
		adam:     actorcritic.NewAdamWithGradClip(network.Actor.Parameters(), settings.LearningRate, settings.MaxGradNorm),
	}, nil
}

// initWorkerIndex is the worker index used to derive Trainer's one-time
// initialization RNG (see New), distinguishing it from any of
// collectRollouts' actual worker indices (which start at 0), and from
// shuffleWorkerIndex (see RunEpoch).
const initWorkerIndex = -1

// shuffleWorkerIndex is the worker index used to derive each epoch's
// minibatch-shuffle RNG (see RunEpoch), distinct from both
// initWorkerIndex and every real rollout-collection worker index.
const shuffleWorkerIndex = -2

// validateParamsShape reports an error if params' layer sizes don't
// match what env and settings expect, mirroring
// pkg/reinforce.validateParamsShape for pkg/actorcritic.Params.
func validateParamsShape(params *actorcritic.Params, env rl.Environment, settings config.Settings) error {
	if params.InputSize() != env.ObservationSize() {
		return fmt.Errorf(
			"ppo: initial params input size %d does not match environment observation size %d",
			params.InputSize(), env.ObservationSize(),
		)
	}
	if params.HiddenSize() != settings.HiddenSize {
		return fmt.Errorf(
			"ppo: initial params hidden size %d does not match settings hidden size %d",
			params.HiddenSize(), settings.HiddenSize,
		)
	}
	if params.OutputSize() != env.ActionSpace() {
		return fmt.Errorf(
			"ppo: initial params output size %d does not match environment action space %d",
			params.OutputSize(), env.ActionSpace(),
		)
	}
	return nil
}

// Params returns the Trainer's current actor-critic parameters, e.g. to
// save a checkpoint (see actorcritic.Params.Save/SaveFile) after
// training so a later session can resume from it via New's
// initialParams.
func (tr *Trainer) Params() *actorcritic.Params {
	return tr.params
}

// RunEpoch runs one full PPO training epoch (rollout collection, GAE,
// and settings.PPOEpochs shuffled-minibatch Adam updates) and returns
// summary statistics. epoch is used only to derive this epoch's
// deterministic RNG streams (see WorkerRNG) and to populate
// EpochStats.Epoch. ctx is threaded through to every
// rl.Environment.Reset/Step call this epoch makes, so a caller can
// cancel or time out rollout collection against an environment that can
// block.
func (tr *Trainer) RunEpoch(ctx context.Context, epoch int) (EpochStats, error) {
	rollouts, err := tr.collectRollouts(ctx, epoch)
	if err != nil {
		return EpochStats{}, err
	}

	scored := make([]scoredRollout, len(rollouts))
	for i, rollout := range rollouts {
		advantages, returns := ComputeGAE(rollout, tr.settings.Gamma, tr.settings.GAELambda)
		scored[i] = scoredRollout{Rollout: rollout, Advantages: advantages, Returns: returns}
	}

	steps := flattenSteps(scored)
	normalizeAdvantages(steps)

	shuffleRNG := reinforce.WorkerRNG(tr.settings.Seed, epoch, shuffleWorkerIndex)
	updateCount := 0
	for range tr.settings.PPOEpochs {
		shuffleRNG.Shuffle(len(steps), func(i, j int) {
			steps[i], steps[j] = steps[j], steps[i]
		})

		for start := 0; start < len(steps); start += tr.settings.MinibatchSize {
			end := min(start+tr.settings.MinibatchSize, len(steps))

			// Lock/Unlock guard the whole per-minibatch training unit
			// (gradient accumulation and the Adam step that mutates Val),
			// since — unlike pkg/reinforce, which accumulates once over
			// the whole batch before a single ApplyGradientStep — PPO
			// repeats this accumulate-then-update cycle once per
			// minibatch. A concurrent Params.Snapshot (e.g. a live
			// inference Actor sharing tr.params) must never observe a
			// partially-applied minibatch update. See
			// docs/plans/09-concurrency-safe-live-inference.md.
			tr.params.Lock()
			tr.trainOnMinibatch(steps[start:end])
			tr.params.Unlock()
			updateCount++
		}
	}

	var returnSum float32
	for _, s := range scored {
		if len(s.Returns) > 0 {
			returnSum += s.Returns[0]
		}
	}

	return EpochStats{
		Epoch:         epoch,
		AverageReturn: returnSum / float32(len(scored)),
		SampleCount:   len(steps),
		UpdateCount:   updateCount,
	}, nil
}

// collectRollouts collects settings.RolloutSize trajectories for this
// epoch, dispatching to collectRolloutsPool if tr was built via
// NewWithPersistentEnvPool, collectRolloutsSequential if tr was built
// via NewWithPersistentEnv, or collectRolloutsParallel (the original,
// per-episode-construction path) otherwise.
func (tr *Trainer) collectRollouts(ctx context.Context, epoch int) ([]*Rollout, error) {
	if len(tr.persistentEnvs) > 0 {
		return tr.collectRolloutsPool(ctx, epoch)
	}
	if tr.persistentEnv != nil {
		return tr.collectRolloutsSequential(ctx, epoch)
	}
	return tr.collectRolloutsParallel(ctx, epoch)
}

// collectRolloutsParallel collects settings.RolloutSize trajectories in
// parallel, bounded to settings.Workers concurrent goroutines,
// mirroring pkg/reinforce.Trainer.collectRolloutsParallel exactly (same
// parallel-goroutine, per-worker-RNG structure via WorkerRNG), but
// calling this package's own collectTrajectory to also capture each
// step's log-probability and value estimate.
func (tr *Trainer) collectRolloutsParallel(ctx context.Context, epoch int) ([]*Rollout, error) {
	rollouts := make([]*Rollout, tr.settings.RolloutSize)
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
			rollout, err := collectTrajectory(ctx, tr.params, tr.envFactory, tr.settings.EpisodeLen, rng)
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

// collectRolloutsSequential collects settings.RolloutSize trajectories
// one at a time, on the calling goroutine, against tr.persistentEnv —
// the counterpart to collectRolloutsParallel for a long-lived
// environment that must be Reset between episodes rather than rebuilt
// per episode. Every episode still gets its own deterministic
// per-(epoch, worker) rng via WorkerRNG, exactly as the parallel path
// does, just consumed in order rather than from concurrent goroutines;
// settings.Workers is not consulted here.
func (tr *Trainer) collectRolloutsSequential(ctx context.Context, epoch int) ([]*Rollout, error) {
	rollouts := make([]*Rollout, tr.settings.RolloutSize)
	for i := range tr.settings.RolloutSize {
		rng := reinforce.WorkerRNG(tr.settings.Seed, epoch, i)
		rollout, err := collectTrajectoryFromEnv(ctx, tr.params, tr.persistentEnv, tr.settings.EpisodeLen, rng)
		if err != nil {
			return nil, err
		}
		rollouts[i] = rollout
	}
	return rollouts, nil
}

// collectRolloutsPool collects settings.RolloutSize trajectories across
// tr.persistentEnvs, bounded to exactly len(tr.persistentEnvs)
// concurrent goroutines — one per pooled environment, for the
// goroutine's entire lifetime, never shared — rather than
// settings.Workers (see NewWithPersistentEnvPool's own doc comment for
// why). Episode indices are striped across workers (worker, worker+n,
// worker+2n, ...) so every episode still gets its own deterministic
// WorkerRNG(settings.Seed, epoch, i) exactly as
// collectRolloutsParallel/collectRolloutsSequential already do — i is
// the global episode index, not a per-worker-local one, so a fixed seed
// continues to reproduce the same set of episode streams regardless of
// how many envs the pool contains or how work happens to interleave
// across them. A worker stops (recording its own error) at its own
// first error rather than continuing to its next striped episode
// against a session that just failed, mirroring
// collectRolloutsSequential's own early-return-on-error behavior.
func (tr *Trainer) collectRolloutsPool(ctx context.Context, epoch int) ([]*Rollout, error) {
	n := len(tr.persistentEnvs)
	rollouts := make([]*Rollout, tr.settings.RolloutSize)
	errs := make([]error, tr.settings.RolloutSize)

	var wg sync.WaitGroup
	for worker := 0; worker < n; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			env := tr.persistentEnvs[worker]
			for i := worker; i < tr.settings.RolloutSize; i += n {
				rng := reinforce.WorkerRNG(tr.settings.Seed, epoch, i)
				rollout, err := collectTrajectoryFromEnv(ctx, tr.params, env, tr.settings.EpisodeLen, rng)
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

// trainOnMinibatch replays every step in minibatch through the shared
// TrainingNetwork sequentially, accumulating gradients (see
// autograd.Graph.Backward), then applies one Adam step averaged over
// len(minibatch).
func (tr *Trainer) trainOnMinibatch(minibatch []trainingStep) {
	tr.network.Actor.ZeroGrad()

	for _, step := range minibatch {
		copy(tr.network.Actor.Input.Val.Data, step.Observation.Values)
		tr.network.Actor.SetActionMask(step.Mask)
		tr.network.SetStep(step.Action, step.OldLogProb, step.Advantage, step.Return)

		tr.network.Graph.Forward()
		tr.network.Graph.Backward()
	}

	tr.adam.Step(len(minibatch))
}

// scoredRollout pairs a collected Rollout with its GAE advantages and
// value-target returns, mirroring pkg/reinforce's scoredEpisode.
type scoredRollout struct {
	Rollout    *Rollout
	Advantages []float32
	Returns    []float32
}

// trainingStep is one flattened (observation, action, ...) tuple ready
// for a minibatch Adam update, gathering everything trainOnMinibatch
// needs from a single step of a scoredRollout.
type trainingStep struct {
	Observation rl.Observation
	Action      rl.Action
	Mask        []bool
	OldLogProb  float32
	Advantage   float32
	Return      float32
}

// flattenSteps gathers every step of every scoredRollout into one pool,
// so PPO's minibatch shuffling can draw from the whole collected batch
// rather than being confined to shuffling within each trajectory.
func flattenSteps(scored []scoredRollout) []trainingStep {
	var steps []trainingStep
	for _, s := range scored {
		for t, transition := range s.Rollout.Episode.Transitions {
			steps = append(steps, trainingStep{
				Observation: transition.Observation,
				Action:      transition.Action,
				Mask:        transition.Mask,
				OldLogProb:  s.Rollout.LogProbs[t],
				Advantage:   s.Advantages[t],
				Return:      s.Returns[t],
			})
		}
	}
	return steps
}

// normalizeAdvantages rescales every step's Advantage in place to
// batch-wide mean 0, standard deviation 1: (advantage - mean) /
// (std + epsilon). This is standard PPO practice (not simply doc03's
// GAE formula) because raw GAE magnitudes can vary widely batch to
// batch, which would otherwise make a fixed Adam learning rate behave
// inconsistently across epochs; it mirrors the same
// batch-statistics-as-variance-reduction idea pkg/reinforce's
// returnStatistics already applies to raw returns.
func normalizeAdvantages(steps []trainingStep) {
	if len(steps) == 0 {
		return
	}

	const advantageEpsilon = 1e-8

	var sum, sumSquares float32
	for _, step := range steps {
		sum += step.Advantage
		sumSquares += step.Advantage * step.Advantage
	}

	mean := sum / float32(len(steps))
	variance := sumSquares/float32(len(steps)) - mean*mean
	std := float32(math.Sqrt(float64(max(variance, 0))))

	for i := range steps {
		steps[i].Advantage = (steps[i].Advantage - mean) / (std + advantageEpsilon)
	}
}
