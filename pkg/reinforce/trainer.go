package reinforce

import (
	"context"
	"fmt"
	"math/rand/v2"
	"sync"

	"github.com/reallyoldfogie/cRL-go/pkg/config"
	"github.com/reallyoldfogie/cRL-go/pkg/policy"
	"github.com/reallyoldfogie/cRL-go/pkg/rl"
)

// EpochStats summarizes the result of one training epoch.
type EpochStats struct {
	Epoch         int
	AverageReturn float32
	SampleCount   int
	ReturnStd     float32
	// UpdateCount is the number of gradient-update steps RunEpoch applied
	// (always 1 for REINFORCE's one-SGD-step-per-epoch design), so a
	// caller accumulating it across epochs can populate a
	// checkpoint.Metadata.TotalUpdates.
	UpdateCount int
}

// Trainer runs the REINFORCE training loop described in
// docs/03-policy-gradients-and-reinforce.md: each epoch collects a batch
// of episodes from the current policy against envFactory (in parallel,
// across settings.Workers goroutines), computes a batch-wide return
// baseline, replays every step to accumulate policy-gradient gradients,
// and applies one SGD step. Trainer itself never references a specific
// rl.Environment implementation, so any environment (snakeenv,
// gridworldenv, or otherwise) trains through the exact same code path.
//
// Rollout collection is parallel; gradient accumulation and the SGD
// update are sequential, operating on a single shared
// policy.TrainingNetwork. See docs/05-porting-notes.md for why the
// concurrency boundary is drawn there rather than around the backward
// pass.
type Trainer struct {
	settings   config.Settings
	envFactory EnvFactory

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

	params  *policy.Params
	network *policy.TrainingNetwork
}

// PersistentEnvFactory builds one rl.Environment meant to be reused
// across many episodes (Reset called between them, via
// Trainer.collectRolloutsSequential) rather than constructed fresh per
// rollout like EnvFactory, for environments too expensive or stateful
// to rebuild per episode (e.g. a live game session). See
// NewWithPersistentEnv and docs/plans/08-context-and-long-lived-environments.md.
type PersistentEnvFactory func(rng *rand.Rand) (rl.Environment, error)

// New constructs a Trainer from settings and envFactory, validating
// settings and sizing the policy from envFactory's environment:
// params.InputSize()/OutputSize() come from
// env.ObservationSize()/env.ActionSpace(), not from any
// environment-specific constant.
//
// If initialParams is nil, a fresh policy is initialized (Xavier/Glorot,
// seeded from settings.Seed), matching a from-scratch training run. If
// initialParams is non-nil (e.g. loaded via policy.LoadFile from a
// previous session's checkpoint), it's used as-is after validating that
// its layer sizes match envFactory's environment and settings.HiddenSize,
// so training resumes from those weights instead of starting over.
//
// The resulting Trainer collects rollouts in parallel across
// settings.Workers goroutines, rebuilding an environment per episode;
// see NewWithPersistentEnv for a long-lived-environment alternative.
func New(settings config.Settings, envFactory EnvFactory, initialParams *policy.Params) (*Trainer, error) {
	if err := settings.Validate(); err != nil {
		return nil, err
	}

	initRNG := rand.New(rand.NewPCG(settings.Seed, settings.Seed))

	env, err := envFactory(initRNG)
	if err != nil {
		return nil, fmt.Errorf("reinforce: constructing environment: %w", err)
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
func NewWithPersistentEnv(settings config.Settings, persistentFactory PersistentEnvFactory, initialParams *policy.Params) (*Trainer, error) {
	if err := settings.Validate(); err != nil {
		return nil, err
	}

	initRNG := rand.New(rand.NewPCG(settings.Seed, settings.Seed))

	env, err := persistentFactory(initRNG)
	if err != nil {
		return nil, fmt.Errorf("reinforce: constructing persistent environment: %w", err)
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
func NewWithPersistentEnvPool(settings config.Settings, envs []rl.Environment, initialParams *policy.Params) (*Trainer, error) {
	if len(envs) == 0 {
		return nil, fmt.Errorf("reinforce: envs must contain at least one environment")
	}
	if err := settings.Validate(); err != nil {
		return nil, err
	}

	initRNG := rand.New(rand.NewPCG(settings.Seed, settings.Seed))

	trainer, err := newTrainer(settings, envs[0], initRNG, initialParams)
	if err != nil {
		return nil, err
	}
	trainer.persistentEnvs = envs
	return trainer, nil
}

// newTrainer builds the params/network shared by New and
// NewWithPersistentEnv from an already-constructed env, used only here
// to read ObservationSize()/ActionSpace() (and, for a non-nil
// initialParams, to validate against it) — it is not itself stored on
// the returned Trainer; callers set either envFactory or persistentEnv
// afterward depending on which constructor they came from.
func newTrainer(settings config.Settings, env rl.Environment, initRNG *rand.Rand, initialParams *policy.Params) (*Trainer, error) {
	params := initialParams
	if params == nil {
		params = policy.NewParams(initRNG, env.ObservationSize(), settings.HiddenSize, env.ActionSpace())
	} else if err := validateParamsShape(params, env, settings); err != nil {
		return nil, err
	}

	network, err := policy.NewTrainingNetwork(params, settings.EntropyCoef)
	if err != nil {
		return nil, fmt.Errorf("reinforce: building training network: %w", err)
	}

	return &Trainer{
		settings: settings,
		params:   params,
		network:  network,
	}, nil
}

// validateParamsShape reports an error if params' layer sizes don't
// match what env and settings expect, so a checkpoint saved for one
// environment/hyperparameter combination can't silently be resumed
// against an incompatible one.
func validateParamsShape(params *policy.Params, env rl.Environment, settings config.Settings) error {
	if params.InputSize() != env.ObservationSize() {
		return fmt.Errorf(
			"reinforce: initial params input size %d does not match environment observation size %d",
			params.InputSize(), env.ObservationSize(),
		)
	}
	if params.HiddenSize() != settings.HiddenSize {
		return fmt.Errorf(
			"reinforce: initial params hidden size %d does not match settings hidden size %d",
			params.HiddenSize(), settings.HiddenSize,
		)
	}
	if params.OutputSize() != env.ActionSpace() {
		return fmt.Errorf(
			"reinforce: initial params output size %d does not match environment action space %d",
			params.OutputSize(), env.ActionSpace(),
		)
	}
	return nil
}

// Params returns the Trainer's current policy parameters, e.g. to save a
// checkpoint (see policy.Params.Save/SaveFile) after training so a later
// session can resume from it via New's initialParams.
func (tr *Trainer) Params() *policy.Params {
	return tr.params
}

// GradientNorm returns the L2 norm of every parameter's gradient as
// accumulated by the most recently completed RunEpoch call (see
// policy.TrainingNetwork.GradientNorm), for diagnosing a run whose
// observable behavior (EpochStats.AverageReturn, or a checksum of
// Params' weights) has gone flat: this distinguishes "the gradient
// itself has vanished" from "the environment/behavior has converged but
// training is still nominally alive underneath," which look identical
// from EpochStats alone. Calling this before the first RunEpoch reports
// 0 (no gradient has been accumulated yet).
func (tr *Trainer) GradientNorm() float32 {
	return tr.network.GradientNorm()
}

// RunEpoch runs one full training epoch (rollout collection, return/
// advantage computation, gradient accumulation, and one SGD step) and
// returns summary statistics. epoch is used only to derive this epoch's
// worker RNG seeds (see WorkerRNG) and to populate EpochStats.Epoch.
// ctx is threaded through to every rl.Environment.Reset/Step call this
// epoch makes, so a caller can cancel or time out rollout collection
// against an environment that can block.
func (tr *Trainer) RunEpoch(ctx context.Context, epoch int) (EpochStats, error) {
	episodes, err := tr.collectRollouts(ctx, epoch)
	if err != nil {
		return EpochStats{}, err
	}

	scored := make([]scoredEpisode, len(episodes))
	for i, episode := range episodes {
		scored[i] = scoredEpisode{
			Episode: episode,
			Returns: computeReturns(episode, tr.settings.Gamma),
		}
	}

	mean, std, sampleCount := returnStatistics(scored)

	tr.trainOnRollouts(scored, mean, std)

	// Lock/Unlock guard only the actual weight mutation (Val), not the
	// preceding gradient accumulation (which only reads Val and writes
	// Grad): that's the narrowest section a concurrent Params.Snapshot
	// (e.g. a live inference Actor sharing tr.params) needs excluded.
	// See docs/plans/09-concurrency-safe-live-inference.md.
	tr.params.Lock()
	tr.network.ApplyGradientStep(tr.settings.LearningRate, sampleCount)
	tr.params.Unlock()

	var returnSum float32
	for _, s := range scored {
		if len(s.Returns) > 0 {
			returnSum += s.Returns[0]
		}
	}

	return EpochStats{
		Epoch:         epoch,
		AverageReturn: returnSum / float32(len(scored)),
		SampleCount:   sampleCount,
		ReturnStd:     std,
		UpdateCount:   1,
	}, nil
}

// collectRollouts collects settings.RolloutSize episodes for this
// epoch, dispatching to collectRolloutsPool if tr was built via
// NewWithPersistentEnvPool, collectRolloutsSequential if tr was built
// via NewWithPersistentEnv, or collectRolloutsParallel (the original,
// per-episode-construction path) otherwise.
func (tr *Trainer) collectRollouts(ctx context.Context, epoch int) ([]*rl.Episode, error) {
	if len(tr.persistentEnvs) > 0 {
		return tr.collectRolloutsPool(ctx, epoch)
	}
	if tr.persistentEnv != nil {
		return tr.collectRolloutsSequential(ctx, epoch)
	}
	return tr.collectRolloutsParallel(ctx, epoch)
}

// collectRolloutsParallel collects settings.RolloutSize episodes in
// parallel, bounded to settings.Workers concurrent goroutines. Each
// goroutine builds its own environment (via tr.envFactory) and
// policy.InferenceNetwork (see collectTrajectory) against tr.params,
// which is only read (never mutated) until every goroutine here has
// finished.
func (tr *Trainer) collectRolloutsParallel(ctx context.Context, epoch int) ([]*rl.Episode, error) {
	episodes := make([]*rl.Episode, tr.settings.RolloutSize)
	errs := make([]error, tr.settings.RolloutSize)

	var wg sync.WaitGroup
	semaphore := make(chan struct{}, tr.settings.Workers)

	for i := range tr.settings.RolloutSize {
		wg.Add(1)
		semaphore <- struct{}{}

		go func(index int) {
			defer wg.Done()
			defer func() { <-semaphore }()

			rng := WorkerRNG(tr.settings.Seed, epoch, index)
			episode, err := collectTrajectory(ctx, tr.params, tr.envFactory, tr.settings.EpisodeLen, rng)
			episodes[index] = episode
			errs[index] = err
		}(i)
	}
	wg.Wait()

	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	return episodes, nil
}

// collectRolloutsSequential collects settings.RolloutSize episodes one
// at a time, on the calling goroutine, against tr.persistentEnv — the
// counterpart to collectRolloutsParallel for a long-lived environment
// that must be Reset between episodes rather than rebuilt per episode.
// Every episode still gets its own deterministic per-(epoch, worker)
// rng via WorkerRNG, exactly as the parallel path does, just consumed
// in order rather than from concurrent goroutines; settings.Workers is
// not consulted here.
func (tr *Trainer) collectRolloutsSequential(ctx context.Context, epoch int) ([]*rl.Episode, error) {
	episodes := make([]*rl.Episode, tr.settings.RolloutSize)
	for i := range tr.settings.RolloutSize {
		rng := WorkerRNG(tr.settings.Seed, epoch, i)
		episode, err := collectTrajectoryFromEnv(ctx, tr.params, tr.persistentEnv, tr.settings.EpisodeLen, rng)
		if err != nil {
			return nil, err
		}
		episodes[i] = episode
	}
	return episodes, nil
}

// collectRolloutsPool collects settings.RolloutSize episodes across
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
func (tr *Trainer) collectRolloutsPool(ctx context.Context, epoch int) ([]*rl.Episode, error) {
	n := len(tr.persistentEnvs)
	episodes := make([]*rl.Episode, tr.settings.RolloutSize)
	errs := make([]error, tr.settings.RolloutSize)

	var wg sync.WaitGroup
	for worker := 0; worker < n; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			env := tr.persistentEnvs[worker]
			for i := worker; i < tr.settings.RolloutSize; i += n {
				rng := WorkerRNG(tr.settings.Seed, epoch, i)
				episode, err := collectTrajectoryFromEnv(ctx, tr.params, env, tr.settings.EpisodeLen, rng)
				if err != nil {
					errs[i] = err
					return
				}
				episodes[i] = episode
			}
		}(worker)
	}
	wg.Wait()

	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	return episodes, nil
}

// trainOnRollouts replays every transition of every scored episode
// through the shared TrainingNetwork sequentially, accumulating
// gradients (Backward adds into each parameter's Grad rather than
// overwriting it — see policy.TrainingNetwork and
// autograd.Graph.Backward). Advantages are the batch-normalized returns:
// (return - mean) / (std + epsilon), which acts as a variance-reduction
// baseline (see docs/03-policy-gradients-and-reinforce.md).
func (tr *Trainer) trainOnRollouts(episodes []scoredEpisode, mean, std float32) {
	const advantageEpsilon = 1e-8

	tr.network.ZeroGrad()

	for _, s := range episodes {
		for t, transition := range s.Episode.Transitions {
			copy(tr.network.Input.Val.Data, transition.Observation.Values)
			tr.network.SetActionMask(transition.Mask)

			tr.network.Advantage.Val.Clear()
			advantage := (s.Returns[t] - mean) / (std + advantageEpsilon)
			tr.network.Advantage.Val.Data[transition.Action] = advantage

			tr.network.Graph.Forward()
			tr.network.Graph.Backward()
		}
	}
}
