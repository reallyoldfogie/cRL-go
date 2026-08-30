package reinforce

import (
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

	params  *policy.Params
	network *policy.TrainingNetwork
}

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
func New(settings config.Settings, envFactory EnvFactory, initialParams *policy.Params) (*Trainer, error) {
	if err := settings.Validate(); err != nil {
		return nil, err
	}

	initRNG := rand.New(rand.NewPCG(settings.Seed, settings.Seed))

	env, err := envFactory(initRNG)
	if err != nil {
		return nil, fmt.Errorf("reinforce: constructing environment: %w", err)
	}

	params := initialParams
	if params == nil {
		params = policy.NewParams(initRNG, env.ObservationSize(), settings.HiddenSize, env.ActionSpace())
	} else if err := validateParamsShape(params, env, settings); err != nil {
		return nil, err
	}

	network, err := policy.NewTrainingNetwork(params)
	if err != nil {
		return nil, fmt.Errorf("reinforce: building training network: %w", err)
	}

	return &Trainer{
		settings:   settings,
		envFactory: envFactory,
		params:     params,
		network:    network,
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

// RunEpoch runs one full training epoch (rollout collection, return/
// advantage computation, gradient accumulation, and one SGD step) and
// returns summary statistics. epoch is used only to derive this epoch's
// worker RNG seeds (see WorkerRNG) and to populate EpochStats.Epoch.
func (tr *Trainer) RunEpoch(epoch int) (EpochStats, error) {
	episodes, err := tr.collectRollouts(epoch)
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
	tr.network.ApplyGradientStep(tr.settings.LearningRate, sampleCount)

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

// collectRollouts collects settings.RolloutSize episodes in parallel,
// bounded to settings.Workers concurrent goroutines. Each goroutine
// builds its own environment (via tr.envFactory) and
// policy.InferenceNetwork (see collectTrajectory) against tr.params,
// which is only read (never mutated) until every goroutine here has
// finished.
func (tr *Trainer) collectRollouts(epoch int) ([]*rl.Episode, error) {
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
			episode, err := collectTrajectory(tr.params, tr.envFactory, tr.settings.EpisodeLen, rng)
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

			tr.network.Advantage.Val.Clear()
			advantage := (s.Returns[t] - mean) / (std + advantageEpsilon)
			tr.network.Advantage.Val.Data[transition.Action] = advantage

			tr.network.Graph.Forward()
			tr.network.Graph.Backward()
		}
	}
}
