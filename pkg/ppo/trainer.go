package ppo

import (
	"fmt"
	"math"
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

	params  *actorcritic.Params
	network *TrainingNetwork
	adam    *actorcritic.Adam
}

// New constructs a Trainer from settings and envFactory, validating
// settings and sizing the actor-critic network from envFactory's
// environment, mirroring pkg/reinforce.New's shape (including the
// initialParams resume-from-checkpoint convention).
func New(settings config.Settings, envFactory reinforce.EnvFactory, initialParams *actorcritic.Params) (*Trainer, error) {
	if err := settings.Validate(); err != nil {
		return nil, err
	}

	initRNG := reinforce.WorkerRNG(settings.Seed, 0, initWorkerIndex)

	env, err := envFactory(initRNG)
	if err != nil {
		return nil, fmt.Errorf("ppo: constructing environment: %w", err)
	}

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
		settings:   settings,
		envFactory: envFactory,
		params:     params,
		network:    network,
		adam:       actorcritic.NewAdam(network.Actor.Parameters(), settings.LearningRate),
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
// EpochStats.Epoch.
func (tr *Trainer) RunEpoch(epoch int) (EpochStats, error) {
	rollouts, err := tr.collectRollouts(epoch)
	if err != nil {
		return EpochStats{}, err
	}

	scored := make([]scoredRollout, len(rollouts))
	for i, rollout := range rollouts {
		advantages, returns := computeGAE(rollout, tr.settings.Gamma, tr.settings.GAELambda)
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
			tr.trainOnMinibatch(steps[start:end])
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

// collectRollouts collects settings.RolloutSize trajectories in
// parallel, bounded to settings.Workers concurrent goroutines,
// mirroring pkg/reinforce.Trainer.collectRollouts exactly (same
// parallel-goroutine, per-worker-RNG structure via WorkerRNG), but
// calling this package's own collectTrajectory to also capture each
// step's log-probability and value estimate.
func (tr *Trainer) collectRollouts(epoch int) ([]*Rollout, error) {
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
			rollout, err := collectTrajectory(tr.params, tr.envFactory, tr.settings.EpisodeLen, rng)
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

// trainOnMinibatch replays every step in minibatch through the shared
// TrainingNetwork sequentially, accumulating gradients (see
// autograd.Graph.Backward), then applies one Adam step averaged over
// len(minibatch).
func (tr *Trainer) trainOnMinibatch(minibatch []trainingStep) {
	tr.network.Actor.ZeroGrad()

	for _, step := range minibatch {
		copy(tr.network.Actor.Input.Val.Data, step.Observation.Values)
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
