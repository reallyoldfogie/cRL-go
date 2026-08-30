package hierarchical

import (
	"context"
	"fmt"
	"math/rand/v2"
	"sync"

	"github.com/reallyoldfogie/cRL-go/pkg/actorcritic"
	"github.com/reallyoldfogie/cRL-go/pkg/config"
	"github.com/reallyoldfogie/cRL-go/pkg/ppo"
	"github.com/reallyoldfogie/cRL-go/pkg/reinforce"
)

// EpochStats summarizes one hierarchical training epoch. SubUpdateCounts
// only has an entry for a subgoal that was actually activated somewhere
// in the batch this epoch; a subgoal that was never chosen has no
// entry, rather than a spurious 0, since "never trained this epoch" and
// "trained but happened to need 0 minibatches" are different things
// worth being able to tell apart.
type EpochStats struct {
	Epoch           int
	AverageReturn   float32
	SampleCount     int
	MetaUpdateCount int
	SubUpdateCounts map[Subgoal]int
}

// Trainer runs the hierarchical PPO training loop described in
// docs/plans/11-hierarchical-meta-controller-and-subpolicies.md: a
// meta-controller and one sub-policy per subgoal, each an independent
// actorcritic.Params/ppo.TrainingNetwork/actorcritic.Adam triple,
// trained with the exact same GAE-then-PPO-clip recipe
// pkg/ppo.Trainer uses for a single flat policy. See
// ppo.NewTrainingNetwork and ppo.ComputeGAE, both reused directly here
// rather than reimplemented.
type Trainer struct {
	settings   config.Settings
	cfg        Config
	envFactory reinforce.EnvFactory

	metaParams  *actorcritic.Params
	metaNetwork *ppo.TrainingNetwork
	metaAdam    *actorcritic.Adam

	subParams   map[Subgoal]*actorcritic.Params
	subNetworks map[Subgoal]*ppo.TrainingNetwork
	subAdams    map[Subgoal]*actorcritic.Adam
}

// initWorkerIndex and metaShuffleWorkerIndex mirror pkg/ppo.Trainer's
// identically-purposed constants: distinct, negative worker indices
// (rollout-collection's own worker indices are always >= 0) used to
// derive one-off deterministic RNG streams via reinforce.WorkerRNG.
const (
	initWorkerIndex        = -1
	metaShuffleWorkerIndex = -2
)

// subShuffleWorkerIndex returns subgoal's own distinct, negative worker
// index for shuffling its minibatch training data, distinct from
// initWorkerIndex, metaShuffleWorkerIndex, and every other subgoal's.
func subShuffleWorkerIndex(subgoal Subgoal) int {
	return -3 - int(subgoal)
}

func lossConfigFrom(settings config.Settings) ppo.LossConfig {
	return ppo.LossConfig{
		ClipEpsilon: settings.ClipEpsilon,
		EntropyCoef: settings.EntropyCoef,
		ValueCoef:   settings.ValueCoef,
	}
}

// New constructs a Trainer from settings, cfg, and envFactory,
// validating both and sizing every network from envFactory's
// environment: the meta-controller's input size and every sub-policy's
// input size (base observation size plus cfg.NumSubgoals — see
// augmentObservation) come from env.ObservationSize(), and every
// sub-policy's output size comes from env.ActionSpace().
func New(settings config.Settings, cfg Config, envFactory reinforce.EnvFactory) (*Trainer, error) {
	if err := settings.Validate(); err != nil {
		return nil, err
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	initRNG := reinforce.WorkerRNG(settings.Seed, 0, initWorkerIndex)

	env, err := envFactory(initRNG)
	if err != nil {
		return nil, fmt.Errorf("hierarchical: constructing environment: %w", err)
	}

	metaParams := actorcritic.NewParams(initRNG, env.ObservationSize(), cfg.MetaHiddenSize, cfg.NumSubgoals)
	metaNetwork, err := ppo.NewTrainingNetwork(metaParams, lossConfigFrom(settings))
	if err != nil {
		return nil, fmt.Errorf("hierarchical: building meta-controller training network: %w", err)
	}

	subParams := make(map[Subgoal]*actorcritic.Params, cfg.NumSubgoals)
	subNetworks := make(map[Subgoal]*ppo.TrainingNetwork, cfg.NumSubgoals)
	subAdams := make(map[Subgoal]*actorcritic.Adam, cfg.NumSubgoals)
	for i := range cfg.NumSubgoals {
		subgoal := Subgoal(i)
		params := actorcritic.NewParams(initRNG, env.ObservationSize()+cfg.NumSubgoals, cfg.SubHiddenSize, env.ActionSpace())
		network, err := ppo.NewTrainingNetwork(params, lossConfigFrom(settings))
		if err != nil {
			return nil, fmt.Errorf("hierarchical: building sub-policy training network for subgoal %d: %w", subgoal, err)
		}

		subParams[subgoal] = params
		subNetworks[subgoal] = network
		subAdams[subgoal] = actorcritic.NewAdam(network.Actor.Parameters(), cfg.SubLearningRate)
	}

	return &Trainer{
		settings:    settings,
		cfg:         cfg,
		envFactory:  envFactory,
		metaParams:  metaParams,
		metaNetwork: metaNetwork,
		metaAdam:    actorcritic.NewAdam(metaNetwork.Actor.Parameters(), cfg.MetaLearningRate),
		subParams:   subParams,
		subNetworks: subNetworks,
		subAdams:    subAdams,
	}, nil
}

// RunEpoch runs one full hierarchical training epoch: collect
// settings.RolloutSize episodes in parallel, then run GAE-then-PPO-clip
// shuffled-minibatch Adam updates on the meta-controller (pooling every
// meta-transition across the batch) and on each subgoal (pooling only
// the segments where that subgoal was active across the batch).
func (tr *Trainer) RunEpoch(ctx context.Context, epoch int) (EpochStats, error) {
	rollouts, err := tr.collectRollouts(ctx, epoch)
	if err != nil {
		return EpochStats{}, err
	}

	metaRollouts := make([]*ppo.Rollout, len(rollouts))
	for i, r := range rollouts {
		metaRollouts[i] = r.MetaRollout
	}
	metaScored := scoreRollouts(metaRollouts, tr.settings.Gamma, tr.settings.GAELambda)
	metaSteps := flattenScoredRollouts(metaScored)
	normalizeAdvantages(metaSteps)
	metaShuffleRNG := reinforce.WorkerRNG(tr.settings.Seed, epoch, metaShuffleWorkerIndex)
	metaUpdateCount := trainNetwork(tr.metaNetwork, tr.metaAdam, metaSteps, tr.settings, metaShuffleRNG)

	sampleCount := len(metaSteps)
	subUpdateCounts := make(map[Subgoal]int, tr.cfg.NumSubgoals)
	for i := range tr.cfg.NumSubgoals {
		subgoal := Subgoal(i)

		var segments []*ppo.Rollout
		for _, r := range rollouts {
			segments = append(segments, r.SubRollouts[subgoal]...)
		}
		if len(segments) == 0 {
			continue
		}

		scored := scoreRollouts(segments, tr.settings.Gamma, tr.settings.GAELambda)
		steps := flattenScoredRollouts(scored)
		normalizeAdvantages(steps)

		shuffleRNG := reinforce.WorkerRNG(tr.settings.Seed, epoch, subShuffleWorkerIndex(subgoal))
		subUpdateCounts[subgoal] = trainNetwork(tr.subNetworks[subgoal], tr.subAdams[subgoal], steps, tr.settings, shuffleRNG)
		sampleCount += len(steps)
	}

	var returnSum float32
	for _, s := range metaScored {
		if len(s.Returns) > 0 {
			returnSum += s.Returns[0]
		}
	}

	return EpochStats{
		Epoch:           epoch,
		AverageReturn:   returnSum / float32(len(metaScored)),
		SampleCount:     sampleCount,
		MetaUpdateCount: metaUpdateCount,
		SubUpdateCounts: subUpdateCounts,
	}, nil
}

// collectRollouts collects settings.RolloutSize hierarchical episodes
// in parallel, bounded to settings.Workers concurrent goroutines,
// mirroring pkg/ppo.Trainer.collectRolloutsParallel's structure.
func (tr *Trainer) collectRollouts(ctx context.Context, epoch int) ([]*HierarchicalRollout, error) {
	rollouts := make([]*HierarchicalRollout, tr.settings.RolloutSize)
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
			rollout, err := collectHierarchicalTrajectory(ctx, tr.metaParams, tr.subParams, tr.cfg, tr.envFactory, tr.settings.EpisodeLen, rng)
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

// trainNetwork runs settings.PPOEpochs shuffled-minibatch Adam update
// passes over steps against network/adam and returns the number of
// Adam.Step calls applied, mirroring pkg/ppo.Trainer.RunEpoch's
// minibatch loop exactly (duplicated here because pkg/ppo.Trainer's own
// version is a private method tied to its own single network/optimizer
// fields, not reusable across packages).
func trainNetwork(network *ppo.TrainingNetwork, adam *actorcritic.Adam, steps []trainingStep, settings config.Settings, shuffleRNG *rand.Rand) int {
	updateCount := 0
	for range settings.PPOEpochs {
		shuffleRNG.Shuffle(len(steps), func(i, j int) {
			steps[i], steps[j] = steps[j], steps[i]
		})

		for start := 0; start < len(steps); start += settings.MinibatchSize {
			end := min(start+settings.MinibatchSize, len(steps))
			trainOnMinibatch(network, adam, steps[start:end])
			updateCount++
		}
	}
	return updateCount
}

// trainOnMinibatch replays every step in minibatch through network
// sequentially, accumulating gradients, then applies one Adam step
// averaged over len(minibatch) — identical in shape to
// pkg/ppo.Trainer.trainOnMinibatch, duplicated for the same reason as
// trainNetwork above.
func trainOnMinibatch(network *ppo.TrainingNetwork, adam *actorcritic.Adam, minibatch []trainingStep) {
	network.Actor.ZeroGrad()

	for _, step := range minibatch {
		copy(network.Actor.Input.Val.Data, step.Observation.Values)
		network.SetStep(step.Action, step.OldLogProb, step.Advantage, step.Return)

		network.Graph.Forward()
		network.Graph.Backward()
	}

	adam.Step(len(minibatch))
}

// Params exposes the meta-controller's and every sub-policy's current
// parameters, e.g. so a future checkpoint-tooling addition (out of
// scope here, see docs/plans/11-hierarchical-meta-controller-and-subpolicies.md's
// "Explicitly out of scope") can persist all N+1 networks.
func (tr *Trainer) Params() (meta *actorcritic.Params, subs map[Subgoal]*actorcritic.Params) {
	return tr.metaParams, tr.subParams
}
