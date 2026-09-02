# cRL-go

A Go reimplementation based on [`harshbhatt7585/cRL`](https://github.com/harshbhatt7585/cRL) — a small, from-scratch reinforcement learning library. This port trains a policy network to play a simple grid-foraging game using REINFORCE, with:

- A stdlib-only autograd engine (`pkg/autograd`), built around Go interfaces rather than the original's hand-rolled function-pointer vtable.
- A small dense matrix library (`pkg/mat`) with the matrix ops, activations, REINFORCE loss/gradients, and general-purpose elementwise ops (multiply, min, negate, exp, log) for composing policy-gradient objectives beyond REINFORCE.
- A 3-layer MLP policy network (`pkg/policy`) with Xavier/Glorot initialization, and JSON checkpoint save/load so training can resume across sessions.
- An actor-critic MLP (`pkg/actorcritic`), a separate network with the same shared trunk plus a scalar value head alongside the policy head, laying the groundwork for PPO-style training without changing `pkg/policy`'s REINFORCE-only shape. Its checkpoints (and, as of this port, `pkg/policy`'s) carry a schema version and an environment identifier, so a checkpoint can't be silently restored into an incompatible environment.
- A PPO trainer (`pkg/ppo`) built on top of `pkg/actorcritic`: Generalized Advantage Estimation (GAE(λ)), rollout collection that also captures each step's action log-probability and value estimate, the clipped-surrogate policy loss plus a value loss and an entropy bonus, and an Adam optimizer (`actorcritic.Adam`) driving `PPOEpochs` shuffled-minibatch updates per collected batch — the `cmd/train-ppo` counterpart to `cmd/train`'s REINFORCE trainer.
- A hierarchical RL trainer (`pkg/hierarchical`) that pairs a meta-controller with one PPO sub-policy per subgoal, each trained with the same `ppo.TrainingNetwork`/`ppo.ComputeGAE` machinery as `pkg/ppo`, plus a multi-objective toy environment (`pkg/hierarchicalgridworld`) and its own binary (`cmd/train-hierarchical`).
- An environment-agnostic training core (`pkg/rl`) with two example environments: a grid-foraging environment (`pkg/snakeenv`) and a goal-seeking gridworld (`pkg/gridworldenv`). `rl.Environment.Reset`/`Step` take a `context.Context`, and both `pkg/reinforce.Trainer` and `pkg/ppo.Trainer` can be built with an `EnvFactory` (rebuilt per episode) or a `PersistentEnvFactory` (built once, reused via `Reset` across episodes) for environments that are expensive to construct or need to stay alive across an episode boundary.
- A REINFORCE trainer (`pkg/reinforce`) with concurrent rollout collection.
- Config-file + CLI-flag support for hyperparameters (`pkg/config`, `configs/config.json`).
- Optional per-epoch CSV metrics export (`-metrics-out`, `pkg/metrics`) on every `cmd/train*` binary, and a `cmd/watch` binary that renders a trained checkpoint acting in `pkg/snakeenv`/`pkg/gridworldenv` step by step in the terminal (via a `Render()` method on each environment, built on `pkg/gridrender`) for eyeballing behavior rather than only reading return numbers.

See `docs/` for a from-first-principles explanation of the machine learning concepts involved: neural networks and backpropagation (`01`-`02`), REINFORCE and numerical stability (`03`-`04`), actor-critic networks, Generalized Advantage Estimation, PPO's clipped objective, and Adam/minibatch training (`07`-`09`), live inference/action masking (`10`), hierarchical meta-controller/sub-policy training (`11`), and context-aware/long-lived environments (`12`). `docs/05-porting-notes.md` covers every deliberate difference from the original C implementation, and `docs/06-checkpoints-and-auto-resume.md` covers the checkpoint save/resume/inspect workflow.

## Attribution

The algorithm, initial environment design, and overall program structure were derived from [`github.com/harshbhatt7585/cRL`](https://github.com/harshbhatt7585/cRL).

## Versioning

This module is tagged with standard Go semantic-versioning git tags (`vMAJOR.MINOR.PATCH` on `main`), currently pre-1.0 (`v0.x.y`) while the exported surface (`pkg/rl`, `pkg/reinforce`, `pkg/ppo`, `pkg/policy`, `pkg/actorcritic`, `pkg/config`) is still evolving. Pre-1.0:

- Any change to an exported API in those packages — including a breaking one — bumps `MINOR`. Go's module rules allow a `v0.x` `MINOR` bump to carry breaking changes, which fits this project's current rate of change better than committing to `v1` stability now.
- Internal-only changes (implementation details, tests, docs) bump `PATCH`.

`pkg/rl`'s interfaces (`Environment`, `Observation`, `Action`) are the surface external consumers depend on most directly, so a tag whose message doesn't call out a change there risks silently breaking a downstream build. Release notes live in each annotated tag's own message (`git tag -a vX.Y.Z -m "..."`) rather than a separate changelog file.

## Quickstart

Build and run from the module root:

```sh
go run ./cmd/train
```

This loads hyperparameters from `configs/config.json` and trains for the configured number of epochs, printing progress like:

```
Epoch 0 | Average return: -8.012 | Samples: 89 | Return std: 3.976
...
```

### Configuration

Hyperparameters live in `configs/config.json`. Any CLI flag overrides the config file value for that field:

```sh
go run ./cmd/train -epochs=500 -learning-rate=0.1 -workers=4
```

Run `go run ./cmd/train -h` for the full list of flags.

### Checkpoints

By default, every run starts from a freshly Xavier/Glorot-initialized policy and the trained weights are discarded when the process exits. To persist weights across sessions, use `-checkpoint-out` to save them and `-checkpoint-in` to resume from a previous save:

```sh
# Train from scratch and save the resulting weights.
go run ./cmd/train -epochs=500 -checkpoint-out=checkpoints/policy.json

# Resume training from those weights instead of starting over.
go run ./cmd/train -epochs=500 -checkpoint-in=checkpoints/policy.json -checkpoint-out=checkpoints/policy.json
```

A checkpoint records the policy's layer sizes alongside its weights, so loading one with a mismatched `-grid-size`/`-hidden-size` (or a different `-env`'s action/observation space) fails with a clear error instead of silently producing a broken network. It also records an environment identifier derived from `-env`/`-grid-size` (e.g. `snake:36`), so a checkpoint trained with one `-env`/`-grid-size` combination is rejected outright if loaded with a different one, even in the rare case where the raw layer sizes happen to coincide.

For periodic saves and automatic resume (including continued epoch/best-return/update counters), use `-checkpoint-dir` with `-checkpoint-interval`:

```sh
go run ./cmd/train -epochs=500 -checkpoint-dir=checkpoints/reinforce -checkpoint-interval=50
go run ./cmd/train-ppo -epochs=500 -checkpoint-dir=checkpoints/ppo -checkpoint-interval=50
```

Inspect the resulting files with `go run ./cmd/checkpoint-tool list <dir>` or `info <checkpoint>`. See `docs/06-checkpoints-and-auto-resume.md` for the complete workflow and metadata format.

### Metrics export

Every `cmd/train*` binary accepts an optional `-metrics-out` flag, writing one CSV row per epoch (in addition to the usual stdout printing) for plotting or analysis with any external tool:

```sh
go run ./cmd/train -epochs=500 -metrics-out=metrics.csv
```

Columns differ slightly per binary to match its own progress fields (e.g. `cmd/train-ppo` includes `update_count`; `cmd/train-hierarchical` includes `meta_update_count` plus one `sub_N_updates` column per `-num-subgoals`).

### Tests

```sh
go test ./...          # full suite, including a slower training-trend test
go test -short ./...   # skip the slower training-trend test
go test -race ./...    # validate the concurrent rollout-collection path
```

## PPO quickstart

`cmd/train-ppo` trains the actor-critic network (`pkg/actorcritic`) with PPO (`pkg/ppo`) instead of REINFORCE, sharing `cmd/train`'s config file, CLI flags, `-env` choice, and `-checkpoint-in`/`-checkpoint-out` conventions, plus its own PPO-specific flags:

```sh
go run ./cmd/train-ppo -epochs=500 -clip-eps=0.2 -entropy-coef=0.01 -value-coef=0.5 -gae-lambda=0.95 -ppo-epochs=4 -minibatch-size=64
```

Run `go run ./cmd/train-ppo -h` for the full list of flags. `pkg/policy`/`pkg/reinforce` (REINFORCE) and `pkg/actorcritic`/`pkg/ppo` (PPO) are independent end to end: separate network types, separate checkpoint formats, and separate `cmd/` binaries, sharing only `pkg/config`'s settings and a handful of algorithm-agnostic helpers (`reinforce.EnvFactory`, `reinforce.SampleAction`/`reinforce.SampleMaskedAction`, `reinforce.WorkerRNG`). Both network types also expose an `Actor` (`policy.Actor`, `actorcritic.Actor`) for making a single live decision outside of training — see `docs/10-live-inference-and-action-masking.md`. `Actor.ActWithInfo` returns a full `rl.Decision` (the sampled action's probability, the distribution before and after any action mask, and a value estimate where available) instead of only the sampled action, for auditing why a decision was made; `cmd/watch` (below) uses it.

## Hierarchical RL quickstart

`cmd/train-hierarchical` trains a meta-controller plus one sub-policy per subgoal (`pkg/hierarchical`) against the multi-objective `pkg/hierarchicalgridworld` environment, sharing `cmd/train-ppo`'s config file and PPO-hyperparameter flags plus its own hierarchy-specific flags:

```sh
go run ./cmd/train-hierarchical -epochs=500 -num-subgoals=4 -subgoal-interval=8 -meta-hidden-size=16 -sub-hidden-size=16
```

Run `go run ./cmd/train-hierarchical -h` for the full list of flags. It supports the same `-checkpoint-in`/`-checkpoint-out`/`-checkpoint-dir`/`-checkpoint-interval` conventions as `cmd/train-ppo` (see "Checkpoints" above), saving the meta-controller's and every sub-policy's weights as one checkpoint file per generation. See `docs/11-hierarchical-meta-controller-and-subpolicies.md` for how the meta-controller/sub-policy split works and why it reuses `pkg/ppo`'s training machinery unchanged.

## Watch mode

`cmd/watch` loads a checkpoint saved by `cmd/train`, `cmd/train-ppo`, or `cmd/train-hierarchical` and renders one episode of it acting in the corresponding environment step by step in the terminal, for eyeballing whether a trained policy behaves sensibly:

```sh
go run ./cmd/train -env=snake -grid-size=36 -epochs=200 -checkpoint-out=checkpoints/reinforce.json
go run ./cmd/watch -env=snake -algo=reinforce -grid-size=36 -checkpoint=checkpoints/reinforce.json
```

Use `-algo=ppo` for a `cmd/train-ppo` checkpoint, or `-algo=hierarchical -env=hierarchicalgridworld` (with matching `-num-subgoals`/`-subgoal-interval`) for a `cmd/train-hierarchical` checkpoint — driven by `hierarchical.Actor` (`pkg/hierarchical/actor.go`), which reports the currently-active subgoal and when the meta-controller made a new decision alongside each step. `-delay` controls the pause between steps and `-episode-len` caps how long it renders before giving up. Every mode prints the chosen action's sampled probability (and, where available, a value estimate) via `Actor.ActWithInfo` — see `docs/plans/16-decision-auditing-and-explainability.md`. Run `go run ./cmd/watch -h` for the full list of flags.
