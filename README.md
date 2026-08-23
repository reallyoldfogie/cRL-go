# cRL-go

A Go reimplementation based on [`harshbhatt7585/cRL`](https://github.com/harshbhatt7585/cRL) — a small, from-scratch reinforcement learning library. This port trains a policy network to play a simple grid-foraging game using REINFORCE, with:

- A stdlib-only autograd engine (`pkg/autograd`), built around Go interfaces rather than the original's hand-rolled function-pointer vtable.
- A small dense matrix library (`pkg/mat`) with the matrix ops, activations, REINFORCE loss/gradients, and general-purpose elementwise ops (multiply, min, negate, exp, log) for composing policy-gradient objectives beyond REINFORCE.
- A 3-layer MLP policy network (`pkg/policy`) with Xavier/Glorot initialization, and JSON checkpoint save/load so training can resume across sessions.
- An environment-agnostic training core (`pkg/rl`) with two example environments: a grid-foraging environment (`pkg/snakeenv`) and a goal-seeking gridworld (`pkg/gridworldenv`).
- A REINFORCE trainer (`pkg/reinforce`) with concurrent rollout collection.
- Config-file + CLI-flag support for hyperparameters (`pkg/config`, `configs/config.json`).

See `docs/` for a from-first-principles explanation of the machine learning concepts involved (neural networks, backpropagation, policy gradients, numerical stability), and `docs/05-porting-notes.md` for a detailed account of every deliberate difference from the original C implementation.

## Attribution

The algorithm, initial environment design, and overall program structure were derived from [`github.com/harshbhatt7585/cRL`](https://github.com/harshbhatt7585/cRL).

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

A checkpoint records the policy's layer sizes alongside its weights, so loading one with a mismatched `-grid-size`/`-hidden-size` (or a different `-env`'s action/observation space) fails with a clear error instead of silently producing a broken network.

### Tests

```sh
go test ./...          # full suite, including a slower training-trend test
go test -short ./...   # skip the slower training-trend test
go test -race ./...    # validate the concurrent rollout-collection path
```
