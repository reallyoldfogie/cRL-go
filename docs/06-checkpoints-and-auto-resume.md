# Checkpoints and auto-resume

Both training commands support two checkpoint workflows:

- `-checkpoint-in`/`-checkpoint-out`: manually load or save one specific file.
- `-checkpoint-dir`/`-checkpoint-interval`: automatically resume the latest checkpoint in a directory, periodically save numbered checkpoints, and save a final checkpoint when training finishes.

The directory workflow is intended for long-running or interrupted training because it preserves run-progress metadata and continues epoch numbering after a restart.

## Automatic checkpointing

REINFORCE:

```sh
go run ./cmd/train \
  -epochs=500 \
  -checkpoint-dir=checkpoints/reinforce \
  -checkpoint-interval=50
```

PPO:

```sh
go run ./cmd/train-ppo \
  -epochs=500 \
  -checkpoint-dir=checkpoints/ppo \
  -checkpoint-interval=50
```

On a new directory, training starts from freshly initialized parameters. On a later invocation using the same directory, the command loads the highest-epoch matching checkpoint and starts at the following epoch.

`-epochs` is the total target epoch count, not the number of additional epochs. For example, if the latest checkpoint records epoch 199, running with `-epochs=500` continues at epoch 200 and finishes after epoch 499.

REINFORCE checkpoints use names such as:

```text
policy-epoch-000000049.json
```

PPO checkpoints use names such as:

```text
ppo-epoch-000000049.json
```

The prefixes keep both formats distinguishable if they share a directory. `pkg/checkpoint.Latest` parses the epoch number and chooses the highest one rather than relying on directory-listing order.

## Saved metadata

Every newly written checkpoint contains:

- The checkpoint schema version.
- An environment identifier derived from `-env` and `-grid-size`, such as `snake:36`.
- Network input, hidden, and output sizes.
- The last completed epoch.
- The best average return observed so far.
- The total number of gradient updates applied so far.
- All network weights and biases.

The environment identifier prevents loading a checkpoint into an incompatible environment even if the network dimensions happen to match. `pkg/policy` continues to load legacy checkpoints written before schema and environment metadata existed; those legacy files have no progress metadata.

## Manual checkpoint files

The original single-file flags remain supported:

```sh
go run ./cmd/train \
  -epochs=500 \
  -checkpoint-out=checkpoints/policy.json

go run ./cmd/train \
  -epochs=500 \
  -checkpoint-in=checkpoints/policy.json
```

The PPO command accepts the same flags using its actor-critic checkpoint format:

```sh
go run ./cmd/train-ppo \
  -epochs=500 \
  -checkpoint-out=checkpoints/ppo.json
```

Manual `-checkpoint-in` restores parameters but retains its historical behavior of starting the command's epoch numbering at zero. Use `-checkpoint-dir` when epoch and update counters must continue across restarts.

If both `-checkpoint-in` and a resumable checkpoint in `-checkpoint-dir` are present, the command returns an error instead of silently choosing one.

## Inspecting checkpoints

`cmd/checkpoint-tool` reads the metadata fields common to both REINFORCE and PPO checkpoint formats.

List checkpoints in a directory:

```sh
go run ./cmd/checkpoint-tool list checkpoints/ppo
```

Inspect one checkpoint:

```sh
go run ./cmd/checkpoint-tool info \
  checkpoints/ppo/ppo-epoch-000000049.json
```

Compare progress metadata from two checkpoints:

```sh
go run ./cmd/checkpoint-tool compare \
  checkpoints/ppo/ppo-epoch-000000049.json \
  checkpoints/ppo/ppo-epoch-000000099.json
```

The tool intentionally does not print weight arrays.
