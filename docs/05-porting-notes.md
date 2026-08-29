# Porting notes: cRL (C) to cRL-go

This project is a Go reimplementation of [github.com/harshbhatt7585/cRL](https://github.com/harshbhatt7585/cRL), a from-scratch C REINFORCE trainer. This doc explains every deliberate difference from the original, and why.

## What stayed the same

The core algorithm is unchanged: a 3-layer MLP policy network, trained with vanilla REINFORCE (discounted reward-to-go, a batch-wide mean/std baseline, full-batch gradient accumulation, plain SGD), on the same simplified grid-foraging environment. Numerically sensitive operations (softmax's max-subtraction trick, the `log`/divide-by-zero clamps, the naive one-pass variance formula with its clamp) were preserved exactly, including their known numerical quirks — see `04-numerical-stability-notes.md`.

## Structural differences

### No memory arena

The original's `arena.c` is a bump allocator: one big `malloc`, then cheap pointer-bumping allocations out of it, freed all at once. Its purpose was avoiding per-call `malloc`/`free` overhead in a hot loop that runs potentially tens of millions of times across a full training run. Go has garbage collection, which removes the correctness need for manual arena management; whether it's also fast enough without one is a separate, empirical question. This port made a deliberate choice: write straightforward, idiomatic Go (plain struct/slice allocation) first, and treat introducing an arena-like buffer-reuse strategy as a follow-up to be justified by profiling data (`pprof`, the benchmark in `pkg/reinforce/trainer_test.go`'s `BenchmarkRunEpoch`) rather than applied speculatively.

### The autograd "vtable" became a Go interface

The original's `VarType` (`autograd.h`) is a hand-rolled vtable: a struct of function pointers (`shape`, `forward`, `backward`) that C uses to fake polymorphism. Go has real interfaces for exactly this purpose. `pkg/autograd`'s `Op` interface (see `pkg/autograd/var.go`) is implemented by small, stateless, zero-size structs (`pkg/autograd/ops.go`) that are constructed fresh at each call site rather than stored in any shared table — this sidesteps needing any package-level variable at all for "which ops exist," which both simplifies the code and avoids a design tension with a strict "no package-level variables" convention.

### Graph construction: O(V+E) instead of O(n²)

The original's `build_graph` (`autograd.c`) does an iterative depth-first search using a manually-managed array as a stack, including an O(n) "splice out an already-queued node and re-insert it" step performed inside the main loop — making the whole traversal O(n²) in the worst case. `pkg/autograd/graph.go`'s `BuildGraph` instead uses a straightforward recursive post-order depth-first search keyed by `Var` identity (a `map[*Var]bool`), which is O(V+E) and doesn't need to know the total number of vars ahead of time (the original needed a fixed-size array bound). For graphs this small (a handful of nodes per network), the performance difference is irrelevant in practice; the simplification is about code clarity and about removing a latent scaling hazard, not about a measured bottleneck.

### Randomness: one generator, not two, and no shared global state

The original mixes two independent random sources: a hand-rolled PCG32 generator (`prng.c`) with **global mutable state**, used for weight initialization and action sampling, and a separate call to libc's `arc4random_uniform` (`randn` in `prng.c`) for food placement, which is never seeded from the same place. This means the environment's randomness isn't actually reproducible even when the PCG state is seeded, and it relies on `arc4random_uniform` being available (a BSD-originated function not universally present across all C standard libraries).

This port uses exactly one generator family throughout: Go's standard library `math/rand/v2`, which implements PCG natively (`rand.NewPCG`), so no custom PRNG code needed to be written at all. Every `*rand.Rand` instance is constructed explicitly and passed to whatever needs it (`pkg/mat.Matrix.FillRand`, `pkg/snakeenv.Env`, `pkg/reinforce.SampleAction`) — there is no package-level/global RNG anywhere in this codebase, both because Go's GC-friendly allocation style makes explicit threading cheap and because a shared global generator would need external synchronization to be used safely from multiple goroutines (see "Concurrency" below).

### Concurrency: parallel rollout collection, sequential training

The original is entirely single-threaded. This port parallelizes the rollout-collection phase (`Trainer.collectRollouts` in `pkg/reinforce/trainer.go`) across goroutines, since every trajectory in a batch is independent given the current (frozen) policy weights. Making this both race-free and reproducible required two supporting design decisions:

1. **Separating shared, read-only weights from private, per-goroutine scratch space.** `policy.Params` (`pkg/policy/params.go`) holds the actual weight/bias matrices. Each rollout goroutine builds its own `policy.InferenceNetwork` (`pkg/policy/network.go`) via `autograd.Constant`, which wraps `Params`' matrices *by reference* (no copy) for the weight `Var`s, while allocating fresh, private matrices for every intermediate activation. This means many goroutines can run forward passes concurrently against the same weights (read-only, so no synchronization needed) while never touching each other's scratch buffers. Weights are only ever mutated by the *sequential* gradient-accumulation and SGD-update phase, which runs strictly after every rollout goroutine has finished (`RunEpoch`'s `wg.Wait()` in `pkg/reinforce/trainer.go`).
2. **Deterministic, race-free per-worker random sources.** Rather than sharing one `*rand.Rand` across goroutines (which is not safe for concurrent use without a mutex, and a mutex would partly defeat the point of parallelizing) or letting each goroutine spawn its own arbitrarily-seeded generator (which would make results irreproducible run-to-run), `workerRNG` (`pkg/reinforce/seed.go`) derives a `*rand.Rand` for each `(epoch, worker index)` pair as a pure function of a single master seed, using a SplitMix64-style bit-mixing step. A fixed master seed therefore always produces the same set of per-worker random streams, regardless of goroutine scheduling order — `TestRunEpochIsDeterministicForAFixedSeed` in `pkg/reinforce/trainer_test.go` verifies this directly, and `go test -race` (see the repository's test suite) confirms no data races in the concurrent rollout path.

The backward/gradient-accumulation/SGD-update phase remains sequential. Parallelizing gradient accumulation across trajectories would require either a reduction step (merging per-goroutine gradients) or synchronized accumulation into shared gradient buffers, either of which adds real complexity; since this port's guiding principle was "write idiomatic Go first, optimize with evidence," this was left as a documented future enhancement rather than spec'd out speculatively.

### Environment API: a single `Step` method

The original scatters environment-state mutation across `take_action`, `get_reward`, and inline bookkeeping in the training loop (`env.c`). `snakeenv.Env.Step` (`pkg/snakeenv/env.go`) consolidates this into the conventional RL-environment shape `Step(action) -> (reward, done)`. The underlying game rules (reward values, boundary detection, food relocation) are unchanged.

### Trajectory storage: slices, not a fixed 1024-slot array

The original's `ReplayBuffer` (`env.c`) embeds a fixed array of 1024 `Trajactory` structs (each itself containing several fixed 100-element arrays) as a stack-local variable inside `train()`, even though only `rollout_size` (64) of those 1024 slots are ever used — a multi-megabyte stack allocation sized far larger than necessary. `pkg/reinforce.Trajectory` (`pkg/reinforce/trajectory.go`) uses ordinary Go slices, pre-allocated with capacity equal to the actual episode length and grown only as needed.

### Input dimension computed from grid size, not hardcoded

The original's `model.c` hardcodes the policy network's input layer size to `76`, which is silently equal to `2 * grid_size + 4` only for the specific `grid_size = 36` used elsewhere in the program — changing the grid size anywhere else in the C code would silently desynchronize this constant, likely corrupting the state vector without any error. `snakeenv.StateVectorSize(gridSize)` (`pkg/snakeenv/env.go`) computes this from the actual grid size (and the number of possible actions), and `policy.Params`/`pkg/reinforce.Trainer` both derive the input layer size from it rather than hardcoding a number.

### Naming: no implied critic network

The original's `create_actor_model` (`model.c`) name suggests an actor-critic architecture, but there is no critic (no second network predicting expected returns) anywhere in the algorithm — only a simple batch-wide mean/std baseline (see `03-policy-gradients-and-reinforce.md`). This port's `policy` package avoids "actor" terminology for this reason, to avoid implying a training approach that isn't actually implemented. The environment is still called `snakeenv` (matching the original's "Snake" framing) even though it has no snake body, tail, growth, or self-collision — only a single point-agent and boundary detection — this is called out explicitly in `pkg/snakeenv/action.go`'s package documentation and in `03-policy-gradients-and-reinforce.md` rather than silently renamed, since "Snake" is how the original project (and its README/training screenshot) describes it.

This claim about `pkg/policy` remains true even though the module has since grown a real critic: `pkg/actorcritic` is a separate package (a shared trunk feeding both a softmax policy head and a scalar value head), added to support PPO-style training without changing `pkg/policy`'s shape or its REINFORCE-only meaning. `pkg/reinforce` still trains exclusively against `pkg/policy`.

## Validation approach

This port does not attempt bit-for-bit numeric parity with the original C binary (the two implementations differ enough — a different random number generator, a restructured graph-building algorithm, a different concurrency model — that end-to-end parity was never realistically achievable, and it wasn't a goal). Instead, correctness is validated through:

- **Gradient checking** (`pkg/autograd/gradcheck_test.go`): every autograd operation's analytically-computed backward pass is checked against a numerical (finite-difference) estimate, which is the standard technique for validating a hand-written backpropagation implementation (see `02-autograd-and-backpropagation.md`).
- **Unit and property tests** throughout every package (matrix shape/value tests, softmax normalization, deterministic environment transitions, config-loading precedence, etc.).
- **A qualitative training-trend test** (`TestTrainingImprovesAverageReturn` in `pkg/reinforce/trainer_test.go`): running the full pipeline end-to-end on a small, fast-converging configuration and asserting the average return actually improves over training, directly demonstrating that rollout collection, return/advantage computation, gradient accumulation, and the SGD update are all wired together correctly.
- **The race detector** (`go test -race`) over the concurrent rollout-collection path.

## Attribution and licensing

The algorithm, environment design, and overall program structure are derived from [github.com/harshbhatt7585/cRL](https://github.com/harshbhatt7585/cRL). At the time this port was written, that repository had no `LICENSE` file. Anyone redistributing this Go port (or the original C project) should confirm licensing terms directly with the original author before doing so.
