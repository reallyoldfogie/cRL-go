# Context-aware and long-lived environments

`01`-`11` all trained against toy environments that behave the same, predictable way: `Reset`/`Step` return instantly, and it's always fine to build a brand-new one for every single episode. This doc covers what changes once an environment isn't so well-behaved — for example, a real, live game session instead of a few in-memory numbers.

## Why a step might need to be cancelled

`rl.Environment.Reset`/`Step` both take a `context.Context` as their first parameter. The motivation: a live session might be waiting on a network round-trip that never comes back (a dropped connection), or you might want to enforce "give up on this step after five seconds" rather than hang forever. `context.Context` is Go's standard mechanism for exactly this — it lets a caller signal "stop waiting" from *outside* the function doing the waiting, without that function needing to poll anything on its own.

None of this project's toy environments (`pkg/snakeenv`, `pkg/gridworldenv`, `pkg/hierarchicalgridworld`) ever block on anything, so their `Adapter.Reset`/`Step` simply accept `ctx` and ignore it — that's enough to satisfy the interface without needing to *do* anything with it. A real, live-session adapter (outside this module) would check `ctx.Done()`, or pass `ctx` through to whatever it's actually waiting on (a network call, for instance).

## Two ways to get an environment: rebuild vs. reuse

Every trainer needs a way to obtain an `rl.Environment` to run episodes against. This project offers two:

- **`reinforce.EnvFactory`** builds a brand-new environment for *every* episode. This is fine — even desirable, since it lets many episodes run on separate goroutines at once with zero shared state — when constructing one is cheap, like every toy environment here (allocating a handful of small matrices).
- **`reinforce.PersistentEnvFactory`** builds *one* environment, *once*, and reuses it across every episode of every epoch by calling `Reset` between episodes rather than throwing it away and building a new one. This matters for anything expensive or stateful to set up: a live game session (connecting, spawning, waiting for the world to load) isn't something you want to redo for every single episode.

Both are the exact same function shape (`func(rng *rand.Rand) (rl.Environment, error)`) — the difference is entirely in how often a `Trainer` calls it and what it does with the result afterward, not in what the function itself looks like.

## Why a persistent environment can't be trained on in parallel

Recall from `09` that `RunEpoch` normally collects `RolloutSize` episodes at once across `Workers` goroutines, since each goroutine builds its *own* environment via `EnvFactory`. That trick depends entirely on there being enough separate environment instances to hand one to each goroutine. A persistent environment breaks that assumption outright — there's only ever *one* instance, so handing it to several goroutines at once would mean multiple goroutines calling `Reset`/`Step` on the *same* live session simultaneously, which doesn't make sense (whose action actually happened?).

`NewWithPersistentEnv` (available on both `pkg/reinforce.Trainer` and `pkg/ppo.Trainer`) sidesteps this by collecting all `RolloutSize` episodes one at a time, sequentially, on the calling goroutine instead — `Workers` simply isn't consulted on this path at all.

## Choosing between the two

Nothing about training itself changes: `RunEpoch` still runs the exact same collect-a-batch-then-train recipe either way, and a `Trainer` picks between the parallel path and the sequential path automatically based on which constructor built it (`New` vs. `NewWithPersistentEnv`). A caller doesn't need to know or care about workers or goroutines — only whether their environment is cheap to rebuild, or expensive and stateful.

## Where to go next

This is the last stop in this curriculum. `05-porting-notes.md` and `06-checkpoints-and-auto-resume.md` cover this project's remaining ground — respectively, everything that differs from the original C project this was ported from, and the practical workflow for saving/resuming/inspecting trained weights across sessions.
