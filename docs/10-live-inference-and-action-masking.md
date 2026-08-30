# Live inference and action masking

`01`-`09` covered building and training a network. This doc covers something different: actually *using* an already-trained network to make one decision, outside of training — the `Actor` type (`pkg/policy/actor.go`, `pkg/actorcritic/actor.go`) — and **action masking**, for situations where not every action a network was trained with is actually available right now.

## From weights to a decision: what "using" a network requires

Recall from `01` that a forward pass produces a probability for every action, and from `03` that an action is *sampled* from that distribution rather than always taking the single best-looking one. Doing this from outside `pkg/policy`/`pkg/actorcritic` means: build an `InferenceNetwork` over some trained `Params`, copy an observation into its `Input`, call `Graph.Forward()`, then sample from the resulting output. That's several steps of "how" for something that's conceptually a single question: "given this observation, what should I do?"

## `Actor`: a single entry point

`Actor` (`NewActor`, then `Act(observation, mask, rng)`) wraps that entire sequence behind one call:

```go
actor, err := policy.NewActor(params)
action, err := actor.Act(observation, nil, rng)
```

`Act` builds a fresh `InferenceNetwork` on every call — deliberately: `InferenceNetwork` was already designed to be cheap enough to build once per rollout-collection goroutine per training epoch (see `01`), so building one per individual decision is a natural extension of that same assumption, not a new performance concern this project needed to solve.

## Inferring against a snapshot, not the live weights

`NewActor` doesn't keep a reference to the `Params` you pass it — it immediately takes a **snapshot** (`Params.Snapshot()`, a full, independent copy of every weight/bias matrix) and holds *that* instead. Every `Act` call reads from this snapshot, never from the live `Params`. To pick up a later training update, a caller calls `actor.Refresh(params)`, which takes a fresh snapshot and swaps it in.

Why not just keep reading the live `Params` directly, so updates show up automatically? Because "the live `Params`" might be a network a trainer is *concurrently* updating (see `pkg/actorcritic/adam.go`'s optimizer, or `ApplyGradientStep`) — for example, a live decision-making process running alongside continued training. Reading a matrix while another goroutine is in the middle of writing to it doesn't just risk stale data, it risks reading a *torn* value (part of the old weights, part of the new). A snapshot sidesteps this entirely: once copied, nothing else ever writes to it, so `Act` can read it freely, from any number of goroutines, with no coordination needed at all.

## Why you'd want to exclude some actions: illegal actions

A trained network always outputs a probability for every action it was trained with — even ones that make no sense in the *current* state. A game character's "attack" action might have no enemy nearby to attack; a "mine" action might have no block adjacent to mine. If you always sample from the full, unrestricted distribution, you'll eventually draw an action that simply can't be carried out right now. **Action masking** is the fix: restrict sampling to only the actions currently known to be legal, using a `[]bool` the same length as the action space (`true` = legal).

## Renormalizing a probability distribution over fewer options

Say a network outputs `[0.1, 0.2, 0.3, 0.4]` over four actions, but only actions `0` and `2` are legal right now. You can't just zero out the illegal ones and sample directly from what's left — `0.1` and `0.3` no longer sum to `1`, so they're no longer a valid probability distribution. The fix is to divide each remaining probability by the sum of *all* the remaining ones:

```
allowed sum = 0.1 + 0.3 = 0.4
renormalized action 0 = 0.1 / 0.4 = 0.25
renormalized action 2 = 0.3 / 0.4 = 0.75
```

These two numbers now sum to `1` and can be sampled from exactly like any ordinary distribution. Notice the *relative* proportions between the still-legal actions are unchanged — action `2` was three times as likely as action `0` before masking, and still is afterward — only the absolute scale changed, to once again occupy the full `[0, 1]` probability space. This is exactly what `reinforce.SampleMaskedAction` (`pkg/reinforce/episode.go`) computes before sampling.

## No legal actions: an explicit error, not a silent guess

If a mask excludes every action, there's nothing left to renormalize or sample from. `SampleMaskedAction` (and `Actor.Act`, which calls it) returns an explicit error in this case, rather than falling back to some arbitrary action. This matters because a caller reaching this state usually indicates a real bug elsewhere (e.g. whatever computed the mask incorrectly decided nothing was legal) — surfacing that loudly is far more useful than having the agent quietly do something nonsensical and continue on as if nothing were wrong.

## Matching the unmasked case exactly, not just approximately

A subtle but important detail: sampling with a `nil` mask, or a mask where every entry happens to be `true`, is defined to behave *exactly* the same as sampling with no masking logic involved at all — not merely "close to it." Even a mathematically correct renormalization, applied when nothing was actually excluded, would divide every probability by a sum that's *supposed* to be `1` but, due to ordinary floating-point rounding, might be `0.9999999` or `1.0000001` instead — which could occasionally shift a result right at a boundary between two actions. `SampleMaskedAction` sidesteps this by detecting the "nothing is actually excluded" case up front and delegating straight to the plain, unmasked sampling function in that case, guaranteeing bit-for-bit identical results rather than merely approximately equivalent ones.

## Keeping a trainer's updates safe to read concurrently

Snapshotting solves the *reading* side, but taking a snapshot still has to read a consistent set of matrix values — if it read halfway through a gradient update, it would copy that same torn data. `Params` guards against this with a read-write lock: `Snapshot` briefly holds the read side while it copies every matrix, and a trainer applying a gradient update (`pkg/reinforce.Trainer.RunEpoch`, `pkg/ppo.Trainer.RunEpoch`) holds the write side (`Params.Lock`/`Unlock`) for the duration of that update. Multiple `Snapshot` calls can happen at once (many readers are fine), but none of them can overlap with a write.

This lock is held only around the update itself, not around every read — an `Actor` never touches it at all, since it only ever reads its own private snapshot. That's the whole point of the snapshot design: a live decision-making process never has to wait on, or contend with, an in-progress training update.

## Why `pkg/policy` and `pkg/actorcritic` don't share one implementation

One architectural detail worth understanding, even as a beginner: `pkg/actorcritic.Actor.Act` calls `reinforce.SampleMaskedAction` directly, but `pkg/policy.Actor.Act` has its own private copy of the same logic instead of doing the same thing. The reason is a **circular dependency**: `pkg/reinforce` already needs to depend on `pkg/policy` (for its own training loop, see `03`), and Go doesn't allow two packages to depend on each other — so `pkg/policy` importing `pkg/reinforce` back would be an error the compiler rejects outright. Duplicating one small, independently-tested function in `pkg/policy` is the practical fix, rather than restructuring the whole package layout to avoid it.

## Where to go next

This closes out the algorithm-by-algorithm curriculum that started in `01`: a single linear layer, through a full computation graph, REINFORCE, actor-critic networks and GAE, PPO's clipped objective, Adam and minibatch training, and finally making a single live decision from a trained network. `05-porting-notes.md` and `06-checkpoints-and-auto-resume.md` cover this project's remaining ground — respectively, everything that differs from the original C project this was ported from, and the practical workflow for saving/resuming/inspecting trained weights across sessions.
