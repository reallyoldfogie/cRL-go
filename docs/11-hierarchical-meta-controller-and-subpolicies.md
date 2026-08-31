# Hierarchical RL: meta-controllers and sub-policies

`07`-`09` covered how a single actor-critic network is trained with PPO. This doc covers a different way of *structuring* a policy in the first place: instead of one flat network choosing directly among every primitive action, a two-level hierarchy — a **meta-controller** periodically picking a coarse **subgoal**, and a dedicated **sub-policy** per subgoal actually choosing primitive actions while it's active (`pkg/hierarchical`).

## The problem: one flat policy juggling competing goals

`pkg/hierarchicalgridworld` (the toy environment this doc validates against) gives an agent four things to weigh against each other: collect a resource, build at a fixed target (which needs a held resource), avoid a fixed hazard cell, and fight or avoid a roaming mob. A single flat network has to encode "should I collect, build, flee, or fight *right now*" as one probability distribution over every primitive action, every single step, no matter how different those situations are. That's a lot to ask of one set of weights.

## The idea: a manager and specialists

Split the decision into two levels. A **meta-controller** looks at the situation only occasionally and picks a coarse **subgoal** — "focus on building for a while." A **sub-policy** dedicated to that subgoal then picks the actual primitive actions (which cell to move to, when to collect) for as long as that subgoal stays active, without also having to weigh whether it should be doing something else entirely. This is the same idea as a manager delegating "go handle the build" to a specialist, rather than deciding every individual footstep themselves while also weighing every other competing priority.

## The key insight: a subgoal choice is just another action choice

The elegant part of this design: from the perspective of "sample a category from a probability distribution, scored with a GAE advantage and the PPO clip objective" (see `07`/`08`), choosing a subgoal (a category among `NumSubgoals` options) is *exactly* the same kind of decision as choosing a primitive action (a category among `env.ActionSpace()` options) — just at a different level of abstraction. So `pkg/hierarchical.Trainer` builds `N+1` completely ordinary `actorcritic.Params`/`ppo.TrainingNetwork`/`actorcritic.Adam` triples — one for the meta-controller, one per subgoal — reusing `07`'s actor-critic architecture and `08`'s PPO loss verbatim, with zero new loss or GAE math written for this package. `Subgoal` (`pkg/hierarchical/subgoal.go`) is just its own small integer-based type, so it's never confused with a primitive `rl.Action` at a call site even though both are plain integers underneath.

## Telling a sub-policy which subgoal is active

Every sub-policy sees the same kind of base observation from the environment, but needs to behave differently depending on which subgoal is currently active. `augmentObservation` (`pkg/hierarchical/augment.go`) appends a one-hot encoding of the active subgoal onto the base observation before feeding it to a sub-policy — the same one-hot idea from `01`'s state vectors, just used to say "you are currently in subgoal 2" rather than "the food is at cell 12." A sub-policy's input size is therefore `env.ObservationSize() + NumSubgoals`; the meta-controller's input size is just the plain `env.ObservationSize()` (it doesn't need to be told which subgoal is active — picking the *next* one is its whole job).

## Collecting one trajectory, two decision rates

`collectHierarchicalTrajectory` (`pkg/hierarchical/rollout.go`) runs one episode, but makes decisions at two different rates: the meta-controller decides again every `Config.SubgoalInterval` environment steps (or immediately if the episode ends first), while the currently-active sub-policy decides every single step. Each "activation interval" produces two things once it closes: one entry in the meta-controller's own trajectory (whose reward is the *sum* of every real environment reward earned during that whole interval, and whose action is which subgoal was chosen), and a perfectly ordinary primitive-action trajectory recorded for whichever sub-policy was active.

## Why a subgoal's segments can't just be concatenated

A subtlety worth understanding: the *same* subgoal can become active more than once in one episode, with a different subgoal active in between. It's tempting to glue all of one subgoal's steps together into one long trajectory for that sub-policy — but `07` explained that GAE(λ) computes each step's advantage by bootstrapping from the value estimate at the *very next* step in the *same* trajectory. Gluing two chronologically-disjoint intervals together would make the last step of the first interval incorrectly bootstrap from the value estimate at the start of a later, unrelated interval. `pkg/hierarchical` avoids this by keeping every activation interval as its own independent `ppo.Rollout` (`HierarchicalRollout.SubRollouts` is a `map[Subgoal][]*ppo.Rollout` — a *list* per subgoal, not one merged rollout), so `ppo.ComputeGAE` always bootstraps from `0` at the true end of each interval, exactly like it already does at the end of an ordinary episode.

## Training: the same recipe, run N+1 times

Per epoch, `Trainer.RunEpoch` (`pkg/hierarchical/trainer.go`) collects a batch of episodes in parallel (mirroring `09`'s trainer), then: pools every meta-transition across the whole batch and runs shuffled-minibatch Adam updates on the meta-controller; and, for each subgoal, pools only the segments where *that* subgoal happened to be active anywhere in the batch, and runs the same shuffled-minibatch Adam recipe on that subgoal's own network. A subgoal that was never chosen this epoch simply isn't trained this epoch — there's nothing collected to learn from.

## A toy environment with genuinely competing objectives

`pkg/hierarchicalgridworld` is what gives this design something worth deciding between — unlike `pkg/snakeenv`/`pkg/gridworldenv`, which each have a single objective. `cmd/train-hierarchical` trains against it, printing the meta-controller's and every subgoal's own update counts each epoch alongside the usual average return; see `README.md`'s "Hierarchical RL quickstart" to run it. It also supports the same `-checkpoint-in`/`-checkpoint-out`/`-checkpoint-dir`/`-checkpoint-interval` flags as `cmd/train-ppo`, saving the meta-controller's and every sub-policy's weights as one atomic checkpoint file per generation (see `pkg/hierarchical.Trainer.Save`/`Load`) rather than N+1 independent files, so a checkpoint can never be resumed from a mix of networks saved at different epochs.

## Where to go next

`12-context-and-long-lived-environments.md` covers a different concern entirely: not how a policy is structured, but how it's run against an environment that behaves less politely than these toy ones — one that can block, time out, or needs to be reused across many episodes instead of rebuilt.
