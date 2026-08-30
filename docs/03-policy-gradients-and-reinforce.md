# Policy gradients and REINFORCE

This doc explains reinforcement learning (RL) from scratch, and specifically the **REINFORCE** algorithm implemented in `pkg/reinforce`, which is what actually trains the policy network described in `01-neural-networks-and-forward-pass.md`.

## What reinforcement learning is

In supervised learning, you train on examples that already have a "correct answer" attached. In reinforcement learning, there's no correct-answer dataset — instead, an **agent** takes **actions** in an **environment**, and the environment responds with a **reward** (a number saying how good or bad that was) and a new **state**. The agent's job is to learn, purely from this trial-and-error feedback, which actions tend to lead to good outcomes.

In this project:
- The **environment** is `pkg/snakeenv`: a small grid where the agent tries to reach a food cell and avoid the grid boundary (see `pkg/snakeenv/action.go`'s package comment for exactly what this environment does and doesn't simulate — despite the name, there's no snake body or self-collision).
- The **agent** is the policy network from `pkg/policy`.
- An **episode** (or **trajectory**, see `pkg/reinforce/trajectory.go`) is one run from a reset state until the agent leaves the grid or a step limit is reached.

## The policy: choosing actions from probabilities

A **policy** is just a rule for choosing actions given a state. This project uses a **stochastic policy**: rather than always picking the single best-looking action, the network outputs a *probability* for every action (via softmax — see `01-neural-networks-and-forward-pass.md`), and an action is *sampled* from that distribution (`SampleAction` in `pkg/reinforce/trajectory.go`).

This matters for **exploration**: early in training, the policy doesn't know which actions are good, so sampling (rather than always taking the current best guess) lets the agent occasionally try different things and discover better strategies. As training progresses and the policy gets more confident, its probability distribution naturally becomes more peaked around the actions it has learned are good.

## Reward-to-go: crediting a whole episode's future, not just one step

A single action's *immediate* reward is often a poor measure of how good that action actually was — moving toward food might not pay off (get a +20 reward) until several steps later. To account for this, every step in a trajectory is credited with its **return** (also called **reward-to-go**): the sum of *all* rewards from that step until the end of the episode, with rewards further in the future discounted by a factor `gamma` (0 < gamma <= 1) per step:

```
Return[t] = Reward[t] + gamma * Reward[t+1] + gamma^2 * Reward[t+2] + ...
```

`computeReturns` in `pkg/reinforce/trajectory.go` computes this efficiently by working *backward* through the episode: `Return[t] = Reward[t] + gamma * Return[t+1]`, reusing the already-computed future return instead of re-summing everything.

**Why discount at all?** Discounting (`gamma < 1`) makes near-term rewards count more than distant ones. This is partly a modeling choice (immediate consequences are often more reliable signals than consequences many steps away) and partly a numerical convenience (it keeps the sum finite even in principle-infinite environments, and it makes the appropriately-weighted algorithm converge better in practice). This project uses `gamma = 0.99` by default (see `configs/config.json`).

## The REINFORCE loss: turning "was this good?" into a training signal

The central idea of REINFORCE (one of the original policy-gradient algorithms) is: **increase the probability of actions that led to good outcomes, and decrease the probability of actions that led to bad outcomes** — where "outcome" is measured by that step's return (or, as below, its *advantage*).

Concretely, the loss for one (state, action) pair with return-derived advantage `A` is:

```
loss = -log(prob_of_action_taken) * A
```

implemented in `ReinforceLoss` in `pkg/mat/mat.go`. Let's build intuition for why this specific formula works:

- If `A` is positive (the action led to a better-than-average outcome), minimizing this loss means minimizing `-log(prob) * A`, i.e. *maximizing* `log(prob) * A`, i.e. pushing `prob` (the probability the policy assigned to the action actually taken) **up**.
- If `A` is negative (a worse-than-average outcome), the same minimization pushes `prob` **down**.
- The `log` isn't arbitrary: differentiating `log(prob)` is what produces the clean gradient formula in `ReinforceAddGrad`, and — more importantly — it means a policy that's already very confident about an action needs a much bigger raw score change to move its probability further (since probabilities are bounded in `[0, 1]` but log-probabilities are unbounded below), which keeps training stable.

This is why the loss only involves the probability of the action *actually taken* — REINFORCE never needs to ask "what would have happened with a different action?", only "how should I adjust my confidence in the action I actually sampled, given how it turned out?"

## The baseline: why raw returns aren't used directly

Using the raw return `G` as `A` directly works in principle, but suffers from very high variance: if all rewards in an environment happen to be positive, *every* action's probability gets pushed up (just by different amounts), even actions that were actually below-average for that batch. This makes learning noisy and slow.

The fix used here is a **baseline**: subtract something representative of "what return did we typically get in this batch" before using the return as the training signal. This project uses a **batch-wide mean/std baseline** (`returnStatistics` in `pkg/reinforce/returns.go`):

```
advantage = (return - batch_mean) / (batch_std + epsilon)
```

This is often called **advantage normalization** or **whitening**: it rescales returns so that, within a batch, above-average outcomes get a positive advantage, below-average outcomes get a negative advantage, and the typical magnitude is roughly consistent from batch to batch (which also makes a single learning rate work reasonably well throughout training, even as raw reward magnitudes change).

**This is not the same as actor-critic.** A common, more sophisticated variant of this idea uses a second, separately-trained neural network (a "critic" or "value function") to predict an *expected* return for a given state, and uses `return - predicted_value` as the advantage. This project's baseline is much simpler: a single mean/std computed once per batch, with no separate network. (The original C project's function was named `create_actor_model`, which suggests an actor-critic architecture; this Go port keeps the same simple batch-baseline algorithm but names things to avoid implying a critic network that doesn't exist — see `docs/05-porting-notes.md`.)

## Putting it together: one training epoch

`Trainer.RunEpoch` in `pkg/reinforce/trainer.go` runs, per epoch:

1. **Rollout collection**: run the current policy for `RolloutSize` independent episodes (in parallel — see `docs/05-porting-notes.md` for the concurrency design), recording every (state, action, reward) along the way.
2. **Return computation**: compute each step's discounted return, then the batch-wide mean/std baseline across every step of every trajectory.
3. **Gradient accumulation**: replay every stored step through the network (forward pass, to reconstruct the same probabilities the policy assigned at the time), compute the REINFORCE loss and its gradient using that step's advantage, and *accumulate* (not overwrite) the resulting parameter gradients — see `02-autograd-and-backpropagation.md` for why gradients accumulate this way, and `policy.TrainingNetwork.ZeroGrad`/`ApplyGradientStep` for where the batch's gradients are reset and eventually applied.
4. **One gradient-descent step**: scale the accumulated gradient by `learning_rate / sample_count` (averaging the gradient over every individual step seen this epoch) and subtract it from every parameter — plain, unadorned **stochastic gradient descent (SGD)**, with no momentum or per-parameter adaptive learning rates (unlike, e.g., Adam).

Because REINFORCE's gradient estimate is noisy (it only ever sees a finite, sampled batch of trajectories, and a single trajectory's outcome depends on a long chain of sampled actions), training tends to be a slow, noisy climb rather than a smooth one — expect the average return to fluctuate a lot epoch-to-epoch even while trending upward over many epochs. `pkg/reinforce/trainer_test.go`'s `TestTrainingImprovesAverageReturn` demonstrates this upward trend directly.

## Where to go next

`04-numerical-stability-notes.md` covers the specific numerical safeguards (clamping, the softmax max-subtraction trick, the variance clamp) that keep this whole pipeline from producing `NaN`/`Inf` in practice.

`07-actor-critic-and-generalized-advantage-estimation.md` picks up the batch-wide baseline idea from this doc and replaces it with a more sophisticated, learned alternative: a second network predicting expected return, combined with real rewards via Generalized Advantage Estimation. This is a genuinely different, more involved technique from the simple mean/std baseline above — the "This is not the same as actor-critic" distinction earlier in this doc still describes exactly what `pkg/reinforce` does today; `07` is where actor-critic is actually implemented, in a separate package.
