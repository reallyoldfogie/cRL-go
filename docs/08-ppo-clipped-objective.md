# PPO: the clipped-surrogate objective

`07-actor-critic-and-generalized-advantage-estimation.md` explained how this project computes a lower-variance advantage signal. This doc explains **Proximal Policy Optimization (PPO)**, the algorithm `pkg/ppo` uses to turn that advantage into an actual weight update — specifically, why it needs a more careful loss than REINFORCE's `-log(prob) * advantage` (see `03-policy-gradients-and-reinforce.md`) once you want to reuse one batch of data for more than a single gradient step.

## Why reuse a batch at all?

REINFORCE (`03`) collects a batch of rollouts, computes one gradient from them, applies one SGD step, and throws the batch away. That's simple, but it means every batch of (often expensive) environment interaction only ever contributes to a single, small step of learning.

`09-adam-optimizer-and-minibatch-training.md` explains a more data-efficient alternative: split the batch into shuffled minibatches and take *several* gradient-update passes over it (`PPOEpochs` in this project's config) before collecting a new batch. This doc explains why that reuse isn't safe with a naive policy-gradient loss, and what PPO does about it.

## The problem: your policy changes underneath the data

REINFORCE's loss assumes the probability you're pushing up or down (`prob_of_action_taken`) was produced by the *current* policy. But if you take several gradient steps against the same batch, the policy is a slightly different (hopefully better) policy after every step — while the batch's rewards/advantages were all collected by the *original* policy that existed before any of those steps happened.

If you keep computing `-log(new_prob) * advantage` against a policy that has already moved away from the one that generated the data, the update can overcorrect: a single unlucky minibatch can push the policy so far that it gets dramatically worse — and, because you're about to take more passes over the *same* stale batch, there's nothing to immediately correct that mistake. In the worst case this snowballs into **policy collapse**, where a network that was learning well suddenly starts performing far worse and struggles to recover. PPO's whole design is aimed at preventing this while still allowing multiple passes over one batch.

## Measuring how far the policy has moved: the probability ratio

PPO doesn't compare the new policy to the old one directly by subtracting probabilities. Instead, it computes a **ratio**:

```
ratio = new_prob_of_action / old_prob_of_action
```

- `ratio == 1` means the new policy assigns the exact same probability to this action as the policy that collected the data did — no change yet.
- `ratio > 1` means the new policy has become *more* confident in this action than the data-collecting policy was.
- `ratio < 1` means *less* confident.

This project computes it in log-space for numerical convenience (`buildPolicyLoss` in `pkg/ppo/loss.go`): `ratio = exp(new_log_prob - old_log_prob)`, which is mathematically the same ratio, just computed as a difference-then-exponentiate rather than a division (`old_log_prob` is recorded once, at rollout-collection time, in `Rollout.LogProbs` — see `pkg/ppo/rollout.go`). This general technique — reweighting a signal by how much more or less likely an outcome has become under a new distribution — is called **importance sampling**, and shows up throughout RL and statistics whenever you want to reuse data collected under one distribution to reason about a different one.

## The clipping trick: capping how far you'll trust the ratio

The core PPO idea is: compute the usual policy-gradient surrogate using this ratio (`ratio * advantage` instead of REINFORCE's `-log(prob) * advantage`), but **cap how much credit or blame a single update is allowed to assign once the ratio has moved too far from 1**. Concretely (`buildPolicyLoss`):

```
surrogate1 = ratio * advantage
surrogate2 = clip(ratio, 1-eps, 1+eps) * advantage
policyLoss = -min(surrogate1, surrogate2)
```

`clip(x, lo, hi)` (`autograd.Clamp`, see `02-autograd-and-backpropagation.md`) just forces `x` into the range `[lo, hi]`, leaving it unchanged if it's already inside. `eps` (`ClipEpsilon`, default `0.2`) sets how far the ratio is allowed to move — `20%` in either direction — before clipping kicks in.

Why `min` of the two surrogates, rather than just always using the clipped one? Work through the two cases:

- **Advantage is positive** (the action was good): the unclipped surrogate keeps growing the more confident the new policy becomes (`ratio` grows), which is exactly the runaway-overcorrection risk described above. Taking the `min` means once `ratio` exceeds `1+eps`, the *clipped* surrogate (a flat, capped value) is smaller, and it's used instead — so the loss stops rewarding *even more* confidence beyond that point. The gradient there becomes zero: this update no longer pushes the policy any further in that direction.
- **Advantage is negative** (the action was bad): symmetric reasoning applies once `ratio` drops below `1-eps`.

The net effect: as long as the new policy stays within `[1-eps, 1+eps]` of the data-collecting policy for a given action, training proceeds normally; once it drifts further, that action's contribution to the loss flattens out, removing the incentive to keep moving further away in the same update. That's what makes reusing one batch for several passes safe enough in practice — each pass can still refine the policy, but no single action's update can run away unchecked.

## The entropy bonus: a standing incentive to keep exploring

`buildLoss` (`pkg/ppo/loss.go`) subtracts one more term from the loss: `EntropyCoef` times the policy's **entropy**, `sum_a(-prob[a] * log(prob[a]))`. Entropy is a standard measure of how "spread out" a probability distribution is — it's highest when every action is equally likely, and lowest (zero) when the policy is completely certain of one action.

Subtracting entropy from the loss (equivalently, *rewarding* higher entropy) works against a policy collapsing into total certainty too early, before it's actually explored enough to know that certainty is warranted — the same exploration concern `03` raised about why this project samples actions rather than always taking the current best guess.

## Putting it together

The final loss `buildLoss` builds is `policyLoss + valueLoss - entropyBonus` — the clipped-surrogate policy loss above, plus a squared-error loss training the critic head toward `07`'s GAE-derived return targets (`buildValueLoss`), minus the entropy bonus. `pkg/ppo/network.go`'s `TrainingNetwork` wires all of this into one computation graph per training step, driven by overwriting a handful of placeholder `Var`s (`SetStep`) rather than rebuilding the graph every time.

## Where to go next

`09-adam-optimizer-and-minibatch-training.md` explains the optimizer and multi-pass training loop this clipping makes safe to use, and contrasts it with REINFORCE's single-pass-per-batch SGD update from `03`.
