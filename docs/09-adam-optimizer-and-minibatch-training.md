# The Adam optimizer and minibatch training

`08-ppo-clipped-objective.md` explained why PPO's clipped objective makes it safe to take several gradient-update passes over one collected batch instead of REINFORCE's single pass (`03-policy-gradients-and-reinforce.md`). This doc explains the two things that actually make that reuse happen: the **Adam** optimizer (`pkg/actorcritic/adam.go`) used instead of REINFORCE's plain SGD, and the **minibatch training loop** (`pkg/ppo/trainer.go`'s `RunEpoch`) that decides how a batch gets reused.

## Plain SGD's blind spot: one learning rate for every parameter

Recall `03`'s update rule: `param -= learningRate/sampleCount * gradient`, using the exact same `learningRate` for every single weight and bias in the network. That's simple, but different parameters can have very differently-scaled gradients — some might consistently need small nudges, others larger ones — and a single shared rate can't be "right" for all of them at once. With REINFORCE's one-step-per-batch training, this is a survivable limitation; but PPO applies many more update steps per batch of collected data (`PPOEpochs` times `sampleCount/MinibatchSize` steps, below), so a poorly-scaled step size has more opportunities to cause trouble before the next batch of fresh data arrives.

## Adam: an adaptive step size, tracked per parameter

**Adam** ("Adaptive Moment Estimation", from Kingma & Ba's 2014 paper) fixes this by tracking two running statistics *for every individual parameter*, built purely from that parameter's own gradient history:

```
m[t] = beta1*m[t-1] + (1-beta1)*gradient
v[t] = beta2*v[t-1] + (1-beta2)*gradient^2
```

- `m` (the **first moment**) is a smoothed running average of the gradient itself — similar in spirit to momentum, damping out noisy back-and-forth gradients so the optimizer doesn't overreact to any single batch's fluctuation.
- `v` (the **second moment**) is a smoothed running average of the *squared* gradient — squaring throws away the sign, so this tracks the typical *magnitude* of recent gradients for this specific parameter, regardless of direction.

The actual update then divides by the square root of `v`:

```
param -= learningRate * m_hat / (sqrt(v_hat) + epsilon)
```

A parameter whose gradients have recently been large gets a *smaller* effective step (dividing by a bigger `sqrt(v)`); a parameter whose gradients have been small and consistent gets a comparatively *bigger* effective step. This per-parameter self-adjustment is what "adaptive" means, and it's the main practical advantage Adam has over plain SGD. `epsilon` (`adamEpsilon` in `pkg/actorcritic/adam.go`) exists purely to prevent dividing by a `sqrt(v)` that's still extremely close to zero — see `04-numerical-stability-notes.md`'s "Adam's division-by-near-zero guard" section.

## Bias correction: why `m_hat`/`v_hat`, not `m`/`v` directly

Both `m` and `v` start at exactly zero and are built as weighted averages that lean heavily on that initial zero for the first several updates — which means, early in training, they systematically *underestimate* the true gradient statistics. Adam corrects for this with a **bias correction** that divides by how much of that initial zero-weighting is still "left over" after `t` steps:

```
m_hat = m[t] / (1 - beta1^t)
v_hat = v[t] / (1 - beta2^t)
```

At `t = 1`, `beta1^t` and `beta2^t` are still close to `beta1`/`beta2` themselves, so the correction is large; as `t` grows, `beta1^t`/`beta2^t` shrink toward zero and the correction fades toward doing nothing — exactly matching how the "leftover" influence of that initial zero genuinely fades as more real gradients accumulate. See `Adam.Step`'s `beta1Correction`/`beta2Correction` in `pkg/actorcritic/adam.go`. This project uses the original paper's default decay rates, `beta1 = 0.9` and `beta2 = 0.999` (`adamBeta1`/`adamBeta2`).

## Minibatches and multiple passes per batch

`pkg/reinforce.Trainer.RunEpoch` (`03`) computes exactly one gradient from an entire collected batch and applies exactly one SGD step. `pkg/ppo.Trainer.RunEpoch` (`pkg/ppo/trainer.go`) does considerably more with the batch it collects:

1. **Flatten**: every step of every collected trajectory is gathered into one large pool (`flattenSteps`), rather than staying grouped by trajectory.
2. **Normalize advantages**: the GAE-derived advantages (`07`) across the *entire* pool are rescaled to batch-wide mean 0, standard deviation 1 (`normalizeAdvantages`) — the same batch-statistics idea as `03`'s baseline, applied on top of GAE rather than replacing it, to keep a consistent update scale from batch to batch.
3. **Repeat `PPOEpochs` times**: shuffle the whole pool into a new random order, walk through it in chunks of `MinibatchSize`, and for each minibatch: zero every parameter's gradient, replay each step in the minibatch through the shared `TrainingNetwork` (accumulating gradients — see `02-autograd-and-backpropagation.md`), then take one `Adam.Step` averaged over that minibatch's size.

This *is* the "reuse one batch for several passes" scenario `08` warned is unsafe for a naive policy-gradient loss — `PPOEpochs` full passes over the same, increasingly-stale batch, each broken into freshly-shuffled minibatches. It only remains a net improvement, rather than a source of instability, because PPO's clipped objective (`08`) bounds how far any single action's contribution can push the policy once the ratio has moved too far from the data-collecting policy — Adam's adaptive step size and REINFORCE's `03`-shared batch-baseline idea in step 2 help this converge smoothly, but the safety property that makes repeated reuse viable at all comes from the clipping itself, not from the optimizer.

## Where to go next

`10-live-inference-and-action-masking.md` moves from *training* a network to actually *using* one to make decisions: a single `Actor.Act` entry point, and how to restrict which actions are legal to sample.
