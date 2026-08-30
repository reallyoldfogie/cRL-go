# Actor-critic networks and Generalized Advantage Estimation

`03-policy-gradients-and-reinforce.md` explained REINFORCE's batch-wide mean/std baseline: a simple way to turn a raw return into a training signal without a second network. This doc explains a more sophisticated alternative built for this project's PPO trainer: predicting expected return with a second, learned **value function**, and combining that prediction with real observed rewards via **Generalized Advantage Estimation (GAE)**. No prior ML background assumed beyond `01`-`03`.

## The problem with waiting for the whole episode

REINFORCE's return-to-go (`Return[t] = Reward[t] + gamma*Reward[t+1] + ...`, see `03`) needs every reward from step `t` until the episode ends before it can be computed. This has two costs: you can't get *any* training signal for step `t` until the episode is over, and the return is a sum of many random rewards, so it's noisy — two episodes that made the exact same decision at step `t` can end up with very different returns just from what happened to occur afterward. That noise is exactly why `03` needed a batch-wide baseline to begin with.

## A value function: predicting the return before the episode ends

A **value function**, written `V(state)`, is a prediction of "how much discounted reward do I expect to collect from this state onward, if I keep following my current policy?" If you had a perfectly accurate value function, you'd already have a low-noise stand-in for the return at every single step, without waiting for the episode to finish.

Of course, `V` starts out untrained and wrong (just like the policy itself starts out untrained and wrong) — but it gets trained alongside the policy, from the same rollouts, and it only has to predict one number (an expected return) rather than choose an action, which turns out to be a comparatively easy prediction to improve quickly.

## TD-error: a one-step estimate of "how much better than expected"

Once you have a value function, you can compute a much shorter-horizon signal called the **temporal-difference error (TD-error)**:

```
delta[t] = Reward[t] + gamma * V(state[t+1]) - V(state[t])
```

Read this as: "the reward I actually just observed, plus my own estimate of everything after it, minus what I predicted *before* taking this step." If the step went better than `V` expected, `delta[t]` is positive; if worse, negative.

This is called **bootstrapping**: instead of summing real rewards all the way to the end of the episode, you use one real reward plus the network's *own guess* about the rest. That guess might be wrong (especially early in training, when `V` hasn't learned much yet), which makes TD-error a **biased** estimate — but it only needs one step of the environment to compute, which makes it far lower variance than a full-episode return.

## Actor-critic: one network, two jobs

To have a value function at all, something has to compute it. `pkg/actorcritic` builds a network with the exact same shared trunk as `pkg/policy`'s (`input -> W0,B0 -> ReLU -> W1,B1 -> ReLU`, see `01-neural-networks-and-forward-pass.md`), but feeds that trunk's output into *two* independent heads instead of one (see `buildForward` in `pkg/actorcritic/network.go`):

```
                          -> Wpi,Bpi -> Softmax -> policy output (the "actor")
input -> W0,B0 -> ReLU -> W1,B1 -> ReLU
                          -> Wv,Bv  -> value output (the "critic")
```

- The **actor** head (`Wpi,Bpi`) is identical in shape and purpose to `pkg/policy`'s only head: a softmax distribution over actions.
- The **critic** head (`Wv,Bv`) is a plain linear layer with a single output number: the value estimate `V(state)` used above. There's no softmax here — a value estimate isn't a probability, it can be any real number.

Both heads read from the *same* trunk output, so most of the network's "understanding" of the state is shared between choosing actions and predicting returns. `pkg/actorcritic.InferenceNetwork` computes both outputs in a single `Graph.Forward()` call (via `autograd.BuildGraphMulti` — see `02-autograd-and-backpropagation.md`) rather than running the shared trunk twice.

This is precisely the architecture `05-porting-notes.md`'s "Naming: no implied critic network" section explains `pkg/policy` deliberately avoids implying by name — `pkg/actorcritic` is where that architecture now actually lives, as a separate package, leaving `pkg/policy`/`pkg/reinforce` unchanged.

## GAE(λ): a dial between low variance and low bias

TD-error (above) and the full-episode return (`03`) sit at two extremes:

- TD-error (equivalent to setting a parameter `λ = 0`, below) is **low variance** (it only depends on one real reward) but **biased** (it trusts a possibly-inaccurate value estimate for everything beyond that one step).
- The full return (`λ = 1`) is **unbiased** (it's built entirely from real, observed rewards) but **high variance** (a long sum of many random rewards).

**Generalized Advantage Estimation** (`computeGAE` in `pkg/ppo/gae.go`) computes something in between, controlled by a tunable parameter `λ` (0 ≤ λ ≤ 1), working backward through a trajectory just like `03`'s `computeReturns` does:

```
delta[t]      = Reward[t] + gamma*V(state[t+1]) - V(state[t])
advantage[t]  = delta[t] + gamma*lambda*advantage[t+1]
return[t]     = advantage[t] + V(state[t])
```

`V(state[t+1])` is treated as `0` at the trajectory's last step (there's nothing further to bootstrap from). Setting `lambda = 0` collapses `advantage[t]` to just `delta[t]` (pure TD-error); setting `lambda = 1` makes it algebraically identical to `03`'s plain discounted reward-to-go, just computed as a sum of TD-errors instead of a sum of raw rewards — `TestComputeGAEWithNoDiscountingReducesToPlainReturns` in `pkg/ppo/gae_test.go` checks this identity directly. Values in between smoothly trade some bias for less variance. This project's PPO trainer uses `lambda = 0.95` by default (`GAELambda` in `configs/config.json`), a common empirical middle ground.

`return[t]` above (advantage plus value estimate) becomes the **target** the critic head is trained to predict more accurately over time — the same rollout that trains the actor to pick better actions also trains the critic to estimate returns more accurately, which is why it's called *actor-critic*.

## Where to go next

`08-ppo-clipped-objective.md` explains exactly how this GAE-derived advantage is used inside PPO's loss function, and why a naive policy-gradient update (even one using this improved advantage) isn't safe to reuse across multiple training passes on the same batch of data.
