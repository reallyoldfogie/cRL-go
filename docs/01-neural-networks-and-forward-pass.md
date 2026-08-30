# Neural networks and the forward pass

This doc explains, from first principles, what the policy network in `pkg/policy` actually is and why it's built the way it is. No prior ML background assumed.

## What a neural network is

At its core, a neural network is just a function that turns numbers in (an "input vector") into numbers out (an "output vector"), where the function has a bunch of tunable numeric knobs called **parameters** (or **weights** and **biases**). "Training" a network means automatically adjusting those knobs so the function produces more useful outputs.

The network in this project (`pkg/policy`) takes a description of the current game state (where the agent is, where the food is, which way it's facing) and outputs 5 numbers: one per possible action, representing how good the network currently thinks each action is.

## Layers: matrix multiply, then add a bias

The simplest building block is a **linear layer**. If `x` is the input vector, a linear layer computes:

```
z = W * x + b
```

- `W` (the **weight matrix**) and `b` (the **bias vector**) are the tunable parameters.
- `W * x` is a matrix-vector multiplication (see `pkg/mat`'s `MatMul`): every output number is a weighted sum of every input number, and the weights are exactly the tunable knobs.
- `b` lets each output shift up or down independently of the input, which a pure weighted sum alone can't do.

This project stacks three of these linear layers:

```
input (state vector) -> W0*x+b0 -> ReLU -> W1*x+b1 -> ReLU -> W2*x+b2 -> Softmax -> output (action probabilities)
```

See `buildForward` in `pkg/policy/network.go` for exactly this chain of operations.

## Why you need more than one linear layer: activation functions

If you stacked linear layers with nothing in between, the whole stack would collapse into a *single* linear layer no matter how many you chained (matrix multiplication composes into another matrix multiplication). To let the network represent more interesting, non-linear relationships, you insert a non-linear **activation function** between layers.

### ReLU

This project uses ReLU ("Rectified Linear Unit"), implemented in `pkg/mat`'s `ReLU`:

```
ReLU(x) = max(0, x)
```

That's it — every negative number becomes zero, every non-negative number passes through unchanged. It's a strange-looking function to reach for, but it's cheap to compute, cheap to differentiate (see `02-autograd-and-backpropagation.md`), and empirically works very well in practice for making networks trainable.

### Softmax: turning numbers into a probability distribution

The final layer's raw output (`z2b` in `buildForward`) is just five arbitrary numbers — they could be negative, huge, tiny, anything. But we want to interpret them as "how likely is each action," which means we need five numbers that are all non-negative and sum to 1 (a **probability distribution**). That conversion is what `Softmax` (in `pkg/mat`) does:

```
softmax(x)_i = exp(x_i) / sum_j(exp(x_j))
```

Intuitively: exponentiate every number (which makes everything positive and exaggerates differences — a slightly bigger input becomes a *much* bigger output), then divide by the total so everything sums to 1. The action with the largest raw score ends up with the highest probability, but every action keeps *some* non-zero probability, which is what lets the agent still occasionally explore actions it doesn't currently favor.

See `docs/04-numerical-stability-notes.md` for why the actual implementation subtracts the maximum value before exponentiating.

## Why the weights start out random (and specifically, Xavier/Glorot-random)

`NewParams` in `pkg/policy/params.go` initializes every weight matrix to small random values before training starts, rather than, say, all zeros.

**Why not all zeros?** If every weight started at the same value (e.g. 0), every neuron in a layer would compute the exact same output and receive the exact same gradient update during training — they'd stay identical forever, and the network would never be able to represent more than one “kind” of neuron per layer. Randomness breaks that symmetry so different neurons can specialize.

**Why not just any random values?** If the random weights are too large, the values flowing through the network can blow up exponentially as they pass through each layer ("exploding activations"); too small, and they shrink toward zero ("vanishing activations"). Either way, training becomes unstable or painfully slow.

**Xavier/Glorot initialization** (named after Xavier Glorot, who proposed it) is a specific recipe for picking a "just right" random range based on a layer's shape. This project draws every weight uniformly from `[-bound, bound]`, where:

```
bound = sqrt(6 / (fan_in + fan_out))
```

`fan_in` is the number of inputs to the layer and `fan_out` is the number of outputs. Wider layers get proportionally smaller initial weights, which is precisely what's needed to keep the *typical size* of values flowing forward (and gradients flowing backward) roughly stable from layer to layer, rather than growing or shrinking as they pass through more layers. See `xavierBound` in `pkg/policy/params.go`.

Biases start at exactly zero — there's no symmetry-breaking concern for biases (they don't multiply against the input), so zero is a fine default that networks quickly adjust away from as needed.

## Where to go next

- `02-autograd-and-backpropagation.md` explains how the network's parameters actually get updated: computing gradients via automatic differentiation.
- `03-policy-gradients-and-reinforce.md` explains what this particular network is being trained *to do* (choose good actions) and the algorithm used to train it.
- `07-actor-critic-and-generalized-advantage-estimation.md` explains `pkg/actorcritic`, which reuses this exact shared-trunk architecture but adds a second linear output head predicting expected return alongside the action distribution.
