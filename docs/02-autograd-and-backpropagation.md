# Autograd and backpropagation

`pkg/autograd` computes **gradients** automatically. This doc explains what a gradient is, why we need one for every parameter in the network, and how `pkg/autograd` computes them without you (or the network's author) having to work out any calculus by hand.

## The problem: how do you know which way to nudge a weight?

Training a network means repeatedly asking: "for each of the network's thousands of weights, should I increase it a little, decrease it a little, or leave it alone, to make the network's output a little better?" The **gradient** of a value with respect to a weight answers exactly this: it's a number saying "if you increase this weight by a tiny amount, the output changes by (gradient × that tiny amount)." A positive gradient means increasing the weight increases the output; a negative gradient means the opposite.

Once you know the gradient of the network's "how badly is it currently doing" measurement (the **loss**) with respect to every weight, you know exactly which direction to nudge every weight to make the loss a little smaller. That's the entire idea behind gradient-based training — see `03-policy-gradients-and-reinforce.md` for what the loss is in this project specifically.

## Computing gradients automatically: the chain rule

Calculus gives us the **chain rule**: if `y` is computed from `x` through a sequence of simple steps, the gradient of `y` with respect to `x` is the *product* of the gradients of each individual step. Concretely, if `y = f(g(x))`:

```
dy/dx = dy/dg * dg/dx
```

This is powerful because it means we never need to differentiate the whole network at once — we only need to know how to differentiate each small building-block operation (matrix multiply, add, ReLU, softmax, ...) with respect to *its own* inputs, and then chain those together.

## Computational graphs

`pkg/autograd` represents the network as a **computational graph**: a directed graph of `Var` nodes (see `pkg/autograd/var.go`), where each non-leaf `Var` was computed by applying some `Op` (ReLU, Softmax, Add, Sub, Mul, Min, Max, Clamp, Neg, Exp, Log, MatMul, or the REINFORCE loss — see `pkg/autograd/ops.go`) to one or two other `Var`s.

- **Leaf** `Var`s have no `Op` — they're either network inputs (the state vector) or a network's raw parameters (weights/biases).
- Every other `Var` was computed by some `Op` from its `Inputs`.

`BuildGraph` (in `pkg/autograd/graph.go`) walks this graph starting from some final `Var` (e.g. the loss) and produces a list of every reachable `Var`, ordered so that every `Var`'s inputs appear *before* it in the list — a **topological order**. This ordering is what makes both passes below possible with a single, simple left-to-right (or right-to-left) walk.

### Graphs with more than one output

Some networks need to compute more than one final value from a single forward pass — for example, an actor-critic network (see `07-actor-critic-and-generalized-advantage-estimation.md`) produces both an action distribution *and* a value estimate from the same shared trunk. `BuildGraphMulti` generalizes `BuildGraph` to accept several root `Var`s at once (`BuildGraph` is now just `BuildGraphMulti` called with one root): it still produces a single, valid topological order, and any `Var`s the roots happen to share (like that common trunk) are visited and included only once rather than being computed twice. This only supports `Forward` (computing values), not `Backward`, for the same reason `Backward` only ever differentiates a single scalar quantity: `pkg/actorcritic.InferenceNetwork` (forward-only) uses `BuildGraphMulti`, while an actual training loss (which does need `Backward`) is always built as one combined `Var` first.

## The forward pass: computing values

`Graph.Forward` walks the topologically-sorted list front-to-back, calling each `Var`'s `Op.Forward` to compute its value from its already-computed inputs. By the time you reach the end of the list, every `Var`'s `Val` (including the final loss) has been computed. This is exactly the "layers, matrix multiply, activation functions" computation described in `01-neural-networks-and-forward-pass.md`, just executed generically through the graph instead of being hardcoded.

## The backward pass: computing gradients

`Graph.Backward` walks the *same* list back-to-front. It starts by setting the final `Var`'s gradient to 1 (the gradient of anything with respect to itself is 1), then for every `Var` (in reverse order), calls its `Op.Backward`, which:

1. Reads the `Var`'s own gradient (`Grad`) — how much the final loss changes per unit change in *this* `Var`.
2. Uses the chain rule to compute how much the loss changes per unit change in each of this `Var`'s *inputs*, and accumulates ("adds into") each input's `Grad`.

Because the list is processed in reverse topological order, by the time we reach any `Var`, every downstream `Var` that used it as an input has already contributed its share of the gradient — so `Grad` is guaranteed to be complete before it's read.

### Why gradients into a `Var` are *added*, not overwritten

A `Var` can be used as an input to more than one downstream operation (for example, a bias vector `b0` is used by exactly one `Add`, but a weight matrix could in principle feed multiple paths). The chain rule says the total gradient is the *sum* of the gradient contributed through every path. That's why every `Op.Backward` in `pkg/autograd/ops.go` (e.g. `addOp.Backward`) uses `+=`-style accumulation (`Grad.Add(...)`) rather than assignment.

### Worked example: softmax's backward pass

`softmax_add_grad`/`SoftmaxAddGrad` (in `pkg/mat/mat.go`) computes:

```
input.Grad[i] += softmaxOut[i] * (grad[i] - dot(grad, softmaxOut))
```

This looks more complex than ReLU's or Add's backward rules because softmax's output depends on the *sum* of all its inputs (through the normalization step), so changing one input slightly affects every output, not just the corresponding one. Working through the calculus, the gradient of softmax's output with respect to its input turns out to have exactly this "each output's own gradient contribution, minus how much it participates in the shared normalization" shape. The `dot(grad, softmaxOut)` term is exactly that shared correction.

You don't need to re-derive this to use it — the point of `pkg/autograd` is that this derivation only needs to happen once per operation (in `pkg/mat`/`pkg/autograd/ops.go`), and then any graph built out of these operations gets correct gradients "for free," no matter how the operations are combined.

## Elementwise ops beyond REINFORCE

Add, Sub, Mul, Min, Max, Neg, Exp, and Log are all elementwise: each output element depends only on the corresponding input element(s), which is what keeps their `Backward` rules simple compared to softmax's (no cross-element interaction). Mul, Min, Max, Neg, Exp, and Log exist specifically so that objectives more complex than REINFORCE's `-log(p) * advantage` can be composed from existing graph nodes rather than requiring a new hand-written `Op` per objective — for example, a clipped probability-ratio objective needs `exp(logNew - logOld)` and a `min(...)` between two candidate surrogate values, both expressible with these primitives (`Clamp`, used by that same objective to bound the ratio into a fixed range, is a small helper built from `Max` and `Min` rather than a further `Op` of its own — see `pkg/autograd/ops.go`). `TestGradientCheckExpOfSubComposition` in `pkg/autograd/gradcheck_test.go` demonstrates exactly this kind of composition (`Exp` of a `Sub`).

## Validating this actually works: gradient checking

Because a bug in a hand-written `Backward` implementation can be subtle (right shape, wrong formula), this project validates every operation with **gradient checking** — see `pkg/autograd/gradcheck_test.go`. The idea: you don't need calculus to *estimate* a gradient — you can approximate it directly from the definition of a derivative, by nudging an input by a tiny amount `ε` and seeing how much the output changes:

```
numerical_gradient ≈ (f(x + ε) - f(x - ε)) / (2 * ε)
```

This "numerical gradient" is slow (it requires re-running the entire forward pass once per input value) and only approximate, but it doesn't depend on the `Backward` implementation being correct at all — it's computed purely from repeated calls to `Forward`. By comparing it against the analytically-computed gradient from `Backward`, a mismatch reliably flags a bug in the backward-pass math. This is a standard, widely-used technique for validating any autograd or backpropagation implementation, not something specific to this project.

## Where to go next

`03-policy-gradients-and-reinforce.md` explains what loss this project actually computes and why — the REINFORCE algorithm — and how gradients flowing back through this graph translate into "make the policy more likely to repeat good decisions." `07-actor-critic-and-generalized-advantage-estimation.md` and `08-ppo-clipped-objective.md` explain a more elaborate loss, built from several of these same primitives (`Min`, `Max`, `Clamp`, `Exp`, `Log`), composed together on top of a `BuildGraphMulti` network.
