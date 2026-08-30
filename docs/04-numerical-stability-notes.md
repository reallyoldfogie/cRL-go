# Numerical stability notes

Floating-point numbers can't represent every real number exactly, and operations like `exp` and `log` can blow up or become undefined near their edge cases. This doc collects every deliberate numerical safeguard in this codebase, what would go wrong without it, and why the chosen fix works.

## Softmax: subtracting the max before exponentiating

`Softmax` in `pkg/mat/mat.go` computes, for each element `x_i`:

```go
e := math.Exp(float64(v - maxValue))
```

rather than the textbook `exp(x_i)` directly. Here's why: `float32` can represent values up to roughly `3.4 * 10^38`, but `exp(x)` grows so fast that even a fairly ordinary-looking input like `x = 100` gives `exp(100) ≈ 2.7 * 10^43` — already larger than the largest representable `float32`, which "overflows" to `+Inf`. Once any single value overflows, the following division (`e / sum`) produces `NaN` (`Inf / Inf`), poisoning the entire softmax output and, from there, every downstream computation.

The fix (the "max-subtraction trick") relies on an algebraic identity: subtracting a constant `c` from every input before exponentiating doesn't change the final softmax result at all —

```
exp(x_i - c) / sum_j(exp(x_j - c)) = exp(x_i) / sum_j(exp(x_j))
```

— because the `exp(-c)` factor cancels between the numerator and denominator. So we're free to pick `c = max(x)`, which guarantees the *largest* exponent computed is `exp(0) = 1` (every other exponent is `exp(negative number) <= 1`), completely eliminating the overflow risk while producing the mathematically identical answer. See `TestSoftmaxSumsToOneAndStaysFinite` in `pkg/mat/mat_test.go`, which explicitly tests inputs (like `1000`) that would overflow without this trick.

## Clamping probabilities away from zero before `log`

`ReinforceLoss` and `ReinforceAddGrad` (in `pkg/mat/mat.go`) both compute `max(p, probEpsilon)` (with `probEpsilon = 1e-8`) before using a probability `p`:

```go
clamped := max(p, probEpsilon)
dst.Data[i] = float32(-math.Log(float64(clamped))) * advantages.Data[i]
```

`log(0)` is `-Inf` (`log` of a number approaching zero from above diverges to negative infinity), and `1/0` in the gradient formula (`ReinforceAddGrad` divides by the probability) is similarly undefined. In principle, softmax's output is always strictly positive (since `exp` of any finite number is positive), so a probability should never be *exactly* zero — but `float32` has limited precision, and if the network becomes extremely confident an action is wrong, the true probability can underflow to a `float32` zero even though the mathematically "true" probability was just extremely small. Clamping to a tiny positive floor (`1e-8`) avoids `-Inf`/`NaN` in that edge case while barely affecting the result any time the probability isn't already vanishingly small.

The general-purpose `Log`/`LogAddGrad` ops (also in `pkg/mat/mat.go`, added for objectives beyond REINFORCE's own loss) reuse this exact same `probEpsilon` clamp, so any caller taking the log of a probability-like value gets the same safeguard without re-deriving it.

## Clamping variance to zero before taking a square root

`returnStatistics` in `pkg/reinforce/returns.go` computes variance with the "naive" one-pass formula:

```
variance = E[X^2] - E[X]^2
```

This is mathematically equivalent to the textbook "average squared deviation from the mean" formula, and is convenient because it only requires one pass over the data (accumulating a running sum and a running sum-of-squares) rather than two (one to compute the mean, a second to compute deviations from it). The tradeoff: it's less numerically robust. When the true variance is small relative to the *squares* of the individual values, `E[X^2]` and `E[X]^2` can be very close floating-point numbers, and subtracting two close floating-point numbers amplifies whatever rounding error each of them already had ("catastrophic cancellation"). In the worst case, this can produce a *negative* result for what is mathematically guaranteed to be a non-negative quantity (variance can never be negative).

Since the very next step is `sqrt(variance)` (to get a standard deviation), a negative input would produce `NaN`. The fix is a defensive clamp:

```go
std = float32(math.Sqrt(float64(max(variance, 0))))
```

This preserves the original algorithm's behavior for the (overwhelmingly common) case where the naive formula produces a valid non-negative variance, while guaranteeing a `NaN` never leaks out of this function due to floating-point rounding.

**A more robust alternative exists** (and is *not* used here, to match the original algorithm faithfully): [Welford's online algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm) computes variance incrementally without ever subtracting two large, similar numbers, and is immune to this cancellation problem by construction. If return-statistics numerical stability ever becomes a practical issue (e.g. with much larger batch sizes or more extreme reward magnitudes than this project currently uses), switching `returnStatistics` to Welford's algorithm would be a natural, drop-in improvement.

## Why `float32`, and does it match the original C code?

Every numeric type in `pkg/mat` is `float32`, matching the original C implementation's `f32` (`float`) throughout. Go and C both implement IEEE 754 binary32 arithmetic, so a given sequence of `float32` operations produces bit-for-bit identical results in both languages *if* the exact same operations are performed in the exact same order. This port deliberately preserves the original operation order in every numerically meaningful spot (softmax, the REINFORCE loss/gradient, the matmul accumulation order) for this reason, even though exact byte-for-byte parity with the C binary was explicitly not a goal of this port (see `05-porting-notes.md`) — the two implementations are structured differently enough (e.g. a different random number generator, a restructured graph-building algorithm) that bit-for-bit parity was never realistically achievable end-to-end, but preserving float32 arithmetic and operation order keeps this port numerically faithful in spirit.

One departure: `math.Exp`/`math.Log` in Go's standard library operate on `float64`, so `pkg/mat` converts to `float64`, calls the standard library function, and converts back to `float32` (Go's standard library has no built-in `float32` exp/log). This is, if anything, slightly *more* precise per individual call than C's native `expf`/`logf` (which compute directly in lower precision), not less.

## Adam's division-by-near-zero guard

`Adam.Step` (`pkg/actorcritic/adam.go`, see `09-adam-optimizer-and-minibatch-training.md` for what this optimizer does and why) computes, for every parameter:

```go
p.Val.Data[j] -= a.learningRate * moment1Hat / (float32(math.Sqrt(float64(moment2Hat))) + epsilon)
```

`moment2Hat` is a bias-corrected estimate of that parameter's recent squared gradient magnitude, which starts at zero and can legitimately stay very close to zero for a parameter whose gradient has been consistently tiny (or exactly zero, e.g. early in training before a particular part of the network has been exercised by any data). Dividing by `sqrt(moment2Hat)` directly would then divide by (near) zero, producing an enormous or `Inf`/`NaN` step for exactly the parameter that most needs a *small*, cautious update. `epsilon` (`adamEpsilon`, `1e-8`) is added to the denominator specifically to prevent this: it's negligible whenever `sqrt(moment2Hat)` is already a reasonable size, but puts a firm floor under the denominator when it isn't, capping how large a single Adam step can ever be for a parameter with little gradient history yet.

## Where to go next

`09-adam-optimizer-and-minibatch-training.md` explains what the Adam optimizer above is actually computing and why, beyond just the numerical guard covered here.
