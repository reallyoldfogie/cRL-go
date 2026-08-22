package reinforce

import "math"

// returnStatistics computes the mean and standard deviation of every
// discounted return across every episode in the batch, along with the
// total number of individual steps ("samples") that contributes.
//
// Variance is computed with the naive one-pass formula
// E[X^2] - E[X]^2, matching env.c's return_mean/return_variance
// computation. This is a numerically fragile formula in general (it can
// go slightly negative due to floating-point cancellation when the
// variance is small relative to the mean), which is why the result is
// clamped to zero before taking its square root — see
// docs/04-numerical-stability-notes.md. Welford's algorithm is a more
// numerically robust alternative, noted there as a possible future
// improvement rather than applied here.
func returnStatistics(episodes []scoredEpisode) (mean, std float32, sampleCount int) {
	var sum, sumSquares float32

	for _, s := range episodes {
		for _, g := range s.Returns {
			sum += g
			sumSquares += g * g
			sampleCount++
		}
	}

	if sampleCount == 0 {
		return 0, 0, 0
	}

	mean = sum / float32(sampleCount)
	variance := sumSquares/float32(sampleCount) - mean*mean
	std = float32(math.Sqrt(float64(max(variance, 0))))
	return mean, std, sampleCount
}
