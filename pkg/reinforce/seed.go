package reinforce

import "math/rand/v2"

// splitmix64 is Sebastiano Vigna's SplitMix64 finalizer, used here purely
// to mix a few small integers (a master seed, an epoch, and a worker
// index) into well-distributed 64-bit values. It is not used as a
// general-purpose random number generator.
func splitmix64(x uint64) uint64 {
	x += 0x9E3779B97F4A7C15
	z := x
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

// WorkerRNG deterministically derives an independent random source for
// the given (epoch, worker) pair from masterSeed, using a pure function
// of its inputs rather than sequential draws from a shared generator.
// That independence is what makes concurrent rollout collection both
// race-free (each goroutine only ever touches its own *rand.Rand) and
// reproducible (a fixed masterSeed always derives the same set of
// per-worker streams, regardless of goroutine scheduling order).
//
// This replaces the original C program's randomness setup, which mixed a
// global PCG stream (prng.c) with a separate, unseeded libc RNG
// (arc4random_uniform) for food placement — see docs/05-porting-notes.md.
//
// Exported (rather than kept private to this package) so pkg/ppo's
// trainer can derive its own reproducible per-worker and per-shuffle
// streams from the same well-tested seed-mixing scheme instead of
// duplicating it.
func WorkerRNG(masterSeed uint64, epoch, worker int) *rand.Rand {
	base := masterSeed ^ uint64(epoch)<<32 ^ uint64(uint32(worker))
	seed1 := splitmix64(base)
	seed2 := splitmix64(base ^ 0xD1B54A32D192ED03)
	return rand.New(rand.NewPCG(seed1, seed2))
}
