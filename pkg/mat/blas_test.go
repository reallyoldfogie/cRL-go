package mat

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func randMatrix(rows, cols int, rng *rand.Rand) *Matrix {
	m := New(rows, cols)
	m.FillRand(rng, -1, 1)
	return m
}

// TestMatMulBLASMatchesNonBLASBackend is SetBLASBackend's correctness
// guard: for every transposeA/transposeB combination, and for both
// zeroOut values, the BLAS-backed and non-BLAS-backed implementations
// of MatMul must agree (within float32 tolerance) on the same inputs —
// SetBLASBackend only ever changes which arithmetic runs, never what a
// caller is allowed to pass in or what shape it produces.
func TestMatMulBLASMatchesNonBLASBackend(t *testing.T) {
	t.Cleanup(func() { SetBLASBackend(true) })

	rng := rand.New(rand.NewPCG(5, 7))
	const m, n, k = 11, 7, 13 // deliberately non-square, non-power-of-2

	for _, transposeA := range []bool{false, true} {
		for _, transposeB := range []bool{false, true} {
			for _, zeroOut := range []bool{true, false} {
				t.Run(fmt.Sprintf("transposeA=%v/transposeB=%v/zeroOut=%v", transposeA, transposeB, zeroOut), func(t *testing.T) {
					aRows, aCols := k, m
					if !transposeA {
						aRows, aCols = m, k
					}
					bRows, bCols := n, k
					if !transposeB {
						bRows, bCols = k, n
					}
					a := randMatrix(aRows, aCols, rng)
					b := randMatrix(bRows, bCols, rng)

					blasDst := randMatrix(m, n, rng)
					nonBLASDst := New(m, n)
					copy(nonBLASDst.Data, blasDst.Data)

					SetBLASBackend(true)
					require.NoError(t, blasDst.MatMul(a, b, zeroOut, transposeA, transposeB))

					SetBLASBackend(false)
					require.NoError(t, nonBLASDst.MatMul(a, b, zeroOut, transposeA, transposeB))

					for i := range blasDst.Data {
						assert.InDelta(t, nonBLASDst.Data[i], blasDst.Data[i], 1e-3, "element %d", i)
					}
				})
			}
		}
	}
}

// BenchmarkMatMulVsGonumBLAS compares this package's hand-written
// matMulNN/matMulTN (SetBLASBackend(false)) against the default,
// gonum-backed MatMul (SetBLASBackend(true), the package default) at
// sizes drawn directly from real recurrent training: a 256x256 weight
// (hiddenSize=256) against a 256x16 batched input (BatchSize=16, the
// sweet spot found in pkg/ppo's benchmarks), for both the forward
// pass's NN case and the backward pass's TN case (weight-gradient
// accumulation).
func BenchmarkMatMulVsGonumBLAS(b *testing.B) {
	const (
		hiddenSize = 256
		batchSize  = 16
	)
	rng := rand.New(rand.NewPCG(1, 2))
	b.Cleanup(func() { SetBLASBackend(true) })

	cases := []struct {
		name        string
		transposeA  bool
		transposeB  bool
		aRows, aCol int
	}{
		{"NN", false, false, hiddenSize, hiddenSize},
		{"TN", true, false, hiddenSize, hiddenSize},
	}
	for _, c := range cases {
		for _, useBLASCase := range []bool{false, true} {
			backend := "mat"
			if useBLASCase {
				backend = "gonum"
			}
			b.Run(fmt.Sprintf("%s/%s/%dx%d_%dx%d", c.name, backend, c.aRows, c.aCol, hiddenSize, batchSize), func(b *testing.B) {
				a := randMatrix(c.aRows, c.aCol, rng)
				x := randMatrix(hiddenSize, batchSize, rng)
				dst := New(hiddenSize, batchSize)
				SetBLASBackend(useBLASCase)

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_ = dst.MatMul(a, x, true, c.transposeA, c.transposeB)
				}
			})
		}
	}
}
