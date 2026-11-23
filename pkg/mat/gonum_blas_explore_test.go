package mat

// Exploratory only — answers "would swapping in a BLAS backend help?"
// for the actual matrix sizes this codebase uses (256-wide, 3-deep
// recurrent training, batched 16-wide). Not wired into MatMul; delete
// along with the gonum.org/v1/gonum go.mod entries if the answer turns
// out to be "no."

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
)

func randMatrix(rows, cols int, rng *rand.Rand) *Matrix {
	m := New(rows, cols)
	m.FillRand(rng, -1, 1)
	return m
}

func toGeneral(m *Matrix) blas32.General {
	return blas32.General{Rows: m.Rows, Cols: m.Cols, Stride: m.Cols, Data: m.Data}
}

// BenchmarkMatMulVsGonumBLAS compares this package's hand-written
// matMulNN/matMulTN against gonum's pure-Go (no cgo) BLAS backend at
// sizes drawn directly from real recurrent training: a 256x256 weight
// (hiddenSize=256) against a 256x16 batched input (BatchSize=16, the
// sweet spot found in pkg/ppo's benchmarks), for both the forward pass's
// NN case and the backward pass's TN case (weight-gradient
// accumulation: dW += grad * input^T-shaped contraction).
func BenchmarkMatMulVsGonumBLAS(b *testing.B) {
	const (
		hiddenSize = 256
		batchSize  = 16
	)
	rng := rand.New(rand.NewPCG(1, 2))

	b.Run(fmt.Sprintf("NN/mat/%dx%d_%dx%d", hiddenSize, hiddenSize, hiddenSize, batchSize), func(b *testing.B) {
		w := randMatrix(hiddenSize, hiddenSize, rng)
		x := randMatrix(hiddenSize, batchSize, rng)
		dst := New(hiddenSize, batchSize)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = dst.MatMul(w, x, true, false, false)
		}
	})

	b.Run(fmt.Sprintf("NN/gonum/%dx%d_%dx%d", hiddenSize, hiddenSize, hiddenSize, batchSize), func(b *testing.B) {
		w := toGeneral(randMatrix(hiddenSize, hiddenSize, rng))
		x := toGeneral(randMatrix(hiddenSize, batchSize, rng))
		c := toGeneral(New(hiddenSize, batchSize))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			blas32.Gemm(blas.NoTrans, blas.NoTrans, 1, w, x, 0, c)
		}
	})

	b.Run(fmt.Sprintf("TN/mat/%dx%d_%dx%d", hiddenSize, hiddenSize, hiddenSize, batchSize), func(b *testing.B) {
		a := randMatrix(hiddenSize, hiddenSize, rng) // a^T * grad: a is (hiddenSize,hiddenSize)
		grad := randMatrix(hiddenSize, batchSize, rng)
		dst := New(hiddenSize, batchSize)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = dst.MatMul(a, grad, true, true, false)
		}
	})

	b.Run(fmt.Sprintf("TN/gonum/%dx%d_%dx%d", hiddenSize, hiddenSize, hiddenSize, batchSize), func(b *testing.B) {
		a := toGeneral(randMatrix(hiddenSize, hiddenSize, rng))
		grad := toGeneral(randMatrix(hiddenSize, batchSize, rng))
		c := toGeneral(New(hiddenSize, batchSize))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			blas32.Gemm(blas.Trans, blas.NoTrans, 1, a, grad, 0, c)
		}
	})
}
