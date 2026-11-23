package mat

import (
	"sync/atomic"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
)

// useBLAS selects MatMul's implementation process-wide: true (the
// default, set by this file's init) delegates to gonum's pure-Go
// blas32.Gemm; false uses this package's own hand-written
// matMulNN/NT/TN/TT loops. See SetBLASBackend.
var useBLAS atomic.Bool

func init() {
	useBLAS.Store(true)
}

// SetBLASBackend turns MatMul's BLAS-backed implementation on (the
// default) or off, process-wide, for every *Matrix. pkg/config.Settings'
// DisableBLAS field is how a training run wires this up (see
// pkg/ppo.newTrainer/newRecurrentTrainer) — this function is what
// actually flips the switch underneath that setting. Exists as its own
// toggle, rather than always using whichever backend was linked in, so
// the original pure-Go loops stay reachable without a code change: as a
// fallback if gonum's blas32.Gemm is ever suspected of a bug (MatMul's
// own shape/argument validation runs identically either way — only the
// actual multiply differs), or to reproduce performance numbers
// measured before this package depended on gonum at all.
//
// Safe to call concurrently with MatMul, but intended to be set once
// before training starts, not toggled mid-run: this doesn't synchronize
// with MatMul calls already in flight on another goroutine, so one
// might briefly still observe the old setting after this returns.
func SetBLASBackend(on bool) {
	useBLAS.Store(on)
}

// BLASBackendEnabled reports MatMul's current process-wide backend
// choice (see SetBLASBackend) - mainly for tests and startup
// diagnostics that want to confirm a config setting actually took
// effect, rather than driving MatMul's own behavior.
func BLASBackendEnabled() bool {
	return useBLAS.Load()
}

// matMulBLAS computes dst = 1*op(a)*op(b) + beta*dst via gonum's
// blas32.Gemm, where beta is 0 (overwrite dst, matching zeroOut=true) or
// 1 (accumulate into dst, matching zeroOut=false) — Gemm's own beta
// parameter already expresses MatMul's zeroOut exactly, so this path
// never calls dst.Clear() itself (contrast the non-BLAS path, which
// clears dst before its own +=-shaped matMulNN/NT/TN/TT helpers run).
func matMulBLAS(dst, a, b *Matrix, zeroOut, transposeA, transposeB bool) {
	tA, tB := blas.NoTrans, blas.NoTrans
	if transposeA {
		tA = blas.Trans
	}
	if transposeB {
		tB = blas.Trans
	}

	beta := float32(1)
	if zeroOut {
		beta = 0
	}

	blas32.Gemm(tA, tB, 1, asGeneral(a), asGeneral(b), beta, asGeneral(dst))
}

// asGeneral views m as a gonum blas32.General. m's row-major Data is
// already exactly what General expects, with Stride equal to m.Cols:
// every *Matrix this package allocates (see New) is a dense, unpadded,
// contiguous row-major buffer, never a sub-view with a larger stride.
func asGeneral(m *Matrix) blas32.General {
	return blas32.General{Rows: m.Rows, Cols: m.Cols, Stride: m.Cols, Data: m.Data}
}
