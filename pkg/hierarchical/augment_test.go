package hierarchical

import (
	"testing"

	"github.com/reallyoldfogie/cRL-go/pkg/rl"
	"github.com/stretchr/testify/assert"
)

func TestAugmentObservationAppendsOneHotSubgoal(t *testing.T) {
	obs := rl.Observation{Values: []float32{0.1, 0.2, 0.3}}

	augmented := augmentObservation(obs, Subgoal(1), 4)

	assert.Equal(t, []float32{0.1, 0.2, 0.3, 0, 1, 0, 0}, augmented.Values)
}

func TestAugmentObservationForFirstSubgoal(t *testing.T) {
	obs := rl.Observation{Values: []float32{1, 2}}

	augmented := augmentObservation(obs, Subgoal(0), 3)

	assert.Equal(t, []float32{1, 2, 1, 0, 0}, augmented.Values)
}

func TestAugmentObservationDoesNotMutateInput(t *testing.T) {
	original := []float32{5, 6}
	obs := rl.Observation{Values: original}

	_ = augmentObservation(obs, Subgoal(1), 2)

	assert.Equal(t, []float32{5, 6}, original, "augmentObservation must not mutate the input Observation's Values")
}
