package hierarchical

import "fmt"

// Config holds the hierarchy-specific knobs that don't fit
// config.Settings' shape (which assumes a single flat network). Knobs
// that already apply uniformly to any PPO-trained network — Gamma,
// GAELambda, ClipEpsilon, EntropyCoef, ValueCoef, PPOEpochs,
// MinibatchSize, RolloutSize, EpisodeLen, Workers, Seed — are reused
// directly from config.Settings by Trainer instead of being duplicated
// here.
type Config struct {
	// NumSubgoals is the width of the meta-controller's output layer:
	// how many coarse subgoals it chooses among. What each index means
	// is entirely up to the caller; see Subgoal's doc comment.
	NumSubgoals int
	// SubgoalInterval is the number of environment steps between
	// meta-controller decisions.
	SubgoalInterval int
	// MetaHiddenSize and SubHiddenSize are the meta-controller's and
	// every sub-policy's hidden layer width, respectively. Kept
	// separate (rather than one shared HiddenSize) since a
	// meta-controller reasoning over coarse subgoals and a sub-policy
	// choosing primitive actions are differently-sized problems.
	MetaHiddenSize int
	SubHiddenSize  int
	// MetaLearningRate and SubLearningRate are the meta-controller's
	// and every sub-policy's Adam learning rate, respectively, kept
	// separate for the same reason as the hidden sizes above —
	// mirroring mc-rl-go's HierarchicalConfig keeping MetaLR/SubLR
	// distinct.
	MetaLearningRate float32
	SubLearningRate  float32
}

// Validate reports whether cfg describes a usable hierarchical
// configuration.
func (cfg Config) Validate() error {
	if cfg.NumSubgoals <= 0 {
		return fmt.Errorf("hierarchical: num_subgoals must be positive, got %d", cfg.NumSubgoals)
	}
	if cfg.SubgoalInterval <= 0 {
		return fmt.Errorf("hierarchical: subgoal_interval must be positive, got %d", cfg.SubgoalInterval)
	}
	if cfg.MetaHiddenSize <= 0 {
		return fmt.Errorf("hierarchical: meta_hidden_size must be positive, got %d", cfg.MetaHiddenSize)
	}
	if cfg.SubHiddenSize <= 0 {
		return fmt.Errorf("hierarchical: sub_hidden_size must be positive, got %d", cfg.SubHiddenSize)
	}
	if cfg.MetaLearningRate <= 0 {
		return fmt.Errorf("hierarchical: meta_learning_rate must be positive, got %g", cfg.MetaLearningRate)
	}
	if cfg.SubLearningRate <= 0 {
		return fmt.Errorf("hierarchical: sub_learning_rate must be positive, got %g", cfg.SubLearningRate)
	}
	return nil
}
