// Package config holds training hyperparameters (Settings), a JSON
// config-file loader, and CLI-flag overlay logic: CLI flags override
// config-file values when both are present.
//
// Config *code* lives here; the actual default config *data* lives in
// configs/config.json at the module root.
package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"runtime"
)

// Settings holds every tunable hyperparameter for a training run. Field
// names use JSON tags matching configs/config.json.
type Settings struct {
	// Epochs is the number of training epochs to run.
	Epochs int `json:"epochs"`
	// RolloutSize is the number of trajectories collected per epoch.
	RolloutSize int `json:"rollout_size"`
	// EpisodeLen is the maximum number of steps per trajectory.
	EpisodeLen int `json:"episode_len"`
	// Gamma is the discount factor applied to future rewards when
	// computing reward-to-go.
	Gamma float32 `json:"gamma"`
	// LearningRate scales the averaged gradient in each SGD step.
	LearningRate float32 `json:"learning_rate"`
	// GridSize is the number of cells in the environment's grid; must be
	// a perfect square (e.g. 36 for a 6x6 grid).
	GridSize int `json:"grid_size"`
	// HiddenSize is the width of both hidden layers in the policy MLP.
	HiddenSize int `json:"hidden_size"`
	// Seed is the master RNG seed. Per-worker seeds for concurrent
	// rollout collection are derived from this value so that a fixed
	// seed reproduces the same set of worker streams regardless of
	// goroutine scheduling order.
	Seed uint64 `json:"seed"`
	// Workers is the number of goroutines used for concurrent rollout
	// collection.
	Workers int `json:"workers"`
}

// Default returns the out-of-the-box hyperparameters, matching the
// original C implementation's hardcoded constants (env.c/model.c) except
// where noted in docs/05-porting-notes.md.
func Default() Settings {
	return Settings{
		Epochs:       2000,
		RolloutSize:  64,
		EpisodeLen:   100,
		Gamma:        0.99,
		LearningRate: 0.05,
		GridSize:     36,
		HiddenSize:   128,
		Seed:         42,
		Workers:      runtime.NumCPU(),
	}
}

// Load reads Settings from a JSON file at path, starting from Default()
// and overwriting only the fields present in the file. If path does not
// exist, Load returns Default() unchanged (a config file is optional).
func Load(path string) (Settings, error) {
	settings := Default()

	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return settings, nil
	}
	if err != nil {
		return Settings{}, fmt.Errorf("config: reading %s: %w", path, err)
	}

	if err := json.Unmarshal(data, &settings); err != nil {
		return Settings{}, fmt.Errorf("config: parsing %s: %w", path, err)
	}
	return settings, nil
}

// Validate reports whether s describes a usable training configuration.
func (s Settings) Validate() error {
	side := int(math.Sqrt(float64(s.GridSize)))
	if side*side != s.GridSize {
		return fmt.Errorf("config: grid_size %d must be a perfect square", s.GridSize)
	}
	if s.RolloutSize <= 0 {
		return fmt.Errorf("config: rollout_size must be positive, got %d", s.RolloutSize)
	}
	if s.EpisodeLen <= 0 {
		return fmt.Errorf("config: episode_len must be positive, got %d", s.EpisodeLen)
	}
	if s.HiddenSize <= 0 {
		return fmt.Errorf("config: hidden_size must be positive, got %d", s.HiddenSize)
	}
	if s.Workers <= 0 {
		return fmt.Errorf("config: workers must be positive, got %d", s.Workers)
	}
	return nil
}
