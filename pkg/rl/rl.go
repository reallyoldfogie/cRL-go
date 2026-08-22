// Package rl defines the environment-agnostic types (Observation, Action,
// StepResult, Environment) and experience representation (Transition,
// Episode) shared by every RL training algorithm in this module,
// decoupling pkg/reinforce and pkg/policy from any specific environment
// implementation (e.g. pkg/snakeenv, pkg/gridworldenv).
package rl

// Observation is the fixed-length numeric feature vector a policy
// network consumes at each step. Every Environment implementation must
// produce Observations of the same length (see Environment.ObservationSize).
type Observation struct {
	Values []float32
}

// Action is an opaque index into an environment's action space. What
// index N actually means (a grid move, a high-level game command, ...)
// is defined entirely by the Environment implementation, and, in turn,
// by whatever consumes an Environment's chosen Action outside of it.
type Action int

// StepResult is what an Environment reports after applying an Action:
// the resulting Observation, the reward earned, and whether the episode
// has ended.
type StepResult struct {
	Observation Observation
	Reward      float32
	Done        bool
}

// Environment is the minimal surface pkg/reinforce needs to collect
// rollouts and train a policy: reset to a fresh episode, and advance one
// step at a time. Implementations are free to be cheap and stateless
// (e.g. pkg/snakeenv, safe to construct per rollout goroutine) or slow
// and stateful (e.g. a live game-session adapter); Environment itself
// makes no promises about construction cost or concurrency safety across
// instances, so callers should not assume every implementation can be
// built and run in parallel the way pkg/snakeenv can.
type Environment interface {
	// Reset starts a new episode and returns the initial Observation.
	Reset() (Observation, error)
	// Step applies action and returns the resulting StepResult.
	Step(action Action) (StepResult, error)
	// ObservationSize is the fixed length of every Observation this
	// Environment produces.
	ObservationSize() int
	// ActionSpace is the number of distinct actions this Environment
	// accepts, i.e. valid Actions are in [0, ActionSpace()).
	ActionSpace() int
}

// Transition records one (observation, action, reward) step of
// experience, plus whether it ended the episode.
type Transition struct {
	Observation Observation
	Action      Action
	Reward      float32
	Done        bool
}

// Episode is one full trajectory: the ordered sequence of Transitions
// collected between a Reset and the episode ending (Done, or the
// collector's step limit), regardless of whether that's 80 grid moves in
// Snake or a dozen high-level decisions in a very different environment.
type Episode struct {
	Transitions []Transition
}
