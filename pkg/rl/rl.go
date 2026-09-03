// Package rl defines the environment-agnostic types (Observation, Action,
// StepResult, Environment) and experience representation (Transition,
// Episode) shared by every RL training algorithm in this module,
// decoupling pkg/reinforce and pkg/policy from any specific environment
// implementation (e.g. pkg/snakeenv, pkg/gridworldenv).
package rl

import "context"

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
// built and run in parallel the way pkg/snakeenv can. See
// pkg/reinforce.EnvFactory (construct a fresh Environment per episode)
// and pkg/reinforce.PersistentEnvFactory (construct one Environment
// once and Reset it between episodes) for the two ways a caller can
// drive one.
//
// Reset and Step take a context.Context so a caller can cancel or time
// out a call against an Environment that can block (e.g. a live game
// session waiting on a network round-trip) — none of this module's own
// toy environments need this, but a real environment adapter built on
// this interface may.
type Environment interface {
	// Reset starts a new episode and returns the initial Observation.
	Reset(ctx context.Context) (Observation, error)
	// Step applies action and returns the resulting StepResult.
	Step(ctx context.Context, action Action) (StepResult, error)
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

// Decision is the result of a policy's action-selection process,
// exposing the full distribution that produced the sampled Action
// instead of only the outcome, so a caller can audit why a decision
// was made rather than only observe what it was. See
// pkg/policy.Actor.ActWithInfo and pkg/actorcritic.Actor.ActWithInfo,
// and docs/plans/16-decision-auditing-and-explainability.md.
type Decision struct {
	Action Action `json:"action"`
	// Probabilities is the final distribution Action was sampled from:
	// the policy's own output, renormalized over whatever action mask
	// (if any) was applied. Its length is always the environment's
	// ActionSpace(); it sums to 1 over legal actions and is 0
	// elsewhere.
	Probabilities []float32 `json:"probabilities"`
	// RawProbabilities is the policy's output distribution before any
	// mask was applied, letting a caller distinguish "confidently chose
	// this among several legal options" from "only one option was
	// legal in the first place." Identical to Probabilities when no
	// mask was given.
	RawProbabilities []float32 `json:"raw_probabilities"`
	// Value is the critic's estimated value of the observation that
	// produced this Decision. HasValue is false for a policy with no
	// critic (e.g. pkg/policy.Actor), distinguishing "no value head"
	// from a genuinely-estimated value of 0.
	Value    float32 `json:"value"`
	HasValue bool    `json:"has_value"`
}
