// Package hierarchical implements a two-level PPO-trained agent: a
// meta-controller that periodically selects a coarse Subgoal, and one
// specialized sub-policy per Subgoal choosing primitive rl.Actions
// while it's active. See
// docs/plans/11-hierarchical-meta-controller-and-subpolicies.md for the
// full design and rationale.
//
// A subgoal-selection decision is structurally identical to a
// primitive-action decision — both are categorical action selection
// scored with GAE advantages and the PPO clip objective — so this
// package composes pkg/ppo/pkg/actorcritic directly (one
// ppo.TrainingNetwork/actorcritic.Adam pair for the meta-controller,
// one more per Subgoal) rather than reimplementing any loss or GAE
// math of its own.
package hierarchical

// Subgoal is an index into the meta-controller's output space: which
// coarse objective is currently active. What each index actually means
// (e.g. "build", "survive", "explore") is defined entirely by the
// caller — mc-agent eventually, a toy environment for validation here —
// not by this package; see Config.NumSubgoals. Subgoal is given its own
// type, rather than a plain int, so it can never be silently confused
// with a primitive rl.Action at a call site even though both are
// int-like.
type Subgoal int
