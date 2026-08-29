package autograd

// Graph holds a topologically-sorted list of Vars: every Var's inputs
// appear before it in Vars. Forward walks the list in order (values flow
// from inputs to outputs); Backward walks it in reverse (gradients flow
// from the output back to the inputs).
type Graph struct {
	Vars []*Var
}

// BuildGraph traverses the computation graph rooted at output and returns
// a Graph containing every reachable Var in topological order.
//
// This uses a straightforward recursive post-order depth-first search
// keyed by Var identity (a map[*Var]bool), which is O(V+E) and needs no
// upfront bound on the number of vars. The original C implementation
// (autograd.c's build_graph) used a hand-rolled iterative stack with an
// O(n^2) removal step to re-order already-queued nodes; that quadratic
// behavior isn't reproduced here. See docs/05-porting-notes.md.
func BuildGraph(output *Var) *Graph {
	return BuildGraphMulti(output)
}

// BuildGraphMulti is BuildGraph generalized to more than one root: it
// traverses every Var reachable from any of roots and returns a single
// Graph containing the union, still in a valid topological order (every
// Var's inputs appear before it), with any Vars shared between roots'
// dependency trees (e.g. a shared hidden trunk feeding two independent
// output heads) visited and included only once. This lets a single
// Graph.Forward call compute several otherwise-unrelated outputs (e.g.
// an actor-critic network's separate policy and value heads) in one
// pass, without requiring them to be combined into a single Var first.
//
// Graph.Backward's single-root seeding (see its doc comment) assumes the
// *last* Var in topological order is the thing being differentiated;
// BuildGraphMulti doesn't change that assumption, so a graph built with
// more than one root should generally only be used for Forward (as
// InferenceNetwork does) rather than Backward.
func BuildGraphMulti(roots ...*Var) *Graph {
	visited := make(map[*Var]bool)
	var order []*Var

	var visit func(v *Var)
	visit = func(v *Var) {
		if v == nil || visited[v] {
			return
		}
		visited[v] = true

		for _, input := range v.Inputs {
			visit(input)
		}
		order = append(order, v)
	}
	for _, root := range roots {
		visit(root)
	}

	return &Graph{Vars: order}
}

// Forward computes every node's Val, in dependency order.
func (g *Graph) Forward() {
	for _, v := range g.Vars {
		if v.Op != nil {
			v.Op.Forward(v)
		}
	}
}

// Backward computes gradients for every node that requires one, seeding
// the final (last topologically-sorted) node's gradient with 1 and
// applying the chain rule in reverse.
//
// Non-parameter gradients are cleared before each call so that stale
// gradients from a previous Backward call don't leak in. Parameter
// gradients (FlagParameter) are left untouched, so callers can accumulate
// gradients across many Forward/Backward calls (e.g. once per step of
// every trajectory in a training batch) before applying a single
// optimizer step and then clearing them explicitly.
func (g *Graph) Backward() {
	if len(g.Vars) == 0 {
		return
	}

	output := g.Vars[len(g.Vars)-1]
	if output.Grad == nil {
		// Nothing requires a gradient; this graph was likely built for
		// inference only (e.g. a forward-only rollout graph).
		return
	}

	for _, v := range g.Vars {
		if !requiresGrad(v) || v.Flags&FlagParameter != 0 {
			continue
		}
		v.Grad.Clear()
	}

	output.Grad.Fill(1.0)

	for i := len(g.Vars) - 1; i >= 0; i-- {
		v := g.Vars[i]
		if !requiresGrad(v) {
			continue
		}
		if v.Op != nil {
			v.Op.Backward(v)
		}
	}
}
