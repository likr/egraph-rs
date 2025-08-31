mod block;
mod constraint;
mod variable;

pub use constraint::Constraint;

use self::{block::Block, variable::Variable};
use ordered_float::OrderedFloat;
use petgraph_drawing::{Delta, Drawing, DrawingValue, MetricCartesian};
use std::collections::{HashMap, HashSet, VecDeque};

/// Manages the state for a one-dimensional separation constraint satisfaction algorithm
/// based on the QPSC (Quadratic Programming Separation Constraints) method from IPSEP-COLA.
///
/// The `ConstraintGraph` implements a gradient projection approach to enforce a set of
/// separation constraints of the form `v_left + gap <= v_right`. This is particularly
/// useful in graph layout algorithms where you want to enforce minimum distances between
/// nodes while still allowing other forces (like stress minimization) to operate.
///
/// The algorithm works by:
/// 1. Organizing variables into "blocks" that move rigidly together
/// 2. When constraints are violated, either merging blocks or splitting blocks
/// 3. Using Lagrangian relaxation to determine which active constraints to maintain or release
///
/// This implementation can be used in conjunction with other layout algorithms (e.g.,
/// stress majorization or force-directed) to enforce separation constraints during or
/// after the main layout process.
///
/// See IPSEP-COLA paper [2] for details on the algorithm.
pub struct ConstraintGraph<S> {
    /// State (`block` index, `offset`) for each variable.
    variables: Vec<Variable<S>>,
    /// All defined separation constraints.
    constraints: Vec<Constraint<S>>,
    /// All current blocks. Blocks can become empty after merging.
    blocks: Vec<Block<S>>,
    /// Static adjacency list storing `(neighbor_variable_index, constraint_index)` tuples
    /// for *all* constraints. Used by `comp_path` to find paths along *active* constraints.
    neighbors: Vec<Vec<(usize, usize)>>,
}

impl<S> ConstraintGraph<S>
where
    S: DrawingValue,
{
    /// Creates and initializes the `ConstraintGraph` for a specific dimension `d`.
    /// Each variable starts in its own block at its initial position.
    /// Corresponds to the initialization described at the end of Section 3.2 [1].
    ///
    /// # Arguments
    ///
    /// * `drawing` - Initial layout providing variable positions.
    /// * `d` - The dimension (0 for x, 1 for y) being constrained.
    /// * `constraints` - All separation constraints for this dimension.
    pub fn new<Diff, D, M>(drawing: &D, d: usize, constraints: &[Constraint<S>]) -> Self
    where
        D: Drawing<Item = M>,
        Diff: Delta<S = S>,
        M: MetricCartesian<D = Diff>,
    {
        let n = drawing.len();
        let constraints = constraints.to_vec();
        let variables = (0..n)
            .map(|i| Variable {
                block: i,
                offset: S::zero(),
            })
            .collect();
        let blocks = (0..n)
            .map(|i| {
                let mut variables = HashSet::new();
                variables.insert(i);
                Block {
                    variables,
                    position: *drawing.raw_entry(i).nth(d),
                    active: HashSet::new(),
                }
            })
            .collect();
        let mut neighbors: Vec<Vec<(usize, usize)>> = vec![vec![]; n];
        // Build the adjacency list used for path finding (`comp_path`).
        // Stores (neighbor_variable, constraint_index) for all constraints.
        for (c_idx, constraint) in constraints.iter().enumerate() {
            neighbors[constraint.left].push((constraint.right, c_idx));
            neighbors[constraint.right].push((constraint.left, c_idx));
        }
        ConstraintGraph {
            variables,
            constraints,
            blocks,
            neighbors,
        }
    }

    /// Calculates the optimal reference position for block `i` that minimizes squared deviation
    /// from the desired variable positions `x`, given the current variable offsets within the block.
    /// Used in `split_blocks` (Figure 10) and needed after `expand_block` (Section 3.2).
    /// Formula: `Sum(x[j] - offset[j] for j in B_i.vars) / B_i.nvars`.
    fn optimal_position(&self, i: usize, x: &[S]) -> S {
        let mut s = S::zero();
        for &j in self.blocks[i].variables.iter() {
            s += x[j] - self.variables[j].offset;
        }
        s / S::from_usize(self.blocks[i].variables.len()).unwrap()
    }

    /// Recursively computes the derivative `dF/dv` (related to Lagrange multipliers) for variable `v`
    /// within a block, considering only movement along active constraints (`active_constraints`).
    /// Used to identify which active constraint to remove during `split_blocks` or `expand_block`.
    /// Populates `lm` with multipliers for visited constraints.
    /// Corresponds to `comp_dfdv(v, AC, u)` in Figure 10 [1].
    ///
    /// # Arguments
    ///
    /// * `v` - Variable index to compute derivative for.
    /// * `active_constraints` - Set of active constraint indices in the block.
    /// * `u` - Parent variable index in recursion (to prevent cycles).
    /// * `lm` - Map to store computed Lagrange multipliers for constraints.
    /// * `x` - Vector of desired variable positions (e.g., post-gradient step).
    fn comp_dfdv(
        &self,
        v: usize,
        active_constraints: &HashSet<usize>,
        u: Option<usize>,
        lm: &mut HashMap<usize, S>,
        x: &[S],
    ) -> S {
        let mut dfdv = self.variable_position(v) - x[v];
        for &i in active_constraints.iter() {
            let c = &self.constraints[i];
            if v == c.left && u != Some(c.right) {
                let value = self.comp_dfdv(c.right, active_constraints, Some(v), lm, x);
                lm.insert(i, value);
                dfdv += value;
            } else if v == c.right && u != Some(c.left) {
                let value = -self.comp_dfdv(c.left, active_constraints, Some(v), lm, x);
                lm.insert(i, value);
                dfdv -= value;
            }
        }
        dfdv
    }

    /// Optimizes the block structure by splitting blocks when beneficial.
    ///
    /// This method lazily splits blocks if an active constraint has a negative Lagrange multiplier,
    /// indicating that making it inactive could improve the objective function. For each block,
    /// it finds the active constraint `sc` with the most negative multiplier and splits the block
    /// by removing that constraint from the active set.
    ///
    /// In graph layout terms, splitting blocks allows parts of the layout to move more independently
    /// when the rigid constraints between them are counter-productive to minimizing the overall
    /// objective function.
    ///
    /// The method adapts the block structure to better minimize the squared distance between the
    /// desired positions `x` and the positions that satisfy all constraints.
    ///
    /// Called periodically within the main QPSC solver (Figure 8).
    /// Corresponds to `split_blocks(x)` in Figure 10 [2].
    ///
    /// # Arguments
    ///
    /// * `x` - Desired variable positions (e.g., positions after a gradient step).
    ///
    /// # Returns
    ///
    /// * `true` if no blocks were split, `false` otherwise.
    ///
    pub fn split_blocks(&mut self, x: &[S]) -> bool {
        let mut nosplit = true;
        // Iterate using indices because splitting can implicitly add new blocks
        // (by reusing existing slots, although this implementation reuses 's').
        let num_blocks = self.blocks.len();
        for i in 0..num_blocks {
            if self.blocks[i].variables.is_empty() {
                continue;
            }
            self.blocks[i].position = self.optimal_position(i, x);
            let mut ac = self.blocks[i].active.clone();
            let mut lm = HashMap::new();
            let v = *self.blocks[i].variables.iter().nth(0).unwrap();
            self.comp_dfdv(v, &self.blocks[i].active, None, &mut lm, x);
            let Some(&sc) = self.blocks[i]
                .active
                .iter()
                .min_by_key(|&&i| OrderedFloat(lm[&i]))
            else {
                break;
            };
            if lm[&sc] >= S::zero() {
                break;
            }
            nosplit = false;
            ac.remove(&sc);
            let s = self.constraints[sc].right;
            self.blocks[s].variables = self.connected(s, &ac);
            for &v in self.blocks[s].variables.iter() {
                self.variables[v].block = s;
            }
            self.blocks[i].variables = self.blocks[i]
                .variables
                .difference(&self.blocks[s].variables)
                .cloned()
                .collect();
            self.blocks[i].position = self.optimal_position(i, x);
            self.blocks[s].position = self.optimal_position(s, x);
            self.blocks[s].active = ac
                .iter()
                .filter(|&&c| {
                    self.blocks[s].variables.contains(&self.constraints[c].left)
                        && self.blocks[s]
                            .variables
                            .contains(&self.constraints[c].right)
                })
                .cloned()
                .collect();
            for &c in self.blocks[s].active.iter() {
                ac.remove(&c);
            }
            self.blocks[i].active = ac;
        }
        nosplit
    }

    /// Finds all variables reachable from `s` by traversing only *active* constraints (`ac`).
    /// Used in `split_blocks` and `expand_block` to identify the component affected by removing
    /// a split constraint (`sc`). Performs a BFS using `self.neighbors` but filters by `ac`.
    fn connected(&self, s: usize, ac: &HashSet<usize>) -> HashSet<usize> {
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        queue.push_back(s);
        visited.insert(s);
        while let Some(u) = queue.pop_front() {
            // Iterate through all potential neighbors based on the full constraint graph.
            for &(neighbor_node, constraint_idx) in self.neighbors[u].iter() {
                // Only traverse if the constraint connecting them is active and the neighbor is unvisited.
                if ac.contains(&constraint_idx) && !visited.contains(&neighbor_node) {
                    // Check if the constraint actually connects u and neighbor_node (handles undirected nature).
                    let c = &self.constraints[constraint_idx];
                    if (c.left == u && c.right == neighbor_node)
                        || (c.right == u && c.left == neighbor_node)
                    {
                        queue.push_back(neighbor_node);
                        visited.insert(neighbor_node);
                    }
                }
            }
        }
        visited
    }

    /// Enforces all separation constraints by projecting positions `x` onto the feasible region.
    ///
    /// This is the main method for applying the constraints to a set of desired positions.
    /// It iteratively finds the most violated constraint and resolves it through block operations
    /// until all constraints are satisfied. The result is a new set of positions where all
    /// separation constraints are honored while minimizing the squared deviation from the
    /// original positions.
    ///
    /// The algorithm alternates between two operations:
    /// - If the violated constraint is between variables in different blocks, it merges the blocks
    /// - If the violated constraint is between variables in the same block, it rearranges the
    ///   block's internal structure by breaking one active constraint and activating the violated one
    ///
    /// Corresponds to the `project(C)` procedure in Figure 9 [2].
    ///
    /// # Arguments
    ///
    /// * `x` - Variable positions (coordinate in the current dimension) to be projected.
    ///         Modified **in-place** to satisfy all constraints.
    ///
    pub fn project(&mut self, x: &mut [S]) {
        // Iteratively find and resolve the most violated constraint (violation > tolerance).
        while let Some(c) = (0..self.constraints.len())
            .filter(|&i| self.violation(i) > (1e-1).into()) // Use tolerance for float comparison.
            .max_by_key(|&c| OrderedFloat(self.violation(c)))
        {
            if self.variables[self.constraints[c].left].block
                != self.variables[self.constraints[c].right].block
            {
                self.merge_block(
                    self.variables[self.constraints[c].left].block,
                    self.variables[self.constraints[c].right].block,
                    c,
                )
            } else {
                self.expand_block(self.variables[self.constraints[c].left].block, c, x);
            }
        }
        // After resolving all violations, update the input positions `x` based on the final block structure.
        for (i, x_i) in x.iter_mut().enumerate().take(self.variables.len()) {
            *x_i = self.blocks[self.variables[i].block].position + self.variables[i].offset;
        }
    }

    /// Calculates the current position of variable `i`: `block_position + offset`.
    #[inline]
    fn variable_position(&self, i: usize) -> S {
        self.blocks[self.variables[i].block].position + self.variables[i].offset
    }

    /// Calculates the violation of constraint `c`: `pos(left) + gap - pos(right)`.
    /// A value > 0 indicates the constraint is violated.
    #[inline]
    fn violation(&self, c: usize) -> S {
        self.variable_position(self.constraints[c].left) + self.constraints[c].gap
            - self.variable_position(self.constraints[c].right)
    }

    /// Merges block `r` into block `l` when constraint `c` (between them) is violated.
    /// Makes `c` active and updates the merged block's position, active set, and variable offsets.
    /// Corresponds to `merge_block(L, R, c)` in Figure 9 [1].
    fn merge_block(&mut self, l: usize, r: usize, c: usize) {
        // Note: The paper uses L and R; here we use l and r indices directly.
        // Block r is merged into block l.

        let constraint = &self.constraints[c];
        // Calculate the relative offset 'd' based on the constraint becoming active.
        let d = self.variables[constraint.left].offset + constraint.gap
            - self.variables[constraint.right].offset;
        let nvar_l = S::from_usize(self.blocks[l].variables.len()).unwrap();
        let nvar_r = S::from_usize(self.blocks[r].variables.len()).unwrap();

        // Calculate the new position for the merged block l using a weighted average.
        self.blocks[l].position = (self.blocks[l].position * nvar_l
            + (self.blocks[r].position - d) * nvar_r)
            / (nvar_l + nvar_r);

        // Update active constraints for the merged block l.
        let block_r_active = self.blocks[r].active.clone(); // Clone to avoid borrow issues.
        self.blocks[l].active.extend(block_r_active);
        self.blocks[l].active.insert(c); // Make the merging constraint c active.

        // Update variables formerly in block r.
        let block_r_vars = self.blocks[r].variables.clone(); // Clone to avoid borrow issues.
        for &i in block_r_vars.iter() {
            self.variables[i].block = l; // Point variable to the merged block l.
            self.variables[i].offset += d; // Adjust offset relative to l's new position.
        }
        self.blocks[l].variables.extend(block_r_vars); // Add r's variables to l.

        // Mark block r as empty. It remains in the `blocks` vector but is no longer used.
        self.blocks[r].variables.clear();
        self.blocks[r].active.clear();
    }

    /// Resolves a violation of constraint `c` where both variables are already in block `b`.
    /// Finds an active constraint `sc` to remove from the block's spanning tree, makes `c` active,
    /// and shifts the part of the block reachable from `right(c)` (without `sc`) to satisfy `c`.
    /// Corresponds to `expand_block(b, c̃)` in Figure 9 [1].
    fn expand_block(&mut self, b: usize, c: usize, x: &[S]) {
        let mut lm = HashMap::new(); // Stores computed Lagrange multipliers for active constraints.
        let mut ac = self.blocks[b].active.clone(); // Work with a copy of the active set.

        // Compute Lagrange multipliers for active constraints by traversing from left(c).
        self.comp_dfdv(self.constraints[c].left, &ac, None, &mut lm, x);
        // Find the path between the variables of the violated constraint c.
        let vs = self.comp_path(self.constraints[c].left, self.constraints[c].right, &ac);
        // Identify active constraints (`ps`) that lie on this path and point in the same direction.
        let vs_adj = (1..vs.len())
            .map(|i| (vs[i - 1], vs[i]))
            .collect::<HashSet<_>>();
        let ps: Vec<usize> = ac // Explicit type for clarity
            .iter()
            .filter(|&&constraint_idx| {
                let constraint = &self.constraints[constraint_idx];
                vs_adj.contains(&(constraint.left, constraint.right))
            })
            .cloned()
            .collect();

        // Find the active constraint `sc` on the path `ps` with the minimum Lagrange multiplier.
        // This is the constraint that will be deactivated to resolve the violation.
        if let Some(&sc) = ps.iter().min_by_key(|&&constraint_idx| {
            // Ensure the lm map contains the key before accessing.
            // If comp_dfdv didn't reach a constraint in ps (shouldn't happen?), treat its lm as infinity.
            OrderedFloat(lm.get(&constraint_idx).copied().unwrap_or(S::infinity()))
        }) {
            // Deactivate the chosen split constraint `sc`.
            ac.remove(&sc);
            // Identify the component connected to right(c) without using `sc`.
            let component_to_shift = self.connected(self.constraints[c].right, &ac);
            // Shift this component by the violation amount to satisfy `c`.
            let violation_amount = self.violation(c); // Calculate violation before modifying offsets.
            for &v in component_to_shift.iter() {
                // Ensure variable exists before modifying offset (safety check)
                if let Some(var) = self.variables.get_mut(v) {
                    var.offset += violation_amount;
                }
            }
            // Activate the originally violated constraint `c`.
            ac.insert(c);
            // Update the block's active set.
            self.blocks[b].active = ac;
            // Recalculate the block's reference position based on new variable positions/offsets.
            self.blocks[b].position = self.optimal_position(b, x);
        }
        // Else: No suitable split constraint found (e.g., path `ps` was empty or lm lookup failed unexpectedly).
        // This might indicate an issue elsewhere (e.g., comp_path, comp_dfdv). The violation remains unresolved.
    }

    /// Finds the path between variables `s` and `t` using only the *active* constraints `ac`.
    /// Used by `expand_block` to determine which active constraints lie on the path
    /// between the variables of an internally violated constraint. Performs a BFS.
    /// Corresponds to `comp_path(u, v, AC)` mentioned in Section 3.2 [1].
    ///
    /// # Arguments
    /// * `s` - Starting variable index.
    /// * `t` - Target variable index.
    /// * `ac` - Set of active constraint indices defining the graph to search.
    ///
    /// # Returns
    /// * A `Vec<usize>` containing the sequence of variable indices from `s` to `t`, inclusive.
    ///   Returns an empty vector if `t` is unreachable from `s` via `ac`.
    fn comp_path(&self, s: usize, t: usize, ac: &HashSet<usize>) -> Vec<usize> {
        let mut queue = VecDeque::new();
        let mut parent: HashMap<usize, Option<usize>> = HashMap::new(); // Store parent pointers for path reconstruction
        queue.push_back(s);
        parent.insert(s, None); // Start node has no parent

        'bfs: while let Some(u) = queue.pop_front() {
            if u == t {
                break 'bfs; // Found target
            }
            // Iterate over neighbors connected by *any* constraint in the original graph.
            for &(neighbor_node, constraint_idx) in self.neighbors[u].iter() {
                // Only consider traversing if the constraint is active and the neighbor hasn't been visited.
                if ac.contains(&constraint_idx) && !parent.contains_key(&neighbor_node) {
                    // Check if the constraint connects u and neighbor_node (handles undirected nature within the path).
                    let c = &self.constraints[constraint_idx];
                    if (c.left == u && c.right == neighbor_node)
                        || (c.right == u && c.left == neighbor_node)
                    {
                        parent.insert(neighbor_node, Some(u));
                        queue.push_back(neighbor_node);
                    }
                }
            }
        }

        // Reconstruct path if target was found.
        let mut path = Vec::new();
        let mut curr = Some(t);
        while let Some(node_idx) = curr {
            // If we are at a node other than start and have no parent entry, target was unreachable.
            if !parent.contains_key(&node_idx) && node_idx != s {
                return Vec::new();
            }
            path.push(node_idx);
            if node_idx == s {
                break; // Reached start
            }
            // Get the parent; if no parent exists and we are not at s, path is broken (unreachable).
            curr = match parent.get(&node_idx) {
                Some(&Some(p)) => Some(p),            // Parent exists
                Some(&None) if node_idx == s => None, // Reached start, successfully finished.
                _ => return Vec::new(),               // Target was unreachable.
            };
        }

        path.reverse(); // Path reconstructed backward, so reverse it.

        // Final check: ensure path starts with s.
        if path.is_empty() || path[0] != s {
            Vec::new() // Return empty if t wasn't reached or path is broken.
        } else {
            path
        }
    }
}

pub fn project_1d<Diff, D, M, S>(drawing: &mut D, k: usize, constraints: &[Constraint<S>])
where
    D: Drawing<Item = M>,
    Diff: Delta<S = S>,
    M: MetricCartesian<D = Diff>,
    S: DrawingValue,
{
    let n = drawing.len();
    let mut cg = ConstraintGraph::new(drawing, k, constraints);
    let mut x = (0..n)
        .map(|i| *drawing.raw_entry(i).nth(k))
        .collect::<Vec<_>>();
    cg.project(&mut x);
    for i in 0..n {
        *drawing.raw_entry_mut(i).nth_mut(k) = x[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::{Graph, NodeIndex, UnGraph};
    use petgraph_drawing::DrawingEuclidean;

    // Helper to create a ConstraintGraph for testing (1D assumed).
    fn create_test_graph(
        n: usize,
        initial_positions: &[f32],
        constraints_data: &[(usize, usize, f32)],
    ) -> ConstraintGraph<f32> {
        let mut graph: UnGraph<(), ()> = Graph::new_undirected();
        // Add nodes to the petgraph Graph (weights are not relevant here).
        let nodes: Vec<NodeIndex<u32>> = (0..n).map(|_| graph.add_node(())).collect();

        // Create a 1D drawing for simplicity in these tests.
        let mut drawing = DrawingEuclidean::new(&graph, 1);
        for i in 0..n {
            drawing.set(nodes[i], 0, initial_positions[i]);
        }

        let constraints = constraints_data
            .iter()
            .map(|&(l, r, g)| Constraint::new(l, r, g))
            .collect::<Vec<_>>();

        ConstraintGraph::new(&drawing, 0, &constraints)
    }

    #[test]
    fn test_variable_position() {
        let cg = create_test_graph(3, &[0.0, 5.0, 15.0], &[]);
        // Test initial variable positions (offset 0, block pos = initial pos).
        assert_eq!(cg.variable_position(0), 0.0);
        assert_eq!(cg.variable_position(1), 5.0);
        assert_eq!(cg.variable_position(2), 15.0);
    }

    #[test]
    fn test_violation() {
        let constraints_data = [(0, 1, 6.0), (1, 2, 8.0)]; // c0: 0+6<=1, c1: 1+8<=2
        let cg = create_test_graph(3, &[0.0, 5.0, 15.0], &constraints_data);

        // Test violation calc: c0: 0 + 6 - 5 = 1.0 (> 0, violated).
        assert!((cg.violation(0) - 1.0).abs() < 1e-6);
        // Test violation calc: c1: 5 + 8 - 15 = -2.0 (<= 0, not violated).
        assert!((cg.violation(1) - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_optimal_position_single_var() {
        let cg = create_test_graph(1, &[10.0], &[]);
        let desired_x = [12.0];
        // Test optimal position for a block with a single variable (should be desired pos - offset).
        // Initial offset is 0.
        assert!((cg.optimal_position(0, &desired_x) - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_optimal_position_multi_var() {
        // Setup: Block 0: {var 0 (off=-1), var 1 (off=2)}, Block 2: {var 2 (off=0)}
        let mut cg = create_test_graph(3, &[0.0, 1.0, 10.0], &[]); // Initial positions irrelevant here
        cg.variables[1].block = 0;
        cg.blocks[0].variables.insert(1);
        cg.blocks[1].variables.clear();
        cg.variables[0].offset = -1.0;
        cg.variables[1].offset = 2.0;
        // variables[2].offset remains 0

        let desired_x = [5.0, 8.0, 20.0]; // Desired positions for vars 0, 1, 2

        // Test optimal position for multi-variable block 0.
        // Expected = avg( (desired_x[j] - offset[j]) for j in block )
        // Expected = ( (5.0 - (-1.0)) + (8.0 - 2.0) ) / 2 = (6.0 + 6.0) / 2 = 6.0
        assert!((cg.optimal_position(0, &desired_x) - 6.0).abs() < 1e-6);

        // Test optimal position for single-variable block 2.
        // Expected = (desired_x[2] - offset[2]) / 1 = (20.0 - 0.0) / 1 = 20.0
        assert!((cg.optimal_position(2, &desired_x) - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_merge_block() {
        let constraints_data = [(0, 1, 5.0)]; // c0: 0 + 5 <= 1
        let mut cg = create_test_graph(3, &[0.0, 10.0, 20.0], &constraints_data); // Initial blocks: {0}, {1}, {2}

        // --- Setup ---
        // Simulate merging block 1 (r=1) into block 0 (l=0) due to constraint c0 (index 0).
        // Initial offsets are 0. Initial positions: pos(0)=0, pos(1)=10.

        // --- Action ---
        cg.merge_block(0, 1, 0);

        // --- Assertions ---
        // Block 1 is now empty.
        assert!(cg.blocks[1].variables.is_empty());
        assert!(cg.blocks[1].active.is_empty());

        // Block 0 contains vars 0 and 1.
        assert_eq!(cg.blocks[0].variables, [0, 1].iter().cloned().collect());
        // Variable 2 remains in block 2.
        assert_eq!(cg.blocks[2].variables, [2].iter().cloned().collect());

        // Variables 0 and 1 now belong to block 0.
        assert_eq!(cg.variables[0].block, 0);
        assert_eq!(cg.variables[1].block, 0);
        assert_eq!(cg.variables[2].block, 2); // Var 2 unaffected.

        // Check offsets relative to block 0's new reference.
        // d = offset_left + gap - offset_right = 0 + 5.0 - 0 = 5.0
        // offset[0] (originally in l) remains 0.
        // offset[1] (originally in r) becomes d + old_offset[1] = 5.0 + 0 = 5.0.
        assert!((cg.variables[0].offset - 0.0).abs() < 1e-6);
        assert!((cg.variables[1].offset - 5.0).abs() < 1e-6);
        assert!((cg.variables[2].offset - 0.0).abs() < 1e-6); // Var 2 unaffected.

        // Check block 0's new position (weighted average).
        // pos_l=0, pos_r=10, nvar_l=1, nvar_r=1, d=5.0
        // new_pos = (0*1 + (10-5)*1) / (1+1) = 5.0 / 2 = 2.5
        assert!((cg.blocks[0].position - 2.5).abs() < 1e-6);

        // Check block 0's active set contains only the merging constraint c0.
        assert_eq!(cg.blocks[0].active, [0].iter().cloned().collect());
    }

    #[test]
    fn test_connected() {
        // Graph: 0 --c0-- 1 --c1-- 2, plus 0 --c2-- 2. Node 3 is isolated.
        let constraints_data = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]; // c0, c1, c2
        let cg = create_test_graph(4, &[0., 1., 2., 3.], &constraints_data);

        // Scenario 1: Active constraints {c0, c2}. Expected connected from 0: {0, 1, 2}.
        let ac1: HashSet<usize> = [0, 2].iter().cloned().collect();
        let connected1 = cg.connected(0, &ac1);
        let expected1: HashSet<usize> = [0, 1, 2].iter().cloned().collect();
        assert_eq!(connected1, expected1);

        // Scenario 2: Active constraint {c1}. Expected connected from 1: {1, 2}.
        let ac2: HashSet<usize> = [1].iter().cloned().collect();
        let connected2 = cg.connected(1, &ac2);
        let expected2: HashSet<usize> = [1, 2].iter().cloned().collect();
        assert_eq!(connected2, expected2);

        // Scenario 3: No active constraints. Expected connected from 0: {0}.
        let ac3: HashSet<usize> = HashSet::new();
        let connected3 = cg.connected(0, &ac3);
        let expected3: HashSet<usize> = [0].iter().cloned().collect();
        assert_eq!(connected3, expected3);

        // Scenario 4: Start from isolated node 3. Expected: {3}.
        let connected4 = cg.connected(3, &ac1); // Use any active set
        let expected4: HashSet<usize> = [3].iter().cloned().collect();
        assert_eq!(connected4, expected4);
    }

    #[test]
    fn test_comp_path() {
        // Graph: 0-c0-1-c1-2-c3-4-c5-5, also 1-c2-3-c4-4
        let constraints_data = [
            (0, 1, 1.0), // c0
            (1, 2, 1.0), // c1
            (1, 3, 1.0), // c2
            (2, 4, 1.0), // c3
            (3, 4, 1.0), // c4
            (4, 5, 1.0), // c5
        ];
        let cg = create_test_graph(6, &[0., 1., 2., 2., 3., 4.], &constraints_data);

        // Test path finding with different active sets.
        // Scenario 1: Path 0->1->3->4->5 is active.
        let ac1: HashSet<usize> = [0, 2, 4, 5].iter().cloned().collect();
        assert_eq!(cg.comp_path(0, 5, &ac1), vec![0, 1, 3, 4, 5]);

        // Scenario 2: Path 0->1->2->4->5 is active.
        let ac2: HashSet<usize> = [0, 1, 3, 5].iter().cloned().collect();
        assert_eq!(cg.comp_path(0, 5, &ac2), vec![0, 1, 2, 4, 5]);

        // Scenario 3: Multiple paths possible, BFS finds one. Find path 3 -> 5.
        // Active: c0, c1, c2, c3, c5. Path 3->1->... or 3->4->...
        // BFS likely finds 3->1->2->4->5 or 3->4->5 depending on neighbor order.
        let ac3: HashSet<usize> = [0, 1, 2, 3, 5].iter().cloned().collect();
        let path3 = cg.comp_path(3, 5, &ac3);
        assert!(
            path3 == vec![3, 1, 2, 4, 5] || path3 == vec![3, 4, 5],
            "Path 3->5 should be one of the valid BFS paths"
        );

        // Scenario 4: Target unreachable with active set {c0, c1}.
        let ac4: HashSet<usize> = [0, 1].iter().cloned().collect();
        assert!(
            cg.comp_path(0, 5, &ac4).is_empty(),
            "Path 0->5 should be empty"
        );
    }

    #[test]
    fn test_comp_dfdv() {
        // Setup: Block 0 = {0, 1, 2}. Active: c0 (0->1, gap 1), c1 (1->2, gap 1).
        // Offsets based on active: off(0)=0, off(1)=1, off(2)=2. Block pos = 0.
        let constraints_data = [(0, 1, 1.0), (1, 2, 1.0)]; // c0, c1
        let mut cg = create_test_graph(3, &[0., 1., 2.], &constraints_data); // Initial pos match active state

        // Form block 0
        cg.variables[1].block = 0;
        cg.variables[2].block = 0;
        cg.blocks[0].variables.insert(1);
        cg.blocks[0].variables.insert(2);
        cg.blocks[1].variables.clear();
        cg.blocks[2].variables.clear();
        // Set offsets consistent with active c0, c1
        cg.variables[0].offset = 0.0;
        cg.variables[1].offset = 1.0;
        cg.variables[2].offset = 2.0;
        cg.blocks[0].position = 0.0;
        cg.blocks[0].active.insert(0); // c0 active
        cg.blocks[0].active.insert(1); // c1 active

        // Desired positions causing some tension.
        let desired_x = [0.5, 1.2, 2.8];
        // current pos: 0.0, 1.0, 2.0

        let mut lm = HashMap::new();
        // Call comp_dfdv starting from root var 0.
        let _root_dfdv = cg.comp_dfdv(0, &cg.blocks[0].active, None, &mut lm, &desired_x);

        // Assert that multipliers were computed for both active constraints.
        assert!(
            lm.contains_key(&0),
            "Lagrange multiplier for c0 should exist"
        );
        assert!(
            lm.contains_key(&1),
            "Lagrange multiplier for c1 should exist"
        );
        assert_eq!(
            lm.len(),
            2,
            "Only multipliers for active constraints c0 and c1 should exist"
        );

        // Assert specific expected lm values based on recursive definition.
        // dfdv(2) = pos(2)-x(2) = 2.0 - 2.8 = -0.8
        // lm(c1) = dfdv(2) = -0.8 (Inserted when v=1=left(c1), value is result of comp_dfdv(2,...))
        // dfdv(1) = (pos(1)-x(1)) + lm(c1) = (1.0 - 1.2) + (-0.8) = -1.0
        // lm(c0) = dfdv(1) = -1.0 (Inserted when v=0=left(c0), value is result of comp_dfdv(1,...))
        assert!((lm[&0] - (-1.0)).abs() < 1e-6, "LM for c0 mismatch");
        assert!((lm[&1] - (-0.8)).abs() < 1e-6, "LM for c1 mismatch"); // Corrected expected value to -0.8
    }

    #[test]
    fn test_expand_block() {
        // Setup: Block 0 = {0, 1, 2}. Active: c0 (0->1, gap 2), c1 (1->2, gap 2).
        // Offsets: off(0)=0, off(1)=2, off(2)=4. Block pos = 0. Actual pos: 0, 2, 4.
        // Violated constraint: c2 (0->2, gap 5). Violation = pos(0)+5-pos(2) = 0+5-4 = 1.0.
        let constraints_data = [(0, 1, 2.0), (1, 2, 2.0), (0, 2, 5.0)]; // c0, c1, c2
        let mut cg = create_test_graph(3, &[0., 2., 4.], &constraints_data); // Initial pos match active

        // Form block 0
        cg.variables[1].block = 0;
        cg.variables[2].block = 0;
        cg.blocks[0].variables.insert(1);
        cg.blocks[0].variables.insert(2);
        cg.blocks[1].variables.clear();
        cg.blocks[2].variables.clear();
        cg.variables[0].offset = 0.0;
        cg.variables[1].offset = 2.0;
        cg.variables[2].offset = 4.0;
        cg.blocks[0].position = 0.0;
        cg.blocks[0].active.insert(0); // c0 active
        cg.blocks[0].active.insert(1); // c1 active

        // Desired positions slightly perturbed to avoid lm tie.
        let desired_x = [0.0, 2.1, 4.0]; // Changed from [0., 2., 4.]
        let violating_c_idx = 2; // Index of constraint c2 (0->2, gap 5)

        // --- Pre-check ---
        assert!(
            (cg.violation(violating_c_idx) - 1.0).abs() < 1e-6,
            "Constraint c2 should be violated initially"
        );

        // --- Action ---
        // Expand block 0 to satisfy the internal violation of c2.
        cg.expand_block(0, violating_c_idx, &desired_x);

        // --- Assertions ---
        // 1. Determine the expected split constraint (sc).
        //    Path 0->2 is 0-c0->1-c1->2. Both c0, c1 are in ps.
        //    Compute lm: current=desired, so dfdv = 0 initially for all vars.
        //    dfdv(2)=0. lm(c1)=-dfdv(2)=0.
        //    dfdv(1)=dfdv(2)=0. lm(c0)=dfdv(1)=0.
        //    Min lm is 0. Tie broken by constraint index? Assume c0 (index 0) is chosen.
        let expected_sc = 0;

        // 2. Check the final active set. Should be {c1, c2} if sc=c0.
        let expected_active_set: HashSet<usize> = [1, 2].iter().cloned().collect();
        assert_eq!(
            cg.blocks[0].active, expected_active_set,
            "Active set should be {{c1, c2}} if c0 was split"
        );

        // 3. Check offset changes.
        //    If sc=c0, ac becomes {c1}. Component connected to right(c2)=2 is {1, 2} (via active c1).
        //    Offsets of {1, 2} should increase by violation(c2) = 1.0.
        //    New offset(1) = 2.0 + 1.0 = 3.0
        //    New offset(2) = 4.0 + 1.0 = 5.0
        assert!(
            (cg.variables[0].offset - 0.0).abs() < 1e-6,
            "Offset 0 should be unchanged"
        );
        assert!(
            (cg.variables[1].offset - 3.0).abs() < 1e-6,
            "Offset 1 should be 3.0"
        );
        assert!(
            (cg.variables[2].offset - 5.0).abs() < 1e-6,
            "Offset 2 should be 5.0"
        );

        // 4. Check block position recalculation.
        //    optimal_pos = ( (dx[0]-off[0]) + (dx[1]-off[1]) + (dx[2]-off[2]) ) / 3
        //              = ( (0.0-0.0) + (2.1-3.0) + (4.0-5.0) ) / 3
        //              = ( 0 - 0.9 - 1.0 ) / 3 = -1.9 / 3
        let expected_new_pos = -1.9 / 3.0; // Updated expected value
        assert!(
            (cg.blocks[0].position - expected_new_pos).abs() < 1e-6,
            "Block position recalculated incorrectly"
        );

        // 5. Check that the violation of c2 is now resolved (close to 0).
        //    New pos(0) = pos + off(0) = -1.9/3 + 0 = -1.9/3
        //    New pos(2) = pos + off(2) = -1.9/3 + 5 = (-1.9 + 15)/3 = 13.1/3
        //    New viol(c2) = pos(0) + gap(c2) - pos(2) = -1.9/3 + 5 - 13.1/3 = (-1.9 + 15 - 13.1)/3 = 0/3 = 0
        assert!(
            cg.violation(violating_c_idx).abs() < 1e-6, // Violation should still resolve to 0
            "Violation of c2 should be resolved"
        );

        // 6. Check that the split constraint sc=c0 now has slack (negative violation).
        //    New pos(1) = pos + off(1) = -1.9/3 + 3 = (-1.9 + 9)/3 = 7.1/3
        //    viol(c0) = pos(0) + gap(c0) - pos(1)
        //             = (-1.9/3) + 2 - (7.1/3)
        //             = (-1.9 + 6 - 7.1) / 3 = -3.0 / 3 = -1.0
        // Negative violation means c0 holds with slack, as expected after splitting it.
        assert!(
            cg.violation(expected_sc) < -1e-6, // Should still be negative
            "Split constraint c0 should now have slack"
        );
    }
}
