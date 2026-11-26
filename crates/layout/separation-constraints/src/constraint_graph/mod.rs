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
/// See IPSEP-COLA paper \[2\] for details on the algorithm.
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
    /// Corresponds to the initialization described at the end of Section 3.2 \[1\].
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
    /// Corresponds to `split_blocks(x)` in Figure 10 \[2\].
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
    /// Corresponds to the `project(C)` procedure in Figure 9 \[2\].
    ///
    /// # Arguments
    ///
    /// * `x` - Variable positions (coordinate in the current dimension) to be projected.
    ///   Modified **in-place** to satisfy all constraints.
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
    for (i, val) in x.into_iter().enumerate() {
        *drawing.raw_entry_mut(i).nth_mut(k) = val;
    }
}

#[cfg(test)]
mod tests;
