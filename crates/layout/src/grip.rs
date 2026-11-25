//! GRIP (Graph dRawing with Intelligent Placement) multilevel force-directed layout solver.
//!
//! Implements Gajer & Kobourov (2002), "GRIP: Graph Drawing with Intelligent Placement"
//! (*JGAA* 6(3), 203–224):
//! - Maximal Independent Set (`MIS`) vertex filtration $V_0 \supset V_1 \supset \dots \supset V_{k-1}$
//!   constructed via depth-$2^{i-1}$ partial BFS ("create-then-destroy") down to $|V_{k-1}| = 3$.
//! - Triangle initialization of $V_{k-1}$ from exact pairwise graph distances.
//! - Coarse-to-fine intelligent placement of $V_i \setminus V_{i+1}$ from up to 3 nearest
//!   ancestors in $V_{i+1}$ with local refinement.
//! - Localized Kamada–Kawai forces on coarse levels ($i > 0$), localized Fruchterman–Reingold
//!   forces on the finest level ($i = 0$), and per-vertex adaptive local temperature
//!   (`updateLocalTemp`).
//! - Connected-component decomposition with [`crate::group_in_a_box::squarified_treemap`]
//!   packing for disconnected graphs, and guards for `N <= 2`.

use std::collections::VecDeque;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, Scalar,
    SolverStatus, VisitStamp,
};

use crate::group_in_a_box::squarified_treemap;

const MIN_SEPARATION_SQ_FACTOR: f64 = 1e-12;
const COINCIDENT_SQ_FACTOR: f64 = 1e-18;
const TREEMAP_PADDING: f64 = 0.08;

/// Hyperparameters for [`GripSolver`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GripConfig {
    /// Minimum refinement rounds at the finest level $i = 0$ (`default: 5`).
    pub rounds_min: usize,
    /// Maximum refinement rounds at the coarsest level (`default: 30`).
    pub rounds_max: usize,
    /// Maximum size of each vertex's local neighborhood $|N_i(v)|$ (`default: 20`).
    pub local_neighborhood_size: usize,
    /// Repulsion scaling factor $s$ used in the finest-level ($i = 0$) FR force (`default: 0.05`).
    pub repulsion_scale: f64,
    /// Target unit edge length (`default: 1.0`).
    pub edge_length: f64,
    /// Local temperature adjustment step $r$ in `updateLocalTemp` (`default: 0.15`).
    pub temp_r: f64,
    /// Local temperature oscillation/acceleration multiplier $s$ (`default: 3.0`).
    pub temp_s: f64,
    /// Explicit deterministic RNG seed (`default: 42`).
    pub seed: u64,
}

impl Default for GripConfig {
    fn default() -> Self {
        Self {
            rounds_min: 5,
            rounds_max: 30,
            local_neighborhood_size: 20,
            repulsion_scale: 0.05,
            edge_length: 1.0,
            temp_r: 0.15,
            temp_s: 3.0,
            seed: 42,
        }
    }
}

impl GripConfig {
    /// Validates all GRIP hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if self.rounds_min > self.rounds_max {
            return Err(CoreError::invalid_input(format!(
                "rounds_min ({}) must be <= rounds_max ({})",
                self.rounds_min, self.rounds_max
            )));
        }
        if self.local_neighborhood_size == 0 {
            return Err(CoreError::invalid_input(
                "local_neighborhood_size must be >= 1",
            ));
        }
        if !self.repulsion_scale.is_finite() {
            return Err(CoreError::non_finite("GripConfig::repulsion_scale"));
        }
        if self.repulsion_scale < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "repulsion_scale must be >= 0.0, got {}",
                self.repulsion_scale
            )));
        }
        if !self.edge_length.is_finite() {
            return Err(CoreError::non_finite("GripConfig::edge_length"));
        }
        if self.edge_length <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "edge_length must be > 0.0, got {}",
                self.edge_length
            )));
        }
        if !self.temp_r.is_finite() {
            return Err(CoreError::non_finite("GripConfig::temp_r"));
        }
        if self.temp_r < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "temp_r must be >= 0.0, got {}",
                self.temp_r
            )));
        }
        if !self.temp_s.is_finite() {
            return Err(CoreError::non_finite("GripConfig::temp_s"));
        }
        if self.temp_s < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "temp_s must be >= 0.0, got {}",
                self.temp_s
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct GripLevel {
    /// Component-local vertex indices belonging to $V_i$.
    vertices: Vec<usize>,
    /// Precomputed local neighbors `(u_local, graph_dist)` packed in CSR order for `vertices`.
    nbr_indptr: Vec<usize>,
    nbr_nodes: Vec<usize>,
    nbr_dists: Vec<f64>,
    /// Newly introduced vertices $v \in V_i \setminus V_{i+1}$ and up to 3 nearest ancestors in $V_{i+1}$.
    new_vertices: Vec<usize>,
    anchor_indptr: Vec<usize>,
    anchor_nodes: Vec<usize>,
    anchor_dists: Vec<f64>,
    /// Scheduled refinement rounds for this level.
    rounds: usize,
}

#[derive(Debug, Clone)]
struct GripComponent {
    /// Global node IDs for vertices in this connected component.
    global_nodes: Vec<u32>,
    /// Component-local simple undirected CSR adjacency (`0..m`).
    adj_indptr: Vec<usize>,
    adj_indices: Vec<usize>,
    /// MIS filtration levels from finest (`0`) to coarsest (`k - 1`).
    levels: Vec<GripLevel>,
    /// Component-local 2D coordinates and per-vertex local temperature state.
    pos_x: Vec<f64>,
    pos_y: Vec<f64>,
    disp_x: Vec<f64>,
    disp_y: Vec<f64>,
    old_disp_x: Vec<f64>,
    old_disp_y: Vec<f64>,
    old_cos: Vec<f64>,
    heat: Vec<f64>,
    /// Current active level index (counts down from `levels.len() - 1` to `0`).
    current_level: usize,
    /// Rounds executed within `current_level`.
    round_in_level: usize,
    /// Whether `current_level` newly introduced vertices have been placed.
    level_initialized: bool,
    /// Whether this component has finished all levels.
    finished: bool,
}

/// Cooperative GRIP (Gajer & Kobourov, 2002) multilevel layout solver implementing [`IterativeSolver`].
#[derive(Debug, Clone)]
pub struct GripSolver<S: Scalar = f64> {
    config: GripConfig,
    num_nodes: usize,
    num_levels: usize,
    total_iterations: usize,
    components: Vec<GripComponent>,
    component_boxes: Vec<[f64; 4]>,
    rng: DeterministicRng,
    iter: usize,
    status: SolverStatus,
    _marker: std::marker::PhantomData<S>,
}

impl<S: Scalar> GripSolver<S> {
    /// Constructs a [`GripSolver`] from a [`CsrGraph`] and [`GripConfig`].
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: GripConfig) -> CoreResult<Self> {
        config.validate()?;
        let n = graph.num_nodes();
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
        }
        if let Some(ws) = graph.weights() {
            for &w in ws {
                let wf = w.to_f64();
                if !wf.is_finite() {
                    return Err(CoreError::non_finite("GripSolver::graph_weights"));
                }
                if wf < 0.0 {
                    return Err(CoreError::invalid_input(
                        "graph edge weights must be non-negative",
                    ));
                }
            }
        }

        let mut rng = DeterministicRng::from_seed_u64(config.seed);

        if n <= 1 {
            return Ok(Self {
                config,
                num_nodes: n,
                num_levels: n,
                total_iterations: 0,
                components: Vec::new(),
                component_boxes: Vec::new(),
                rng,
                iter: 0,
                status: SolverStatus::Initialized,
                _marker: std::marker::PhantomData,
            });
        }

        let (adj_indptr, adj_indices) = build_simple_undirected_adj(graph);
        let raw_components = extract_connected_components(n, &adj_indptr, &adj_indices);

        let mut components = Vec::with_capacity(raw_components.len());
        let mut max_levels = 0usize;
        let mut max_rounds_sum = 0usize;

        for comp_nodes in raw_components {
            let comp =
                build_grip_component(comp_nodes, &adj_indptr, &adj_indices, &config, &mut rng);
            max_levels = max_levels.max(comp.levels.len());
            let rounds_sum: usize = comp.levels.iter().map(|lvl| lvl.rounds).sum();
            max_rounds_sum = max_rounds_sum.max(rounds_sum);
            components.push(comp);
        }

        let component_boxes = if components.len() > 1 {
            let weights: Vec<f64> = components
                .iter()
                .map(|c| c.global_nodes.len() as f64)
                .collect();
            let side = (n as f64).sqrt() * config.edge_length * 2.5;
            squarified_treemap(&weights, -0.5 * side, -0.5 * side, side, side)?
        } else {
            Vec::new()
        };

        // Ensure `run()` executes at least one step for `n >= 2` so disconnected tiny
        // components or `rounds_max == 0` configs still emit valid initial/packed coordinates.
        let total_iterations = max_rounds_sum.max(1);

        Ok(Self {
            config,
            num_nodes: n,
            num_levels: max_levels,
            total_iterations,
            components,
            component_boxes,
            rng,
            iter: 0,
            status: SolverStatus::Initialized,
            _marker: std::marker::PhantomData,
        })
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &GripConfig {
        &self.config
    }

    /// Returns the expected number of vertices in [`Layout2d`].
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the maximum number of MIS filtration levels across connected components.
    #[inline]
    pub fn num_levels(&self) -> usize {
        self.num_levels
    }

    fn write_to_layout(&self, layout: &mut Layout2d<S>) {
        let (out_x, out_y) = layout.xy_mut();
        if self.components.len() == 1 {
            let comp = &self.components[0];
            for (local_idx, &global_u) in comp.global_nodes.iter().enumerate() {
                let ui = global_u as usize;
                let nx = S::from_f64(comp.pos_x[local_idx]);
                let ny = S::from_f64(comp.pos_y[local_idx]);
                if nx.is_finite() {
                    out_x[ui] = nx;
                }
                if ny.is_finite() {
                    out_y[ui] = ny;
                }
            }
            return;
        }

        for (comp, &[x_min, x_max, y_min, y_max]) in
            self.components.iter().zip(self.component_boxes.iter())
        {
            map_component_into_box(comp, x_min, x_max, y_min, y_max, out_x, out_y);
        }
    }
}

impl<S: Scalar> IterativeSolver<S> for GripSolver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "GripSolver::step",
            ));
        }
        layout.validate_finite()?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        if self.num_nodes <= 1 {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        let mut any_active = false;
        for comp in &mut self.components {
            if !comp.finished {
                step_grip_component(comp, &self.config, &mut self.rng);
                if !comp.finished {
                    any_active = true;
                }
            }
        }

        self.write_to_layout(layout);
        self.iter += 1;

        self.status = if !any_active || self.iter >= self.total_iterations {
            SolverStatus::Completed
        } else {
            SolverStatus::Running
        };
        Ok(self.status)
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.status
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.iter
    }

    #[inline]
    fn total_iterations(&self) -> usize {
        self.total_iterations
    }
}

fn build_simple_undirected_adj<W: Scalar>(graph: &CsrGraph<W>) -> (Vec<usize>, Vec<usize>) {
    let n = graph.num_nodes();
    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(graph.nnz() * 2);
    for u in 0..n {
        let u32_u = u as u32;
        for &v_u32 in graph.neighbors(u) {
            if u32_u != v_u32 {
                pairs.push((u32_u, v_u32));
                pairs.push((v_u32, u32_u));
            }
        }
    }
    pairs.sort_unstable();
    pairs.dedup();

    let mut indptr = vec![0usize; n + 1];
    for &(u, _) in &pairs {
        indptr[u as usize + 1] += 1;
    }
    for i in 0..n {
        indptr[i + 1] += indptr[i];
    }
    let mut indices = Vec::with_capacity(pairs.len());
    for &(_, v) in &pairs {
        indices.push(v as usize);
    }
    (indptr, indices)
}

fn extract_connected_components(
    n: usize,
    adj_indptr: &[usize],
    adj_indices: &[usize],
) -> Vec<Vec<u32>> {
    let mut visited = VisitStamp::new(n);
    let mut queue = VecDeque::new();
    let mut components = Vec::new();

    for start in 0..n {
        if !visited.visit(start) {
            continue;
        }
        let mut comp = Vec::new();
        queue.push_back(start);
        while let Some(u) = queue.pop_front() {
            comp.push(u as u32);
            for &v in &adj_indices[adj_indptr[u]..adj_indptr[u + 1]] {
                if visited.visit(v) {
                    queue.push_back(v);
                }
            }
        }
        comp.sort_unstable();
        components.push(comp);
    }

    components.sort_unstable_by(|a, b| b.len().cmp(&a.len()).then_with(|| a[0].cmp(&b[0])));
    components
}

fn build_grip_component(
    global_nodes: Vec<u32>,
    global_indptr: &[usize],
    global_indices: &[usize],
    config: &GripConfig,
    rng: &mut DeterministicRng,
) -> GripComponent {
    let m = global_nodes.len();
    let mut adj_indptr = vec![0usize; m + 1];
    let mut adj_indices = Vec::new();

    if m >= 2 {
        let mut global_to_local = vec![usize::MAX; global_indptr.len() - 1];
        for (local_idx, &g_u) in global_nodes.iter().enumerate() {
            global_to_local[g_u as usize] = local_idx;
        }
        for (local_u, &g_u) in global_nodes.iter().enumerate() {
            let gu = g_u as usize;
            let row_start = adj_indices.len();
            for &g_v in &global_indices[global_indptr[gu]..global_indptr[gu + 1]] {
                let lv = global_to_local[g_v];
                if lv != usize::MAX && lv != local_u {
                    adj_indices.push(lv);
                }
            }
            adj_indices[row_start..].sort_unstable();
            adj_indptr[local_u + 1] = adj_indices.len();
        }
    }

    let l = config.edge_length;
    if m <= 2 {
        let mut pos_x = vec![0.0; m];
        let pos_y = vec![0.0; m];
        if m == 2 {
            pos_x[0] = -0.5 * l;
            pos_x[1] = 0.5 * l;
        }
        return GripComponent {
            global_nodes,
            adj_indptr,
            adj_indices,
            levels: vec![GripLevel {
                vertices: (0..m).collect(),
                nbr_indptr: vec![0; m + 1],
                nbr_nodes: Vec::new(),
                nbr_dists: Vec::new(),
                new_vertices: Vec::new(),
                anchor_indptr: vec![0; m + 1],
                anchor_nodes: Vec::new(),
                anchor_dists: Vec::new(),
                rounds: 0,
            }],
            pos_x,
            pos_y,
            disp_x: vec![0.0; m],
            disp_y: vec![0.0; m],
            old_disp_x: vec![0.0; m],
            old_disp_y: vec![0.0; m],
            old_cos: vec![0.0; m],
            heat: vec![l / 6.0; m],
            current_level: 0,
            round_in_level: 0,
            level_initialized: true,
            finished: true,
        };
    }

    // Build MIS filtration V_0 > V_1 > ... > V_{k-1} with |V_{k-1}| == 3.
    let raw_levels = build_mis_filtration(m, &adj_indptr, &adj_indices, rng);
    let num_levels = raw_levels.len();

    let mut level_stamp = VisitStamp::new(m);
    let mut next_stamp = VisitStamp::new(m);
    let mut bfs_stamp = VisitStamp::new(m);
    let mut queue = VecDeque::with_capacity(m);
    let mut levels = Vec::with_capacity(num_levels);

    for i in 0..num_levels {
        let verts = &raw_levels[i];
        let v_len = verts.len();
        let nbr_cap = config.local_neighborhood_size.min(v_len.saturating_sub(1));

        level_stamp.clear();
        for &v in verts {
            level_stamp.mark(v);
        }

        let has_next = i + 1 < num_levels;
        if has_next {
            next_stamp.clear();
            for &v in &raw_levels[i + 1] {
                next_stamp.mark(v);
            }
        }

        let mut nbr_indptr = Vec::with_capacity(v_len + 1);
        nbr_indptr.push(0);
        let mut nbr_nodes = Vec::new();
        let mut nbr_dists = Vec::new();

        let mut new_vertices = Vec::new();
        let mut anchor_indptr = Vec::new();
        let mut anchor_nodes = Vec::new();
        let mut anchor_dists = Vec::new();
        if has_next {
            anchor_indptr.push(0);
        }

        for &v in verts {
            let needs_anchors = has_next && !next_stamp.is_visited(v);
            if needs_anchors {
                new_vertices.push(v);
            }

            bfs_stamp.clear();
            queue.clear();
            bfs_stamp.mark(v);
            queue.push_back((v, 0usize));

            let mut collected_nbrs = 0usize;
            let mut collected_anchors = 0usize;

            while let Some((u, d)) = queue.pop_front() {
                if u != v {
                    if collected_nbrs < nbr_cap && level_stamp.is_visited(u) {
                        nbr_nodes.push(u);
                        nbr_dists.push(d as f64);
                        collected_nbrs += 1;
                    }
                    if needs_anchors && collected_anchors < 3 && next_stamp.is_visited(u) {
                        anchor_nodes.push(u);
                        anchor_dists.push(d as f64);
                        collected_anchors += 1;
                    }
                    if collected_nbrs >= nbr_cap && (!needs_anchors || collected_anchors >= 3) {
                        break;
                    }
                }
                for &w in &adj_indices[adj_indptr[u]..adj_indptr[u + 1]] {
                    if bfs_stamp.visit(w) {
                        queue.push_back((w, d + 1));
                    }
                }
            }

            nbr_indptr.push(nbr_nodes.len());
            if needs_anchors {
                anchor_indptr.push(anchor_nodes.len());
            }
        }

        let rounds = if num_levels <= 1 {
            config.rounds_min
        } else {
            config.rounds_min
                + ((config.rounds_max - config.rounds_min) * i + (num_levels - 1) / 2)
                    / (num_levels - 1)
        };

        levels.push(GripLevel {
            vertices: verts.clone(),
            nbr_indptr,
            nbr_nodes,
            nbr_dists,
            new_vertices,
            anchor_indptr,
            anchor_nodes,
            anchor_dists,
            rounds,
        });
    }

    let mut comp = GripComponent {
        global_nodes,
        adj_indptr,
        adj_indices,
        levels,
        pos_x: vec![0.0; m],
        pos_y: vec![0.0; m],
        disp_x: vec![0.0; m],
        disp_y: vec![0.0; m],
        old_disp_x: vec![0.0; m],
        old_disp_y: vec![0.0; m],
        old_cos: vec![0.0; m],
        heat: vec![l / 6.0; m],
        current_level: num_levels - 1,
        round_in_level: 0,
        level_initialized: false,
        finished: false,
    };

    initialize_coarsest_triangle(&mut comp, l);
    comp
}

fn build_mis_filtration(
    m: usize,
    adj_indptr: &[usize],
    adj_indices: &[usize],
    rng: &mut DeterministicRng,
) -> Vec<Vec<usize>> {
    let mut levels: Vec<Vec<usize>> = Vec::new();
    let v0: Vec<usize> = (0..m).collect();
    levels.push(v0);

    if m == 3 {
        return levels;
    }

    let mut destroyed = VisitStamp::new(m);
    let mut bfs_stamp = VisitStamp::new(m);
    let mut queue = VecDeque::with_capacity(m);
    let mut dist = vec![usize::MAX; m];

    let mut level_idx = 1usize;
    loop {
        let prev = &levels[level_idx - 1];
        if prev.len() <= 3 {
            break;
        }

        let depth_limit = 1usize
            .checked_shl((level_idx - 1) as u32)
            .unwrap_or(m)
            .min(m);

        let mut order = prev.clone();
        rng.shuffle(&mut order);

        destroyed.clear();
        let mut next = Vec::new();
        for &v in &order {
            if destroyed.is_visited(v) {
                continue;
            }
            next.push(v);
            bfs_stamp.clear();
            queue.clear();
            bfs_stamp.mark(v);
            destroyed.mark(v);
            queue.push_back((v, 0usize));

            while let Some((u, d)) = queue.pop_front() {
                if d >= depth_limit {
                    continue;
                }
                for &w in &adj_indices[adj_indptr[u]..adj_indptr[u + 1]] {
                    if bfs_stamp.visit(w) {
                        destroyed.mark(w);
                        queue.push_back((w, d + 1));
                    }
                }
            }
        }

        if next.len() < 3 {
            // Promote farthest vertices from `prev \ next` via multi-source BFS so |V_k| == 3.
            dist.fill(usize::MAX);
            queue.clear();
            for &v in &next {
                dist[v] = 0;
                queue.push_back((v, 0usize));
            }
            while let Some((u, d)) = queue.pop_front() {
                for &w in &adj_indices[adj_indptr[u]..adj_indptr[u + 1]] {
                    if dist[w] == usize::MAX {
                        dist[w] = d + 1;
                        queue.push_back((w, d + 1));
                    }
                }
            }

            while next.len() < 3 {
                let mut best_v = None;
                let mut best_d = 0usize;
                for &cand in prev {
                    if dist[cand] > 0 && (best_v.is_none() || dist[cand] > best_d) {
                        best_v = Some(cand);
                        best_d = dist[cand];
                    }
                }
                if let Some(chosen) = best_v {
                    next.push(chosen);
                    dist[chosen] = 0;
                    queue.push_back((chosen, 0usize));
                    while let Some((u, d)) = queue.pop_front() {
                        for &w in &adj_indices[adj_indptr[u]..adj_indptr[u + 1]] {
                            if d + 1 < dist[w] {
                                dist[w] = d + 1;
                                queue.push_back((w, d + 1));
                            }
                        }
                    }
                } else {
                    break;
                }
            }
            next.sort_unstable();
            levels.push(next);
            break;
        }

        next.sort_unstable();
        let reached_target = next.len() == 3;
        levels.push(next);
        if reached_target {
            break;
        }
        level_idx += 1;
    }

    levels
}

fn initialize_coarsest_triangle(comp: &mut GripComponent, edge_length: f64) {
    let top_idx = comp.levels.len() - 1;
    let top = &comp.levels[top_idx];
    debug_assert_eq!(top.vertices.len(), 3);
    let a = top.vertices[0];
    let b = top.vertices[1];
    let c = top.vertices[2];

    let d_ab = bfs_pair_distance(a, b, &comp.adj_indptr, &comp.adj_indices) as f64;
    let d_ac = bfs_pair_distance(a, c, &comp.adj_indptr, &comp.adj_indices) as f64;
    let d_bc = bfs_pair_distance(b, c, &comp.adj_indptr, &comp.adj_indices) as f64;

    let l_ab = d_ab * edge_length;
    let l_ac = d_ac * edge_length;
    let l_bc = d_bc * edge_length;

    comp.pos_x[a] = 0.0;
    comp.pos_y[a] = 0.0;
    comp.pos_x[b] = l_ab;
    comp.pos_y[b] = 0.0;

    let denom = 2.0 * l_ab * l_ac;
    let cos_theta = if denom > 1e-18 {
        ((l_ab * l_ab + l_ac * l_ac - l_bc * l_bc) / denom).clamp(-1.0, 1.0)
    } else {
        0.5
    };
    let mut sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    if sin_theta < 1e-6 {
        // Keep triangle non-collinear when graph distances satisfy exact triangle equality.
        sin_theta = 0.05;
    }
    comp.pos_x[c] = l_ac * cos_theta;
    comp.pos_y[c] = l_ac * sin_theta;

    for &v in &[a, b, c] {
        comp.heat[v] = edge_length / 6.0;
        comp.old_cos[v] = 0.0;
        comp.old_disp_x[v] = 0.0;
        comp.old_disp_y[v] = 0.0;
    }
    comp.level_initialized = true;
}

fn bfs_pair_distance(src: usize, dst: usize, adj_indptr: &[usize], adj_indices: &[usize]) -> usize {
    if src == dst {
        return 0;
    }
    let m = adj_indptr.len() - 1;
    let mut visited = vec![false; m];
    let mut queue = VecDeque::new();
    visited[src] = true;
    queue.push_back((src, 0usize));
    while let Some((u, d)) = queue.pop_front() {
        for &w in &adj_indices[adj_indptr[u]..adj_indptr[u + 1]] {
            if w == dst {
                return d + 1;
            }
            if !visited[w] {
                visited[w] = true;
                queue.push_back((w, d + 1));
            }
        }
    }
    1
}

fn initialize_level_vertices(
    comp: &mut GripComponent,
    level_idx: usize,
    edge_length: f64,
    rng: &mut DeterministicRng,
) {
    let lvl = &comp.levels[level_idx];
    let l_sq = edge_length * edge_length;
    for (idx, &t) in lvl.new_vertices.iter().enumerate() {
        let a_start = lvl.anchor_indptr[idx];
        let a_end = lvl.anchor_indptr[idx + 1];
        let anchors = &lvl.anchor_nodes[a_start..a_end];
        let dists = &lvl.anchor_dists[a_start..a_end];

        let (mut px, mut py) = if anchors.is_empty() {
            (0.0, 0.0)
        } else {
            let mut sx = 0.0;
            let mut sy = 0.0;
            for &anc in anchors {
                sx += comp.pos_x[anc];
                sy += comp.pos_y[anc];
            }
            let inv = 1.0 / (anchors.len() as f64);
            (sx * inv, sy * inv)
        };

        // Add tiny deterministic jitter if the barycenter coincides with an anchor.
        let near_anchor = anchors.iter().any(|&anc| {
            let dx = px - comp.pos_x[anc];
            let dy = py - comp.pos_y[anc];
            dx * dx + dy * dy <= COINCIDENT_SQ_FACTOR * l_sq
        });
        if near_anchor || anchors.len() <= 1 {
            let angle = rng.next_f64() * std::f64::consts::TAU;
            let mag = 1e-3 * edge_length;
            px += mag * angle.cos();
            py += mag * angle.sin();
        }

        // Three local spring steps against the 3 nearest placed ancestors (Gajer 2002 §4).
        if !anchors.is_empty() {
            for _ in 0..3 {
                let mut fx = 0.0;
                let mut fy = 0.0;
                for (&anc, &dg) in anchors.iter().zip(dists.iter()) {
                    let dx = comp.pos_x[anc] - px;
                    let dy = comp.pos_y[anc] - py;
                    let target_sq = (dg * edge_length).powi(2).max(1e-18);
                    let d_sq = dx * dx + dy * dy;
                    let coeff = d_sq / target_sq - 1.0;
                    fx += coeff * dx;
                    fy += coeff * dy;
                }
                let norm = fx.hypot(fy);
                if norm > 1e-18 {
                    let step = norm.min(0.5 * edge_length) / norm;
                    px += step * fx;
                    py += step * fy;
                }
            }
        }

        comp.pos_x[t] = px;
        comp.pos_y[t] = py;
        comp.heat[t] = edge_length / 6.0;
        comp.old_cos[t] = 0.0;
        comp.old_disp_x[t] = 0.0;
        comp.old_disp_y[t] = 0.0;
    }
    comp.level_initialized = true;
}

fn step_grip_component(comp: &mut GripComponent, config: &GripConfig, rng: &mut DeterministicRng) {
    if comp.finished {
        return;
    }

    // Advance past empty/zero-round levels cleanly.
    while !comp.finished {
        let lvl_idx = comp.current_level;
        if !comp.level_initialized {
            initialize_level_vertices(comp, lvl_idx, config.edge_length, rng);
        }
        if comp.round_in_level < comp.levels[lvl_idx].rounds {
            break;
        }
        if comp.current_level == 0 {
            comp.finished = true;
            return;
        }
        comp.current_level -= 1;
        comp.round_in_level = 0;
        comp.level_initialized = false;
    }

    let lvl_idx = comp.current_level;
    let l = config.edge_length;
    let l_sq = l * l;
    let min_sep_sq = MIN_SEPARATION_SQ_FACTOR * l_sq;
    let coinc_sq = COINCIDENT_SQ_FACTOR * l_sq;
    let rep_scale = config.repulsion_scale;

    {
        let lvl = &comp.levels[lvl_idx];
        for (pos_in_lvl, &v) in lvl.vertices.iter().enumerate() {
            let vx = comp.pos_x[v];
            let vy = comp.pos_y[v];
            let mut fx = 0.0;
            let mut fy = 0.0;

            let n_start = lvl.nbr_indptr[pos_in_lvl];
            let n_end = lvl.nbr_indptr[pos_in_lvl + 1];

            if lvl_idx > 0 {
                // Coarse level: localized Kamada-Kawai force.
                for idx in n_start..n_end {
                    let u = lvl.nbr_nodes[idx];
                    let dg = lvl.nbr_dists[idx];
                    let dx = comp.pos_x[u] - vx;
                    let dy = comp.pos_y[u] - vy;
                    let d_sq = dx * dx + dy * dy;
                    let target_sq = (dg * l).powi(2).max(1e-18);
                    let coeff = d_sq / target_sq - 1.0;
                    fx += coeff * dx;
                    fy += coeff * dy;
                }
            } else {
                // Finest level (i == 0): localized Fruchterman-Reingold force.
                for &u in &comp.adj_indices[comp.adj_indptr[v]..comp.adj_indptr[v + 1]] {
                    let dx = comp.pos_x[u] - vx;
                    let dy = comp.pos_y[u] - vy;
                    let d_sq = dx * dx + dy * dy;
                    let coeff = d_sq / l_sq;
                    fx += coeff * dx;
                    fy += coeff * dy;
                }
                for &u in &lvl.nbr_nodes[n_start..n_end] {
                    let dx = vx - comp.pos_x[u];
                    let dy = vy - comp.pos_y[u];
                    let d_sq = dx * dx + dy * dy;
                    if d_sq <= coinc_sq {
                        let angle = rng.next_f64() * std::f64::consts::TAU;
                        let mag = rep_scale * l_sq / min_sep_sq.sqrt();
                        fx += mag * angle.cos();
                        fy += mag * angle.sin();
                    } else {
                        let coeff = (rep_scale * l_sq) / (d_sq + min_sep_sq);
                        fx += coeff * dx;
                        fy += coeff * dy;
                    }
                }
            }

            // Update local temperature (Gajer 2002 Fig. 6).
            let norm = fx.hypot(fy);
            if norm > 1e-18 {
                let old_x = comp.old_disp_x[v];
                let old_y = comp.old_disp_y[v];
                let old_norm = old_x.hypot(old_y);
                if old_norm > 1e-18 {
                    let cos_val = ((fx * old_x + fy * old_y) / (norm * old_norm)).clamp(-1.0, 1.0);
                    if comp.old_cos[v] * cos_val > 0.0 {
                        comp.heat[v] += cos_val * config.temp_r * config.temp_s * l;
                    } else {
                        comp.heat[v] += cos_val * config.temp_r * l;
                    }
                    comp.old_cos[v] = cos_val;
                }
                comp.heat[v] = comp.heat[v].clamp(1e-6 * l, 2.0 * l);
                let step_scale = comp.heat[v] / norm;
                comp.disp_x[v] = fx * step_scale;
                comp.disp_y[v] = fy * step_scale;
            } else {
                comp.disp_x[v] = 0.0;
                comp.disp_y[v] = 0.0;
            }
        }

        for &v in &lvl.vertices {
            let dx = comp.disp_x[v];
            let dy = comp.disp_y[v];
            comp.pos_x[v] += dx;
            comp.pos_y[v] += dy;
            comp.old_disp_x[v] = dx;
            comp.old_disp_y[v] = dy;
        }
    }

    comp.round_in_level += 1;
    while !comp.finished {
        let cur_idx = comp.current_level;
        if comp.round_in_level < comp.levels[cur_idx].rounds {
            break;
        }
        if cur_idx == 0 {
            comp.finished = true;
            break;
        }
        comp.current_level -= 1;
        comp.round_in_level = 0;
        comp.level_initialized = false;
        initialize_level_vertices(comp, comp.current_level, config.edge_length, rng);
    }
}

fn map_component_into_box<S: Scalar>(
    comp: &GripComponent,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    out_x: &mut [S],
    out_y: &mut [S],
) {
    let m = comp.global_nodes.len();
    if m == 0 {
        return;
    }
    let bw = x_max - x_min;
    let bh = y_max - y_min;
    let ix_min = x_min + TREEMAP_PADDING * bw;
    let ix_max = x_max - TREEMAP_PADDING * bw;
    let iy_min = y_min + TREEMAP_PADDING * bh;
    let iy_max = y_max - TREEMAP_PADDING * bh;
    let cx_box = 0.5 * (ix_min + ix_max);
    let cy_box = 0.5 * (iy_min + iy_max);

    if m == 1 {
        let ui = comp.global_nodes[0] as usize;
        out_x[ui] = S::from_f64(cx_box);
        out_y[ui] = S::from_f64(cy_box);
        return;
    }

    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for i in 0..m {
        min_x = min_x.min(comp.pos_x[i]);
        max_x = max_x.max(comp.pos_x[i]);
        min_y = min_y.min(comp.pos_y[i]);
        max_y = max_y.max(comp.pos_y[i]);
    }

    let span_x = max_x - min_x;
    let span_y = max_y - min_y;
    let pad_w = (ix_max - ix_min).max(0.0);
    let pad_h = (iy_max - iy_min).max(0.0);
    let cx_sub = 0.5 * (min_x + max_x);
    let cy_sub = 0.5 * (min_y + max_y);
    let eps = 1e-12;

    if span_x <= eps && span_y <= eps {
        for &g_u in &comp.global_nodes {
            let ui = g_u as usize;
            out_x[ui] = S::from_f64(cx_box);
            out_y[ui] = S::from_f64(cy_box);
        }
    } else if span_x <= eps {
        let scale = pad_h / span_y;
        for (i, &g_u) in comp.global_nodes.iter().enumerate() {
            let ui = g_u as usize;
            let my = (cy_box + (comp.pos_y[i] - cy_sub) * scale).clamp(iy_min, iy_max);
            out_x[ui] = S::from_f64(cx_box);
            out_y[ui] = S::from_f64(my);
        }
    } else if span_y <= eps {
        let scale = pad_w / span_x;
        for (i, &g_u) in comp.global_nodes.iter().enumerate() {
            let ui = g_u as usize;
            let mx = (cx_box + (comp.pos_x[i] - cx_sub) * scale).clamp(ix_min, ix_max);
            out_x[ui] = S::from_f64(mx);
            out_y[ui] = S::from_f64(cy_box);
        }
    } else {
        let scale = (pad_w / span_x).min(pad_h / span_y);
        for (i, &g_u) in comp.global_nodes.iter().enumerate() {
            let ui = g_u as usize;
            let mx = (cx_box + (comp.pos_x[i] - cx_sub) * scale).clamp(ix_min, ix_max);
            let my = (cy_box + (comp.pos_y[i] - cy_sub) * scale).clamp(iy_min, iy_max);
            out_x[ui] = S::from_f64(mx);
            out_y[ui] = S::from_f64(my);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_graph(rows: usize, cols: usize) -> CsrGraph<f64> {
        let n = rows * cols;
        let mut edges = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let u = (r * cols + c) as u32;
                if c + 1 < cols {
                    edges.push((u, u + 1));
                }
                if r + 1 < rows {
                    edges.push((u, u + cols as u32));
                }
            }
        }
        CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
    }

    fn path_graph(n: usize) -> CsrGraph<f64> {
        let edges: Vec<(u32, u32)> = (0..n.saturating_sub(1))
            .map(|i| (i as u32, (i + 1) as u32))
            .collect();
        CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
    }

    #[test]
    fn test_grip_config_validation() {
        let default_cfg = GripConfig::default();
        assert!(default_cfg.validate().is_ok());

        assert!(
            GripConfig {
                rounds_min: 10,
                rounds_max: 5,
                ..default_cfg
            }
            .validate()
            .is_err()
        );
        assert!(
            GripConfig {
                local_neighborhood_size: 0,
                ..default_cfg
            }
            .validate()
            .is_err()
        );
        assert!(
            GripConfig {
                edge_length: 0.0,
                ..default_cfg
            }
            .validate()
            .is_err()
        );
        assert!(
            GripConfig {
                repulsion_scale: f64::NAN,
                ..default_cfg
            }
            .validate()
            .is_err()
        );
        assert!(
            GripConfig {
                temp_r: -0.1,
                ..default_cfg
            }
            .validate()
            .is_err()
        );
        assert!(
            GripConfig {
                temp_s: f64::INFINITY,
                ..default_cfg
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn test_grip_step_run_bit_exact_f64_and_f32() {
        let g = grid_graph(5, 5);
        let cfg = GripConfig::default();

        // f64 parity
        let mut s_run = GripSolver::<f64>::new(&g, cfg).unwrap();
        let mut s_step = GripSolver::<f64>::new(&g, cfg).unwrap();
        assert!(s_run.num_levels() >= 2);
        let mut l_run = Layout2d::<f64>::zeros(g.num_nodes());
        let mut l_step = Layout2d::<f64>::zeros(g.num_nodes());

        assert_eq!(s_run.run(&mut l_run).unwrap(), SolverStatus::Completed);
        while !s_step.status().is_finished() {
            s_step.step(&mut l_step).unwrap();
        }
        assert_eq!(s_step.iteration(), s_run.iteration());
        assert_eq!(
            l_run.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            l_step.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            l_run.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            l_step.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert!(l_run.is_all_finite());

        // f32 parity
        let mut s32_run = GripSolver::<f32>::new(&g, cfg).unwrap();
        let mut s32_step = GripSolver::<f32>::new(&g, cfg).unwrap();
        let mut l32_run = Layout2d::<f32>::zeros(g.num_nodes());
        let mut l32_step = Layout2d::<f32>::zeros(g.num_nodes());
        assert_eq!(s32_run.run(&mut l32_run).unwrap(), SolverStatus::Completed);
        while !s32_step.status().is_finished() {
            s32_step.step(&mut l32_step).unwrap();
        }
        assert_eq!(
            l32_run.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            l32_step.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            l32_run.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            l32_step.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert!(l32_run.is_all_finite());
    }

    #[test]
    fn test_grip_edge_cases_n0_n1_n2_edgeless_disconnected_directed() {
        let cfg = GripConfig::default();

        // N = 0
        let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        let mut s0 = GripSolver::<f64>::new(&g0, cfg).unwrap();
        let mut l0 = Layout2d::<f64>::zeros(0);
        assert_eq!(s0.run(&mut l0).unwrap(), SolverStatus::Converged);

        // N = 1
        let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
        let mut s1 = GripSolver::<f64>::new(&g1, cfg).unwrap();
        let mut l1 = Layout2d::<f64>::from_xy(vec![3.5], vec![-1.2]).unwrap();
        assert_eq!(s1.run(&mut l1).unwrap(), SolverStatus::Converged);
        assert_eq!(l1.x()[0], 3.5);

        // N = 2 connected
        let g2 = path_graph(2);
        let mut s2 = GripSolver::<f64>::new(&g2, cfg).unwrap();
        let mut l2 = Layout2d::<f64>::zeros(2);
        assert_eq!(s2.run(&mut l2).unwrap(), SolverStatus::Completed);
        let d2 = (l2.x()[0] - l2.x()[1]).hypot(l2.y()[0] - l2.y()[1]);
        assert!((d2 - cfg.edge_length).abs() < 1e-12);

        // N = 3 path (collinear triangle inequality test)
        let g3 = path_graph(3);
        let mut s3 = GripSolver::<f64>::new(
            &g3,
            GripConfig {
                local_neighborhood_size: 1,
                ..cfg
            },
        )
        .unwrap();
        let mut l3 = Layout2d::<f64>::zeros(3);
        assert_eq!(s3.run(&mut l3).unwrap(), SolverStatus::Completed);
        assert!(l3.is_all_finite());

        // Edgeless graph (N = 5) packed without overlap
        let ge = CsrGraph::<f64>::from_edges(5, &[], None, false, false).unwrap();
        let mut se = GripSolver::<f64>::new(&ge, cfg).unwrap();
        let mut le = Layout2d::<f64>::zeros(5);
        assert_eq!(se.run(&mut le).unwrap(), SolverStatus::Completed);
        assert!(le.is_all_finite());
        for i in 0..5 {
            for j in (i + 1)..5 {
                let d = (le.x()[i] - le.x()[j]).hypot(le.y()[i] - le.y()[j]);
                assert!(d > 0.1, "isolated nodes {i} and {j} collapsed: d={d}");
            }
        }

        // Disconnected multi-component (2 paths of size 6 + 1 isolated node)
        let mut edges = Vec::new();
        for i in 0..5 {
            edges.push((i, i + 1));
        }
        for i in 6..11 {
            edges.push((i, i + 1));
        }
        let gd = CsrGraph::<f64>::from_edges(13, &edges, None, false, false).unwrap();
        let mut sd = GripSolver::<f64>::new(&gd, cfg).unwrap();
        let mut ld = Layout2d::<f64>::zeros(13);
        assert_eq!(sd.run(&mut ld).unwrap(), SolverStatus::Completed);
        assert!(ld.is_all_finite());

        // Directed graph with reciprocal edges
        let gdir = CsrGraph::<f64>::from_edges(
            4,
            &[(0, 1), (1, 0), (1, 2), (2, 3), (3, 0)],
            Some(&[1.0, 2.0, 1.5, 1.0, 1.0]),
            true,
            false,
        )
        .unwrap();
        let mut sdir = GripSolver::<f64>::new(&gdir, cfg).unwrap();
        let mut ldir = Layout2d::<f64>::zeros(4);
        assert_eq!(sdir.run(&mut ldir).unwrap(), SolverStatus::Completed);
        assert!(ldir.is_all_finite());

        // Non-finite input and dimension mismatch guards
        let mut l_bad = Layout2d::<f64>::from_xy(vec![f64::NAN; 4], vec![0.0; 4]).unwrap();
        let mut sdir2 = GripSolver::<f64>::new(&gdir, cfg).unwrap();
        assert!(sdir2.step(&mut l_bad).is_err());
        let mut l_wrong = Layout2d::<f64>::zeros(3);
        assert!(sdir2.step(&mut l_wrong).is_err());
    }
}
