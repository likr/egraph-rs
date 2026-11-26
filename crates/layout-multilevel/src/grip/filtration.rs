use std::collections::VecDeque;

use egraph_core::{CsrGraph, DeterministicRng, Scalar, VisitStamp};

use super::GripConfig;
use super::forces::initialize_coarsest_triangle;

#[derive(Debug, Clone)]
pub(super) struct GripLevel {
    /// Component-local vertex indices belonging to $V_i$.
    pub(super) vertices: Vec<usize>,
    /// Precomputed local neighbors `(u_local, graph_dist)` packed in CSR order for `vertices`.
    pub(super) nbr_indptr: Vec<usize>,
    pub(super) nbr_nodes: Vec<usize>,
    pub(super) nbr_dists: Vec<f64>,
    /// Newly introduced vertices $v \in V_i \setminus V_{i+1}$ and up to 3 nearest ancestors in $V_{i+1}$.
    pub(super) new_vertices: Vec<usize>,
    pub(super) anchor_indptr: Vec<usize>,
    pub(super) anchor_nodes: Vec<usize>,
    pub(super) anchor_dists: Vec<f64>,
    /// Scheduled refinement rounds for this level.
    pub(super) rounds: usize,
}

#[derive(Debug, Clone)]
pub(super) struct GripComponent {
    /// Global node IDs for vertices in this connected component.
    pub(super) global_nodes: Vec<u32>,
    /// Component-local simple undirected CSR adjacency (`0..m`).
    pub(super) adj_indptr: Vec<usize>,
    pub(super) adj_indices: Vec<usize>,
    /// MIS filtration levels from finest (`0`) to coarsest (`k - 1`).
    pub(super) levels: Vec<GripLevel>,
    /// Component-local 2D coordinates and per-vertex local temperature state.
    pub(super) pos_x: Vec<f64>,
    pub(super) pos_y: Vec<f64>,
    pub(super) disp_x: Vec<f64>,
    pub(super) disp_y: Vec<f64>,
    pub(super) old_disp_x: Vec<f64>,
    pub(super) old_disp_y: Vec<f64>,
    pub(super) old_cos: Vec<f64>,
    pub(super) heat: Vec<f64>,
    /// Current active level index (counts down from `levels.len() - 1` to `0`).
    pub(super) current_level: usize,
    /// Rounds executed within `current_level`.
    pub(super) round_in_level: usize,
    /// Whether `current_level` newly introduced vertices have been placed.
    pub(super) level_initialized: bool,
    /// Whether this component has finished all levels.
    pub(super) finished: bool,
}

pub(super) fn build_simple_undirected_adj<W: Scalar>(
    graph: &CsrGraph<W>,
) -> (Vec<usize>, Vec<usize>) {
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

pub(super) fn extract_connected_components(
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

pub(super) fn build_grip_component(
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
