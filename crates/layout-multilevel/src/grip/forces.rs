use std::collections::VecDeque;

use egraph_core::{DeterministicRng, Scalar};

use super::GripConfig;
use super::filtration::GripComponent;

const MIN_SEPARATION_SQ_FACTOR: f64 = 1e-12;
const COINCIDENT_SQ_FACTOR: f64 = 1e-18;
const TREEMAP_PADDING: f64 = 0.08;

pub(super) fn initialize_coarsest_triangle(comp: &mut GripComponent, edge_length: f64) {
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

pub(super) fn step_grip_component(
    comp: &mut GripComponent,
    config: &GripConfig,
    rng: &mut DeterministicRng,
) {
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

pub(super) fn map_component_into_box<S: Scalar>(
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
