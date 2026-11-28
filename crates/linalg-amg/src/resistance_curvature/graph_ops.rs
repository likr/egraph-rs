use std::collections::VecDeque;

use egraph_core::{CoreError, CoreResult, CsrGraph, EdgeWeight, VisitStamp};
use egraph_linalg::CsrMatrix;

pub(super) struct CanonicalEdges {
    pub(super) edge_u: Vec<u32>,
    pub(super) edge_v: Vec<u32>,
    pub(super) edge_lengths: Vec<f64>,
    pub(super) conductances: Vec<f64>,
}

pub(super) fn extract_canonical_edges<W: EdgeWeight>(
    graph: &CsrGraph<W>,
    weights_are_lengths: bool,
) -> CoreResult<CanonicalEdges> {
    let n = graph.num_nodes();
    if let Some(ws) = graph.weights() {
        for &w in ws {
            let wf = w.weight_to_f64();
            if !wf.is_finite() {
                return Err(CoreError::non_finite("CsrGraph::weights"));
            }
            if wf <= 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "graph edge weights must be strictly positive, got {wf}"
                )));
            }
        }
    }

    if n == 0 || graph.nnz() == 0 {
        return Ok(CanonicalEdges {
            edge_u: Vec::new(),
            edge_v: Vec::new(),
            edge_lengths: Vec::new(),
            conductances: Vec::new(),
        });
    }

    let weights_opt = graph.weights();
    let mut raw_edges: Vec<(u32, u32, f64)> = Vec::with_capacity(graph.nnz());
    for u in 0..n {
        let start = graph.indptr()[u] as usize;
        let end = graph.indptr()[u + 1] as usize;
        let u32_u = u as u32;
        for pos in start..end {
            let v32 = graph.indices()[pos];
            let v = v32 as usize;
            if v == u || v >= n {
                continue;
            }
            let val = match weights_opt {
                Some(ws) => ws[pos].weight_to_f64(),
                None => 1.0,
            };
            let (a, b) = if u32_u < v32 {
                (u32_u, v32)
            } else {
                (v32, u32_u)
            };
            raw_edges.push((a, b, val));
        }
    }

    raw_edges.sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));

    let mut edge_u = Vec::with_capacity(raw_edges.len());
    let mut edge_v = Vec::with_capacity(raw_edges.len());
    let mut edge_lengths = Vec::with_capacity(raw_edges.len());
    let mut conductances = Vec::with_capacity(raw_edges.len());

    let mut i = 0usize;
    while i < raw_edges.len() {
        let (u, v, mut val) = raw_edges[i];
        i += 1;
        while i < raw_edges.len() && raw_edges[i].0 == u && raw_edges[i].1 == v {
            val = val.max(raw_edges[i].2);
            i += 1;
        }
        let inv = 1.0 / val;
        if !inv.is_finite() || inv <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "edge weight {val} produces invalid reciprocal {inv}"
            )));
        }
        edge_u.push(u);
        edge_v.push(v);
        if weights_are_lengths {
            edge_lengths.push(val);
            conductances.push(inv);
        } else {
            conductances.push(val);
            edge_lengths.push(inv);
        }
    }

    Ok(CanonicalEdges {
        edge_u,
        edge_v,
        edge_lengths,
        conductances,
    })
}

pub(super) fn compute_components(
    num_nodes: usize,
    edge_u: &[u32],
    edge_v: &[u32],
) -> (usize, Vec<usize>, Vec<usize>) {
    if num_nodes == 0 {
        return (0, Vec::new(), Vec::new());
    }
    if edge_u.is_empty() {
        return (num_nodes, (0..num_nodes).collect(), vec![1; num_nodes]);
    }

    let mut deg = vec![0usize; num_nodes];
    for (&u, &v) in edge_u.iter().zip(edge_v.iter()) {
        deg[u as usize] += 1;
        deg[v as usize] += 1;
    }
    let mut indptr = Vec::with_capacity(num_nodes + 1);
    indptr.push(0usize);
    let mut acc = 0usize;
    for &d in &deg {
        acc += d;
        indptr.push(acc);
    }
    let mut cursor = indptr[..num_nodes].to_vec();
    let mut adj = vec![0u32; acc];
    for (&u, &v) in edge_u.iter().zip(edge_v.iter()) {
        let uu = u as usize;
        let vv = v as usize;
        adj[cursor[uu]] = v;
        cursor[uu] += 1;
        adj[cursor[vv]] = u;
        cursor[vv] += 1;
    }

    let mut visited = VisitStamp::new(num_nodes);
    let mut queue = VecDeque::new();
    let mut components = 0usize;
    let mut comp_id = vec![0usize; num_nodes];
    let mut comp_sizes = Vec::new();
    for start in 0..num_nodes {
        if !visited.visit(start) {
            continue;
        }
        let cid = components;
        components += 1;
        let mut size = 0usize;
        comp_id[start] = cid;
        queue.push_back(start);
        while let Some(u) = queue.pop_front() {
            size += 1;
            for &v32 in &adj[indptr[u]..indptr[u + 1]] {
                let v = v32 as usize;
                if visited.visit(v) {
                    comp_id[v] = cid;
                    queue.push_back(v);
                }
            }
        }
        comp_sizes.push(size);
    }
    (components, comp_id, comp_sizes)
}

pub(super) fn count_components(num_nodes: usize, edge_u: &[u32], edge_v: &[u32]) -> usize {
    compute_components(num_nodes, edge_u, edge_v).0
}

pub(super) fn degree_regularity_stats(
    num_nodes: usize,
    edge_u: &[u32],
    edge_v: &[u32],
) -> (f64, f64) {
    if num_nodes == 0 {
        return (0.0, 0.0);
    }
    let mut deg = vec![0usize; num_nodes];
    for (&u, &v) in edge_u.iter().zip(edge_v.iter()) {
        deg[u as usize] += 1;
        deg[v as usize] += 1;
    }
    let mut leaf_count = 0usize;
    let mut sum_deg = 0.0_f64;
    for &d in &deg {
        if d == 1 {
            leaf_count += 1;
        }
        sum_deg += d as f64;
    }
    let leaf_fraction = (leaf_count as f64) / (num_nodes as f64);
    let mean_deg = sum_deg / (num_nodes as f64);
    if mean_deg <= 0.0 {
        return (leaf_fraction, 0.0);
    }
    let mut var = 0.0_f64;
    for &d in &deg {
        let diff = (d as f64) - mean_deg;
        var += diff * diff;
    }
    let std_deg = (var / (num_nodes as f64)).sqrt();
    (leaf_fraction, std_deg / mean_deg)
}

/// Checks in $O(\alpha(G) |E|)$ time via Chiba-Nishizeki triangle counting whether the
/// non-isolated subgraph forms a closed triangulated spherical surface (every edge belongs to
/// $\ge 2$ triangles, no leaves/degree-2 vertices, and positive Euler characteristic $V - E + T > 0$).
///
/// This distinguishes closed 3D triangulated spheres (e.g. `can_96`, `sierpinski3d`, icospheres)
/// from triangulated tori ($V - E + T = 0$), 4-regular tori ($T = 0$), and open planar meshes.
pub(super) fn is_closed_spherical_surface(
    num_nodes: usize,
    edge_u: &[u32],
    edge_v: &[u32],
    leaf_fraction: f64,
) -> bool {
    let m = edge_u.len();
    if num_nodes < 4 || m == 0 || leaf_fraction > 0.0 {
        return false;
    }

    let mut deg = vec![0usize; num_nodes];
    for (&u, &v) in edge_u.iter().zip(edge_v.iter()) {
        deg[u as usize] += 1;
        deg[v as usize] += 1;
    }

    let mut active_nodes = 0usize;
    for &d in &deg {
        if d > 0 {
            if d < 3 {
                // Edges incident to degree-1 or degree-2 vertices cannot lie in >= 2 triangles.
                return false;
            }
            active_nodes += 1;
        }
    }

    let mut out_deg = vec![0usize; num_nodes];
    for (&u, &v) in edge_u.iter().zip(edge_v.iter()) {
        let ui = u as usize;
        let vi = v as usize;
        if (deg[ui], ui) < (deg[vi], vi) {
            out_deg[ui] += 1;
        } else {
            out_deg[vi] += 1;
        }
    }

    let mut out_indptr = Vec::with_capacity(num_nodes + 1);
    out_indptr.push(0usize);
    let mut total_out = 0usize;
    for &od in &out_deg {
        total_out += od;
        out_indptr.push(total_out);
    }
    let mut cursor = out_indptr[..num_nodes].to_vec();
    let mut out_targets = vec![(0u32, 0usize); total_out];
    for (edge_idx, (&u, &v)) in edge_u.iter().zip(edge_v.iter()).enumerate() {
        let ui = u as usize;
        let vi = v as usize;
        if (deg[ui], ui) < (deg[vi], vi) {
            let pos = cursor[ui];
            out_targets[pos] = (v, edge_idx);
            cursor[ui] += 1;
        } else {
            let pos = cursor[vi];
            out_targets[pos] = (u, edge_idx);
            cursor[vi] += 1;
        }
    }

    let mut visited = VisitStamp::new(num_nodes);
    let mut edge_to_w = vec![0usize; num_nodes];
    let mut edge_tri = vec![0usize; m];
    let mut num_triangles = 0usize;

    for u in 0..num_nodes {
        let u_start = out_indptr[u];
        let u_end = out_indptr[u + 1];
        if u_start == u_end {
            continue;
        }
        visited.clear();
        for &(w, e_uw) in &out_targets[u_start..u_end] {
            let wi = w as usize;
            visited.mark(wi);
            edge_to_w[wi] = e_uw;
        }
        for &(v, e_uv) in &out_targets[u_start..u_end] {
            let vi = v as usize;
            for &(w, e_vw) in &out_targets[out_indptr[vi]..out_indptr[vi + 1]] {
                let wi = w as usize;
                if visited.is_visited(wi) {
                    num_triangles += 1;
                    edge_tri[e_uv] += 1;
                    edge_tri[e_vw] += 1;
                    edge_tri[edge_to_w[wi]] += 1;
                }
            }
        }
    }

    if edge_tri.iter().any(|&c| c < 2) {
        return false;
    }
    let euler = (active_nodes as isize) - (m as isize) + (num_triangles as isize);
    euler > 0
}

pub(super) fn build_combinatorial_laplacian_from_edges(
    num_nodes: usize,
    edge_u: &[u32],
    edge_v: &[u32],
    conductances: &[f64],
) -> CoreResult<CsrMatrix> {
    if num_nodes == 0 {
        return CsrMatrix::from_csr(0, vec![0], Vec::new(), Vec::new());
    }

    let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); num_nodes];
    let mut diag = vec![0.0_f64; num_nodes];
    for e in 0..edge_u.len() {
        let u = edge_u[e];
        let v = edge_v[e];
        let w = conductances[e];
        rows[u as usize].push((v, -w));
        rows[v as usize].push((u, -w));
        diag[u as usize] += w;
        diag[v as usize] += w;
    }

    let mut indptr = Vec::with_capacity(num_nodes + 1);
    indptr.push(0u32);
    let mut indices = Vec::with_capacity(edge_u.len() * 2 + num_nodes);
    let mut values = Vec::with_capacity(edge_u.len() * 2 + num_nodes);

    for (u, row) in rows.iter_mut().enumerate() {
        let u32_u = u as u32;
        let d = diag[u];
        if d > 0.0 {
            row.push((u32_u, d));
        }
        row.sort_unstable_by_key(|&(col, _)| col);
        for &(col, val) in row.iter() {
            indices.push(col);
            values.push(val);
        }
        indptr.push(indices.len() as u32);
    }

    CsrMatrix::from_csr(num_nodes, indptr, indices, values)
}

pub(super) fn build_canonical_graph_f64(
    num_nodes: usize,
    edge_u: &[u32],
    edge_v: &[u32],
    weights: &[f64],
) -> CoreResult<CsrGraph<f64>> {
    let edges: Vec<(u32, u32)> = edge_u.iter().copied().zip(edge_v.iter().copied()).collect();
    CsrGraph::<f64>::from_edges(num_nodes, &edges, Some(weights), false, false)
}

pub(super) fn build_canonical_graph_f32(
    num_nodes: usize,
    edge_u: &[u32],
    edge_v: &[u32],
    weights: &[f64],
) -> CoreResult<CsrGraph<f32>> {
    let edges: Vec<(u32, u32)> = edge_u.iter().copied().zip(edge_v.iter().copied()).collect();
    let weights_f32: Vec<f32> = weights
        .iter()
        .map(|&v| {
            let vf = v as f32;
            if vf.is_infinite() {
                f32::MAX
            } else if vf <= 0.0 {
                f32::MIN_POSITIVE
            } else {
                vf
            }
        })
        .collect();
    CsrGraph::<f32>::from_edges(num_nodes, &edges, Some(&weights_f32), false, false)
}
