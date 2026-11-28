//! Deterministic procedural synthetic graph and manifold-mesh generators.
//!
//! All generators rely solely on `egraph_core` (`CsrGraph`, `DeterministicRng`) and work
//! identically with `--no-default-features` across Native Rust, Python, and WebAssembly.

use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng};

#[inline]
fn canonical_edge(u: u32, v: u32) -> (u32, u32) {
    if u < v { (u, v) } else { (v, u) }
}

#[inline]
fn validate_unit_prob(val: f64, name: &'static str) -> CoreResult<()> {
    if !val.is_finite() {
        return Err(CoreError::non_finite(name));
    }
    if !(0.0..=1.0).contains(&val) {
        return Err(CoreError::invalid_input(format!(
            "{name} must be in [0.0, 1.0], got {val}"
        )));
    }
    Ok(())
}

/// Generates a deterministic undirected weighted graph on `num_nodes` vertices with edge weights
/// uniformly sampled in `[min_weight, max_weight]` via [`DeterministicRng`].
///
/// When `num_nodes >= 2` and `avg_degree >= 2`, includes a cycle backbone so the graph is
/// guaranteed connected before adding deterministic random chords.
pub fn generate_weighted_graph(
    num_nodes: usize,
    avg_degree: usize,
    min_weight: f64,
    max_weight: f64,
    seed: u64,
) -> CoreResult<CsrGraph<f64>> {
    if u32::try_from(num_nodes).is_err() {
        return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
    }
    if !min_weight.is_finite() {
        return Err(CoreError::non_finite("generate_weighted_graph::min_weight"));
    }
    if !max_weight.is_finite() {
        return Err(CoreError::non_finite("generate_weighted_graph::max_weight"));
    }
    if min_weight <= 0.0 || max_weight < min_weight {
        return Err(CoreError::invalid_input(format!(
            "weights must satisfy 0 < min_weight <= max_weight, got [{min_weight}, {max_weight}]"
        )));
    }
    if num_nodes <= 1 || avg_degree == 0 {
        return CsrGraph::from_unweighted_edges(num_nodes, &[], false, false);
    }

    let mut rng = DeterministicRng::from_seed_u64(seed);
    let mut edges: Vec<(u32, u32)> = Vec::new();
    let span = max_weight - min_weight;

    if num_nodes >= 3 && avg_degree >= 2 {
        for u in 0..num_nodes {
            let v = (u + 1) % num_nodes;
            edges.push(canonical_edge(u as u32, v as u32));
        }
    } else {
        for u in 0..(num_nodes - 1) {
            edges.push((u as u32, (u + 1) as u32));
        }
    }

    let target_edges = (num_nodes.saturating_mul(avg_degree) / 2).max(edges.len());
    let max_possible = num_nodes.saturating_mul(num_nodes - 1) / 2;
    let target = target_edges.min(max_possible);
    let extra_trials = target.saturating_sub(edges.len()).saturating_mul(3);
    for _ in 0..extra_trials {
        if edges.len() >= target {
            break;
        }
        let u = rng.gen_range_usize(num_nodes) as u32;
        let v = rng.gen_range_usize(num_nodes) as u32;
        if u != v {
            edges.push(canonical_edge(u, v));
        }
    }

    edges.sort_unstable();
    edges.dedup();
    let weights: Vec<f64> = (0..edges.len())
        .map(|_| min_weight + span * rng.next_f64())
        .collect();
    CsrGraph::from_edges(num_nodes, &edges, Some(&weights), false, false)
}

/// Generates a deterministic layered Directed Acyclic Graph (DAG) with `num_layers * nodes_per_layer`
/// vertices where every directed edge points strictly forward (`layer(u) < layer(v)`).
pub fn generate_layered_dag(
    num_layers: usize,
    nodes_per_layer: usize,
    edge_prob: f64,
    seed: u64,
) -> CoreResult<CsrGraph<f64>> {
    validate_unit_prob(edge_prob, "generate_layered_dag::edge_prob")?;
    let num_nodes = num_layers
        .checked_mul(nodes_per_layer)
        .ok_or_else(|| CoreError::invalid_input("DAG node count overflow"))?;
    if u32::try_from(num_nodes).is_err() {
        return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
    }
    if num_layers <= 1 || nodes_per_layer == 0 {
        return CsrGraph::from_unweighted_edges(num_nodes, &[], true, false);
    }

    let mut rng = DeterministicRng::from_seed_u64(seed);
    let mut edges: Vec<(u32, u32)> = Vec::new();
    for l1 in 0..(num_layers - 1) {
        let base1 = l1 * nodes_per_layer;
        for i in 0..nodes_per_layer {
            let u = (base1 + i) as u32;
            for l2 in (l1 + 1)..num_layers {
                let base2 = l2 * nodes_per_layer;
                for j in 0..nodes_per_layer {
                    if rng.next_f64() < edge_prob {
                        edges.push((u, (base2 + j) as u32));
                    }
                }
            }
        }
    }
    CsrGraph::from_unweighted_edges(num_nodes, &edges, true, false)
}

/// Generates a deterministic Stochastic Block Model (SBM) undirected graph alongside its
/// ground-truth community labels (`(CsrGraph<f64>, Vec<u32>)`).
pub fn generate_sbm_graph(
    block_sizes: &[usize],
    p_intra: f64,
    p_inter: f64,
    seed: u64,
) -> CoreResult<(CsrGraph<f64>, Vec<u32>)> {
    validate_unit_prob(p_intra, "generate_sbm_graph::p_intra")?;
    validate_unit_prob(p_inter, "generate_sbm_graph::p_inter")?;

    let mut num_nodes = 0usize;
    for &sz in block_sizes {
        num_nodes = num_nodes
            .checked_add(sz)
            .ok_or_else(|| CoreError::invalid_input("SBM node count overflow"))?;
    }
    if u32::try_from(num_nodes).is_err() {
        return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
    }

    let mut labels = Vec::with_capacity(num_nodes);
    for (block_id, &sz) in block_sizes.iter().enumerate() {
        let bid = u32::try_from(block_id)
            .map_err(|_| CoreError::invalid_input("block count exceeds u32::MAX"))?;
        labels.resize(labels.len() + sz, bid);
    }
    if num_nodes <= 1 {
        let g = CsrGraph::from_unweighted_edges(num_nodes, &[], false, false)?;
        return Ok((g, labels));
    }

    let mut rng = DeterministicRng::from_seed_u64(seed);
    let mut edges: Vec<(u32, u32)> = Vec::new();
    for u in 0..num_nodes {
        let lu = labels[u];
        let u_u32 = u as u32;
        for (v, &lv) in labels.iter().enumerate().skip(u + 1) {
            let p = if lu == lv { p_intra } else { p_inter };
            if rng.next_f64() < p {
                edges.push((u_u32, v as u32));
            }
        }
    }
    let graph = CsrGraph::from_unweighted_edges(num_nodes, &edges, false, false)?;
    Ok((graph, labels))
}

/// Generates a periodic 2D torus mesh of size `rows x cols` (`rows >= 3`, `cols >= 3`),
/// either 4-regular (`triangulated = false`) or 6-regular triangulated (`triangulated = true`).
pub fn generate_torus_grid(
    rows: usize,
    cols: usize,
    triangulated: bool,
) -> CoreResult<CsrGraph<f64>> {
    if rows < 3 || cols < 3 {
        return Err(CoreError::invalid_input(format!(
            "torus grid requires rows >= 3 and cols >= 3 to stay simple, got {rows}x{cols}"
        )));
    }
    let num_nodes = rows
        .checked_mul(cols)
        .ok_or_else(|| CoreError::invalid_input("torus grid size overflow"))?;
    if u32::try_from(num_nodes).is_err() {
        return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
    }

    let mut edges = Vec::with_capacity(num_nodes * if triangulated { 3 } else { 2 });
    for r in 0..rows {
        for c in 0..cols {
            let u = (r * cols + c) as u32;
            let right = (r * cols + (c + 1) % cols) as u32;
            let down = (((r + 1) % rows) * cols + c) as u32;
            edges.push(canonical_edge(u, right));
            edges.push(canonical_edge(u, down));
            if triangulated {
                let diag = (((r + 1) % rows) * cols + (c + 1) % cols) as u32;
                edges.push(canonical_edge(u, diag));
            }
        }
    }
    edges.sort_unstable();
    edges.dedup();
    CsrGraph::from_unweighted_edges(num_nodes, &edges, false, false)
}

/// Generates a closed genus-0 triangulated spherical icosphere by subdividing a regular
/// icosahedron `subdivisions` times ($V = 10 \cdot 4^k + 2$, $E = 3V - 6$, capped at `k <= 7`).
pub fn generate_spherical_icosphere(subdivisions: usize) -> CoreResult<CsrGraph<f64>> {
    if subdivisions > 7 {
        return Err(CoreError::invalid_input(format!(
            "subdivisions must be <= 7 to avoid excessive memory usage, got {subdivisions}"
        )));
    }
    // 12 icosahedron vertices and 20 CCW faces.
    let mut tris: Vec<[u32; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    let mut num_nodes = 12u32;

    for _ in 0..subdivisions {
        // Deduplicate subdivided edges via a sorted edge-key vector instead of HashMap.
        let mut edge_keys: Vec<(u32, u32)> = Vec::with_capacity(tris.len() * 3);
        for &[a, b, c] in &tris {
            edge_keys.push(if a < b { (a, b) } else { (b, a) });
            edge_keys.push(if b < c { (b, c) } else { (c, b) });
            edge_keys.push(if c < a { (c, a) } else { (a, c) });
        }
        edge_keys.sort_unstable();
        edge_keys.dedup();

        let base_mid = num_nodes;
        num_nodes = num_nodes
            .checked_add(edge_keys.len() as u32)
            .ok_or_else(|| CoreError::invalid_csr("icosphere vertex count exceeds u32::MAX"))?;

        let lookup_mid = |a: u32, b: u32| -> u32 {
            let key = if a < b { (a, b) } else { (b, a) };
            let idx = edge_keys
                .binary_search(&key)
                .expect("triangle edge must exist in sorted edge table");
            base_mid + idx as u32
        };

        let mut next_tris = Vec::with_capacity(tris.len() * 4);
        for &[a, b, c] in &tris {
            let ab = lookup_mid(a, b);
            let bc = lookup_mid(b, c);
            let ca = lookup_mid(c, a);
            next_tris.push([a, ab, ca]);
            next_tris.push([b, bc, ab]);
            next_tris.push([c, ca, bc]);
            next_tris.push([ab, bc, ca]);
        }
        tris = next_tris;
    }

    let mut edges = Vec::with_capacity(tris.len() * 3);
    for &[a, b, c] in &tris {
        edges.push(canonical_edge(a, b));
        edges.push(canonical_edge(b, c));
        edges.push(canonical_edge(c, a));
    }
    edges.sort_unstable();
    edges.dedup();
    CsrGraph::from_unweighted_edges(num_nodes as usize, &edges, false, false)
}

/// Generates a closed UV spherical grid mesh with north/south poles and `latitudes` intermediate
/// rings of `longitudes` vertices (`latitudes >= 1`, `longitudes >= 3`).
pub fn generate_spherical_grid(latitudes: usize, longitudes: usize) -> CoreResult<CsrGraph<f64>> {
    if latitudes == 0 || longitudes < 3 {
        return Err(CoreError::invalid_input(format!(
            "spherical grid requires latitudes >= 1 and longitudes >= 3, got {latitudes}x{longitudes}"
        )));
    }
    let ring_nodes = latitudes
        .checked_mul(longitudes)
        .ok_or_else(|| CoreError::invalid_input("spherical grid size overflow"))?;
    let num_nodes = ring_nodes
        .checked_add(2)
        .ok_or_else(|| CoreError::invalid_input("spherical grid size overflow"))?;
    if u32::try_from(num_nodes).is_err() {
        return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
    }

    let north = 0u32;
    let south = (num_nodes - 1) as u32;
    let mut edges = Vec::new();
    for j in 0..longitudes {
        edges.push(canonical_edge(north, (1 + j) as u32));
        edges.push(canonical_edge(
            south,
            (1 + (latitudes - 1) * longitudes + j) as u32,
        ));
    }
    for i in 0..latitudes {
        let base = 1 + i * longitudes;
        for j in 0..longitudes {
            let u = (base + j) as u32;
            let right = (base + (j + 1) % longitudes) as u32;
            edges.push(canonical_edge(u, right));
            if i + 1 < latitudes {
                let down = (1 + (i + 1) * longitudes + j) as u32;
                edges.push(canonical_edge(u, down));
            }
        }
    }
    edges.sort_unstable();
    edges.dedup();
    CsrGraph::from_unweighted_edges(num_nodes, &edges, false, false)
}

/// Generates a deterministic regular hyperbolic branching tree with `branching_factor` children
/// per internal node up to `depth` levels (`depth = 0` yields a single root vertex).
pub fn generate_hyperbolic_tree(
    branching_factor: usize,
    depth: usize,
) -> CoreResult<CsrGraph<f64>> {
    if branching_factor == 0 && depth > 0 {
        return Err(CoreError::invalid_input(
            "branching_factor must be >= 1 when depth > 0",
        ));
    }
    let mut total = 1usize;
    let mut level_size = 1usize;
    for _ in 0..depth {
        level_size = level_size
            .checked_mul(branching_factor)
            .ok_or_else(|| CoreError::invalid_input("hyperbolic tree node count overflow"))?;
        total = total
            .checked_add(level_size)
            .ok_or_else(|| CoreError::invalid_input("hyperbolic tree node count overflow"))?;
    }
    if u32::try_from(total).is_err() {
        return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
    }
    if total == 1 {
        return CsrGraph::from_unweighted_edges(1, &[], false, false);
    }

    let mut edges = Vec::with_capacity(total - 1);
    let mut next_child = 1u32;
    let parent_limit = total - level_size;
    for parent in 0..parent_limit {
        let p_u32 = parent as u32;
        for _ in 0..branching_factor {
            edges.push((p_u32, next_child));
            next_child += 1;
        }
    }
    CsrGraph::from_unweighted_edges(total, &edges, false, false)
}
