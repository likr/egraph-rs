use num_traits::Float;
use petgraph::visit::{IntoNeighbors, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use rand::prelude::*;
use std::collections::HashMap;

/// Sparse symmetric joint probability matrix P stored as adjacency rows `(neighbor_idx, p_ij)`.
pub type SparseP<S> = Vec<Vec<(usize, S)>>;

/// Validates that `perplexities` has length `n` and contains only finite, strictly positive values.
fn validate_perplexities<S: Float>(n: usize, perplexities: &[S]) -> Result<(), String> {
    if perplexities.len() != n {
        return Err(format!(
            "perplexities length ({}) must match node count ({n})",
            perplexities.len()
        ));
    }
    for (i, &p) in perplexities.iter().enumerate() {
        if !p.is_finite() || p <= S::zero() {
            return Err(format!(
                "perplexities[{i}] must be finite and strictly positive"
            ));
        }
    }
    Ok(())
}

/// Validates CSR k-NN arrays (`indptr`, `indices`, `distances`) for `n` nodes without allocating.
fn validate_csr_knn<S: Float>(
    n: usize,
    indptr: &[usize],
    indices: &[usize],
    distances: &[S],
) -> Result<(), String> {
    if indptr.len() != n + 1 {
        return Err(format!(
            "indptr length ({}) must equal node count + 1 ({})",
            indptr.len(),
            n + 1
        ));
    }
    if indptr[0] != 0 {
        return Err(format!("indptr[0] must be 0, got {}", indptr[0]));
    }
    for i in 0..n {
        if indptr[i + 1] < indptr[i] {
            return Err(format!(
                "indptr must be non-decreasing: indptr[{i}]={} > indptr[{}]={}",
                indptr[i],
                i + 1,
                indptr[i + 1]
            ));
        }
    }
    let nnz = indptr[n];
    if indices.len() != nnz {
        return Err(format!(
            "indices length ({}) must equal indptr[n] ({nnz})",
            indices.len()
        ));
    }
    if distances.len() != nnz {
        return Err(format!(
            "distances length ({}) must equal indptr[n] ({nnz})",
            distances.len()
        ));
    }
    for (k, (&idx, &dist)) in indices.iter().zip(distances.iter()).enumerate() {
        if idx >= n {
            return Err(format!(
                "indices[{k}] = {idx} is out of range for {n} nodes"
            ));
        }
        if !dist.is_finite() || dist < S::zero() {
            return Err(format!("distances[{k}] must be finite and non-negative"));
        }
    }
    Ok(())
}

/// Converts validated CSR k-NN arrays into row-wise neighbor lists `Vec<Vec<(usize, S)>>`.
pub fn knn_from_csr<S>(
    n: usize,
    indptr: &[usize],
    indices: &[usize],
    distances: &[S],
) -> Result<Vec<Vec<(usize, S)>>, String>
where
    S: DrawingValue + Float,
{
    validate_csr_knn(n, indptr, indices, distances)?;
    let mut knn = Vec::with_capacity(n);
    for i in 0..n {
        let start = indptr[i];
        let end = indptr[i + 1];
        let mut row = Vec::with_capacity(end - start);
        for (&j, &d) in indices[start..end].iter().zip(distances[start..end].iter()) {
            row.push((j, d));
        }
        knn.push(row);
    }
    Ok(knn)
}

/// Computes sparse high-dimensional symmetric joint probabilities
/// `P_{ij} = (P_{j|i} + P_{i|j}) / (2N)` from precomputed k-NN neighbor lists.
///
/// Each row `knn[i]` contains `(neighbor_idx, distance)` pairs for node `i`. Self-loops (`j == i`),
/// out-of-range indices (`j >= n`), and non-finite distances are filtered out, duplicate neighbors in
/// a row are deduplicated by retaining the minimum distance, and empty rows are handled safely.
pub fn compute_knn_probabilities<S>(n: usize, knn: &[Vec<(usize, S)>], perplexity: S) -> SparseP<S>
where
    S: DrawingValue + Float + Default,
{
    compute_knn_probabilities_with_params(
        n,
        knn,
        perplexity,
        S::from_f32(2.0).unwrap(),
        50,
        S::from_f32(1e-5).unwrap(),
    )
}

/// Computes sparse high-dimensional symmetric joint probabilities from precomputed k-NN neighbor
/// lists using explicit distance exponent and perplexity binary search parameters.
pub fn compute_knn_probabilities_with_params<S>(
    n: usize,
    knn: &[Vec<(usize, S)>],
    perplexity: S,
    power: S,
    sigma_iters: usize,
    sigma_tolerance: S,
) -> SparseP<S>
where
    S: DrawingValue + Float + Default,
{
    compute_knn_probabilities_core(
        n,
        |i| knn.get(i).into_iter().flat_map(|row| row.iter().copied()),
        |_| perplexity,
        power,
        sigma_iters,
        sigma_tolerance,
    )
}

/// Computes sparse high-dimensional symmetric joint probabilities from precomputed k-NN neighbor
/// lists using per-node target perplexities.
pub fn compute_knn_probabilities_with_perplexities<S>(
    n: usize,
    knn: &[Vec<(usize, S)>],
    perplexities: &[S],
    power: S,
    sigma_iters: usize,
    sigma_tolerance: S,
) -> Result<SparseP<S>, String>
where
    S: DrawingValue + Float + Default,
{
    if knn.len() != n {
        return Err(format!(
            "knn row count ({}) must match node count ({n})",
            knn.len()
        ));
    }
    validate_perplexities(n, perplexities)?;
    if n < 2 {
        return Ok(vec![Vec::new(); n]);
    }
    Ok(compute_knn_probabilities_core(
        n,
        |i| knn[i].iter().copied(),
        |i| perplexities[i],
        power,
        sigma_iters,
        sigma_tolerance,
    ))
}

/// Computes sparse high-dimensional symmetric joint probabilities directly from CSR k-NN arrays
/// `(indptr, indices, distances)` and per-node target perplexities without intermediate row copies.
#[allow(clippy::too_many_arguments)]
pub fn compute_knn_probabilities_from_csr<S>(
    n: usize,
    indptr: &[usize],
    indices: &[usize],
    distances: &[S],
    perplexities: &[S],
    power: S,
    sigma_iters: usize,
    sigma_tolerance: S,
) -> Result<SparseP<S>, String>
where
    S: DrawingValue + Float + Default,
{
    validate_csr_knn(n, indptr, indices, distances)?;
    validate_perplexities(n, perplexities)?;
    if n < 2 {
        return Ok(vec![Vec::new(); n]);
    }
    Ok(compute_knn_probabilities_core(
        n,
        |i| {
            let start = indptr[i];
            let end = indptr[i + 1];
            indices[start..end]
                .iter()
                .copied()
                .zip(distances[start..end].iter().copied())
        },
        |i| perplexities[i],
        power,
        sigma_iters,
        sigma_tolerance,
    ))
}

fn compute_knn_probabilities_core<S, I, FRow, FPerp>(
    n: usize,
    mut row_iter: FRow,
    mut row_perplexity: FPerp,
    power: S,
    sigma_iters: usize,
    sigma_tolerance: S,
) -> SparseP<S>
where
    S: DrawingValue + Float + Default,
    I: IntoIterator<Item = (usize, S)>,
    FRow: FnMut(usize) -> I,
    FPerp: FnMut(usize) -> S,
{
    if n < 2 {
        return vec![Vec::new(); n];
    }

    let two = S::from_f32(2.0).unwrap();
    let min_sum = S::from_f32(1e-12).unwrap();
    let perp_margin = S::from_f32(0.01).unwrap();

    let mut row_probs: Vec<Vec<(usize, S)>> = vec![Vec::new(); n];
    let mut seen_pos: HashMap<usize, usize> = HashMap::new();
    let mut collected: Vec<(usize, S)> = Vec::new();
    let mut dp_values: Vec<S> = Vec::new();
    let mut weights: Vec<S> = Vec::new();

    for (i, row_prob) in row_probs.iter_mut().enumerate() {
        seen_pos.clear();
        collected.clear();

        for (j, d) in row_iter(i) {
            if j == i || j >= n || !d.is_finite() {
                continue;
            }
            let d_nonneg = if d < S::zero() { S::zero() } else { d };
            if let Some(&pos) = seen_pos.get(&j) {
                if d_nonneg < collected[pos].1 {
                    collected[pos].1 = d_nonneg;
                }
            } else {
                seen_pos.insert(j, collected.len());
                collected.push((j, d_nonneg));
            }
        }

        let m = collected.len();
        if m == 0 {
            continue;
        }
        if m == 1 {
            row_prob.push((collected[0].0, S::one()));
            continue;
        }

        // Normalize distances before applying power and shifting by min_dp so binary search
        // operates over [0, 1] without overflow/underflow across arbitrary distance scales.
        let mut max_d = S::zero();
        for &(_, d) in &collected {
            if d > max_d {
                max_d = d;
            }
        }

        if max_d <= S::zero() {
            let p_uni = S::one() / S::from_usize(m).unwrap();
            for &(j, _) in &collected {
                row_prob.push((j, p_uni));
            }
            row_prob.sort_unstable_by_key(|&(j, _)| j);
            continue;
        }

        dp_values.clear();
        let mut min_dp = S::infinity();
        let mut max_dp = S::neg_infinity();
        for &(_, d) in &collected {
            let dp = (d / max_d).powf(power);
            let dp_safe = if dp.is_finite() { dp } else { S::one() };
            if dp_safe < min_dp {
                min_dp = dp_safe;
            }
            if dp_safe > max_dp {
                max_dp = dp_safe;
            }
            dp_values.push(dp_safe);
        }

        let range = max_dp - min_dp;
        if range <= S::zero() {
            let p_uni = S::one() / S::from_usize(m).unwrap();
            for &(j, _) in &collected {
                row_prob.push((j, p_uni));
            }
            row_prob.sort_unstable_by_key(|&(j, _)| j);
            continue;
        }

        for dp in &mut dp_values {
            *dp = (*dp - min_dp) / range;
        }

        let m_s = S::from_usize(m).unwrap();
        let max_perp = (m_s - perp_margin).max(S::one());
        let perp = row_perplexity(i);
        // Guard against NaN because Float::min returns the non-NaN argument, which would
        // silently turn NaN into max_perp on unvalidated scalar paths.
        let perp_safe = if perp.is_finite() { perp } else { S::one() };
        let perplexity_clamped = perp_safe.min(max_perp).max(S::one());
        let target_entropy = perplexity_clamped.ln();

        let mut beta = S::one();
        let mut beta_min = S::zero();
        let mut beta_max: Option<S> = None;

        weights.resize(m, S::zero());
        for _ in 0..sigma_iters {
            let mut sum_w = S::zero();
            for (idx, &dp) in dp_values.iter().enumerate() {
                let w = (-beta * dp).exp();
                weights[idx] = w;
                sum_w += w;
            }
            sum_w = sum_w.max(min_sum);

            let mut sum_dp_p = S::zero();
            for (idx, &dp) in dp_values.iter().enumerate() {
                sum_dp_p += (weights[idx] / sum_w) * dp;
            }

            let h = beta * sum_dp_p + sum_w.ln();
            let h_diff = h - target_entropy;

            if h_diff.abs() < sigma_tolerance {
                break;
            }

            if h_diff > S::zero() {
                beta_min = beta;
                match beta_max {
                    None => {
                        let next = beta * two;
                        if !next.is_finite() {
                            break;
                        }
                        beta = next;
                    }
                    Some(bmax) => {
                        beta = (beta_min + bmax) / two;
                    }
                }
            } else {
                beta_max = Some(beta);
                beta = (beta_min + beta) / two;
            }
        }

        // Compute final conditional probabilities p_{j|i}
        let mut sum_w = S::zero();
        for (idx, &dp) in dp_values.iter().enumerate() {
            let w = (-beta * dp).exp();
            weights[idx] = w;
            sum_w += w;
        }
        sum_w = sum_w.max(min_sum);

        for (idx, &(j, _)) in collected.iter().enumerate() {
            row_prob.push((j, weights[idx] / sum_w));
        }
        row_prob.sort_unstable_by_key(|&(j, _)| j);
    }

    // Symmetrize joint probabilities: p_{ij} = p_{ji} = (p_{j|i} + p_{i|j}) / (2 * N)
    let two_n = S::from_usize(2 * n).unwrap();
    let mut sparse_p: SparseP<S> = vec![Vec::new(); n];

    for i in 0..n {
        for &(j, p_ji) in &row_probs[i] {
            match row_probs[j].binary_search_by_key(&i, |&(col, _)| col) {
                Ok(pos) => {
                    if i < j {
                        let p_sym = (p_ji + row_probs[j][pos].1) / two_n;
                        if p_sym > S::zero() {
                            sparse_p[i].push((j, p_sym));
                            sparse_p[j].push((i, p_sym));
                        }
                    }
                }
                Err(_) => {
                    let p_sym = p_ji / two_n;
                    if p_sym > S::zero() {
                        sparse_p[i].push((j, p_sym));
                        sparse_p[j].push((i, p_sym));
                    }
                }
            }
        }
    }

    for row in &mut sparse_p {
        row.sort_unstable_by_key(|&(j, _)| j);
    }

    sparse_p
}

/// Computes sparse high-dimensional probabilities via Partial BFS in O(N) time (C0 component of BH-tsNET).
///
/// For each vertex `v`, Partial BFS explores the graph until `k` neighbors have been visited.
/// Ties at equal graph distances are broken randomly to ensure unbiased neighbor sampling.
/// Gaussian similarities are then normalized using binary search over standard deviation `sigma_v`
/// to attain the target perplexity `u`.
pub fn compute_partial_bfs_probabilities<G, S, R>(
    graph: G,
    perplexity: S,
    k: usize,
    power: S,
    sigma_iters: usize,
    sigma_tolerance: S,
    rng: &mut R,
) -> SparseP<S>
where
    G: IntoNeighbors + IntoNodeIdentifiers + NodeCount + NodeIndexable,
    G::NodeId: Copy + Eq + std::hash::Hash + DrawingIndex,
    S: DrawingValue + Float + Default,
    R: Rng + ?Sized,
{
    let n = graph.node_count();
    if n < 2 {
        return vec![Vec::new(); n];
    }

    let k_eff = k.min(n - 1).max(1);

    // Map each node identifier to a dense index 0..n
    let node_to_idx: HashMap<G::NodeId, usize> = graph
        .node_identifiers()
        .enumerate()
        .map(|(i, id)| (id, i))
        .collect();

    let mut knn: Vec<Vec<(usize, S)>> = Vec::with_capacity(n);
    // Allocate stamp and frontier buffers once outside the outer loop to keep per-vertex cost O(k * deg).
    let mut visited_stamp = vec![0usize; n];
    let mut current_layer: Vec<G::NodeId> = Vec::new();
    let mut next_layer: Vec<G::NodeId> = Vec::new();

    // Step 1: Run Partial BFS from each vertex
    for (i, node_id) in graph.node_identifiers().enumerate() {
        let stamp = i + 1;
        visited_stamp[i] = stamp;

        current_layer.clear();
        current_layer.push(node_id);

        let mut collected: Vec<(usize, S)> = Vec::with_capacity(k_eff);
        let mut current_dist = S::zero();

        while !current_layer.is_empty() && collected.len() < k_eff {
            current_dist += S::one();
            next_layer.clear();

            for &u in &current_layer {
                for neighbor in graph.neighbors(u) {
                    let neighbor_idx = node_to_idx[&neighbor];
                    if visited_stamp[neighbor_idx] != stamp {
                        visited_stamp[neighbor_idx] = stamp;
                        next_layer.push(neighbor);
                    }
                }
            }

            if next_layer.is_empty() {
                break;
            }

            let remaining_budget = k_eff - collected.len();
            if next_layer.len() <= remaining_budget {
                for &v_neighbor in &next_layer {
                    collected.push((node_to_idx[&v_neighbor], current_dist));
                }
                std::mem::swap(&mut current_layer, &mut next_layer);
            } else {
                // Break ties randomly when next layer exceeds remaining neighbor budget
                next_layer.shuffle(rng);
                for &v_neighbor in next_layer.iter().take(remaining_budget) {
                    collected.push((node_to_idx[&v_neighbor], current_dist));
                }
                break;
            }
        }

        knn.push(collected);
    }

    compute_knn_probabilities_with_params(n, &knn, perplexity, power, sigma_iters, sigma_tolerance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::UnGraph;
    use std::collections::HashSet;

    fn reference_partial_bfs_probabilities<G, R>(
        graph: G,
        perplexity: f64,
        k: usize,
        power: f64,
        sigma_iters: usize,
        sigma_tolerance: f64,
        rng: &mut R,
    ) -> SparseP<f64>
    where
        G: IntoNeighbors + IntoNodeIdentifiers + NodeCount + NodeIndexable,
        G::NodeId: Copy + Eq + std::hash::Hash + DrawingIndex,
        R: Rng + ?Sized,
    {
        let n = graph.node_count();
        if n < 2 {
            return vec![Vec::new(); n];
        }
        let k_eff = k.min(n - 1).max(1);
        let node_to_idx: HashMap<G::NodeId, usize> = graph
            .node_identifiers()
            .enumerate()
            .map(|(i, id)| (id, i))
            .collect();

        let mut knn = Vec::with_capacity(n);
        for node_id in graph.node_identifiers() {
            let mut visited: HashSet<G::NodeId> = HashSet::new();
            visited.insert(node_id);
            let mut current_layer = vec![node_id];
            let mut collected = Vec::with_capacity(k_eff);
            let mut current_dist = 0.0f64;

            while !current_layer.is_empty() && collected.len() < k_eff {
                current_dist += 1.0;
                let mut next_layer = Vec::new();
                for &u in &current_layer {
                    for neighbor in graph.neighbors(u) {
                        if visited.insert(neighbor) {
                            next_layer.push(neighbor);
                        }
                    }
                }
                if next_layer.is_empty() {
                    break;
                }
                let remaining = k_eff - collected.len();
                if next_layer.len() <= remaining {
                    for &v in &next_layer {
                        collected.push((node_to_idx[&v], current_dist));
                    }
                    current_layer = next_layer;
                } else {
                    next_layer.shuffle(rng);
                    for &v in next_layer.iter().take(remaining) {
                        collected.push((node_to_idx[&v], current_dist));
                    }
                    break;
                }
            }
            knn.push(collected);
        }

        compute_knn_probabilities_with_params(
            n,
            &knn,
            perplexity,
            power,
            sigma_iters,
            sigma_tolerance,
        )
    }

    fn assert_sparse_p_bit_equal(actual: &SparseP<f64>, expected: &SparseP<f64>, label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: row count mismatch");
        for (i, (row_a, row_e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(row_a.len(), row_e.len(), "{label}: row {i} len mismatch");
            for (&(ja, pa), &(je, pe)) in row_a.iter().zip(row_e.iter()) {
                assert_eq!(ja, je, "{label}: row {i} col mismatch");
                assert_eq!(
                    pa.to_bits(),
                    pe.to_bits(),
                    "{label}: row {i} col {ja} value mismatch: {pa} vs {pe}"
                );
            }
        }
    }

    #[test]
    fn test_partial_bfs_matches_naive_reference_bit_for_bit() {
        // 1. Connected 4x4 grid with tie-breaking (k = 5 < 2-hop neighborhood)
        let mut grid = UnGraph::<(), ()>::new_undirected();
        let grid_nodes: Vec<_> = (0..16).map(|_| grid.add_node(())).collect();
        for r in 0..4 {
            for c in 0..4 {
                let u = r * 4 + c;
                if c + 1 < 4 {
                    grid.add_edge(grid_nodes[u], grid_nodes[u + 1], ());
                }
                if r + 1 < 4 {
                    grid.add_edge(grid_nodes[u], grid_nodes[u + 4], ());
                }
            }
        }
        let actual = compute_partial_bfs_probabilities(
            &grid,
            4.0,
            5,
            2.0,
            50,
            1e-5,
            &mut StdRng::seed_from_u64(42),
        );
        let expected = reference_partial_bfs_probabilities(
            &grid,
            4.0,
            5,
            2.0,
            50,
            1e-5,
            &mut StdRng::seed_from_u64(42),
        );
        assert_sparse_p_bit_equal(&actual, &expected, "grid_4x4_k5");

        // 2. Disconnected components (triangle + edge + isolated node)
        let mut disc = UnGraph::<(), ()>::new_undirected();
        let dn: Vec<_> = (0..6).map(|_| disc.add_node(())).collect();
        disc.add_edge(dn[0], dn[1], ());
        disc.add_edge(dn[1], dn[2], ());
        disc.add_edge(dn[2], dn[0], ());
        disc.add_edge(dn[3], dn[4], ());
        let actual_disc = compute_partial_bfs_probabilities(
            &disc,
            2.0,
            4,
            2.0,
            50,
            1e-5,
            &mut StdRng::seed_from_u64(42),
        );
        let expected_disc = reference_partial_bfs_probabilities(
            &disc,
            2.0,
            4,
            2.0,
            50,
            1e-5,
            &mut StdRng::seed_from_u64(42),
        );
        assert_sparse_p_bit_equal(&actual_disc, &expected_disc, "disconnected");
        assert!(actual_disc[5].is_empty());

        // 3. Self-loops and parallel edges
        let mut multi = UnGraph::<(), ()>::new_undirected();
        let mn: Vec<_> = (0..5).map(|_| multi.add_node(())).collect();
        multi.add_edge(mn[0], mn[0], ());
        multi.add_edge(mn[0], mn[1], ());
        multi.add_edge(mn[0], mn[1], ());
        multi.add_edge(mn[1], mn[2], ());
        multi.add_edge(mn[2], mn[3], ());
        multi.add_edge(mn[3], mn[4], ());
        multi.add_edge(mn[4], mn[2], ());
        let actual_multi = compute_partial_bfs_probabilities(
            &multi,
            2.0,
            3,
            2.0,
            50,
            1e-5,
            &mut StdRng::seed_from_u64(42),
        );
        let expected_multi = reference_partial_bfs_probabilities(
            &multi,
            2.0,
            3,
            2.0,
            50,
            1e-5,
            &mut StdRng::seed_from_u64(42),
        );
        assert_sparse_p_bit_equal(&actual_multi, &expected_multi, "self_loop_parallel");

        // 4. Non-contiguous NodeIndex gaps after remove_node on StableUnGraph
        use petgraph::stable_graph::StableUnGraph;
        let mut gapped = StableUnGraph::<(), ()>::default();
        let gn: Vec<_> = (0..6).map(|_| gapped.add_node(())).collect();
        for i in 0..5 {
            gapped.add_edge(gn[i], gn[i + 1], ());
        }
        gapped.add_edge(gn[0], gn[2], ());
        gapped.add_edge(gn[2], gn[4], ());
        gapped.add_edge(gn[4], gn[5], ());
        gapped.remove_node(gn[1]);
        gapped.remove_node(gn[3]);
        assert!(gapped.node_bound() > gapped.node_count());
        let actual_gapped = compute_partial_bfs_probabilities(
            &gapped,
            2.0,
            3,
            2.0,
            50,
            1e-5,
            &mut StdRng::seed_from_u64(42),
        );
        let expected_gapped = reference_partial_bfs_probabilities(
            &gapped,
            2.0,
            3,
            2.0,
            50,
            1e-5,
            &mut StdRng::seed_from_u64(42),
        );
        assert_sparse_p_bit_equal(&actual_gapped, &expected_gapped, "remove_node_gaps");

        // 5. Edge cases: n = 0, n = 1, k = 0
        let empty = UnGraph::<(), ()>::new_undirected();
        assert!(compute_partial_bfs_probabilities(
            &empty,
            2.0,
            5,
            2.0,
            50,
            1e-5,
            &mut StdRng::seed_from_u64(42)
        )
        .is_empty());

        let mut single = UnGraph::<(), ()>::new_undirected();
        single.add_node(());
        assert_eq!(
            compute_partial_bfs_probabilities(
                &single,
                2.0,
                5,
                2.0,
                50,
                1e-5,
                &mut StdRng::seed_from_u64(42)
            ),
            vec![Vec::<(usize, f64)>::new()]
        );

        let actual_k0 = compute_partial_bfs_probabilities(
            &grid,
            2.0,
            0,
            2.0,
            50,
            1e-5,
            &mut StdRng::seed_from_u64(42),
        );
        let expected_k0 = reference_partial_bfs_probabilities(
            &grid,
            2.0,
            0,
            2.0,
            50,
            1e-5,
            &mut StdRng::seed_from_u64(42),
        );
        assert_sparse_p_bit_equal(&actual_k0, &expected_k0, "k_zero");
    }

    #[test]
    fn test_partial_bfs_medium_large_sparse_sanity() {
        let n = 2000;
        let mut graph = UnGraph::<(), ()>::new_undirected();
        let nodes: Vec<_> = (0..n).map(|_| graph.add_node(())).collect();
        for i in 0..n {
            graph.add_edge(nodes[i], nodes[(i + 1) % n], ());
            graph.add_edge(nodes[i], nodes[(i + 7) % n], ());
        }

        let mut rng = StdRng::seed_from_u64(42);
        let sparse_p =
            compute_partial_bfs_probabilities(&graph, 10.0f64, 20, 2.0, 50, 1e-5, &mut rng);
        assert_eq!(sparse_p.len(), n);

        let mut total_sum = 0.0;
        for (i, row) in sparse_p.iter().enumerate() {
            assert!(!row.is_empty(), "row {i} unexpectedly empty");
            let mut prev = None;
            for &(j, p_ij) in row {
                assert_ne!(i, j);
                assert!(p_ij.is_finite() && p_ij > 0.0);
                if let Some(p) = prev {
                    assert!(p < j);
                }
                prev = Some(j);
                total_sum += p_ij;
            }
        }
        assert!(
            (total_sum - 1.0).abs() < 1e-5,
            "expected probability sum 1.0, got {total_sum}"
        );
    }

    #[test]
    fn test_compute_knn_probabilities_with_perplexities_matches_scalar_uniform() {
        let knn: Vec<Vec<(usize, f64)>> = vec![
            vec![
                (0, 0.0),
                (99, 0.2),
                (1, 3.0),
                (1, 0.4),
                (2, f64::NAN),
                (2, 1.2),
                (3, 2.5),
                (4, 4.0),
            ],
            vec![(0, 0.4), (2, 0.8), (3, 1.5)],
            vec![(1, 0.8), (3, 1.1)],
            vec![(2, 1.1)],
            vec![],
            vec![(4, 0.5), (3, 1.0), (1, 2.2), (0, 3.5), (2, 5.0)],
        ];
        let n = knn.len();
        let perp = 2.25f64;
        let expected = compute_knn_probabilities_with_params(n, &knn, perp, 2.0, 50, 1e-5);
        let perplexities = vec![perp; n];
        let actual =
            compute_knn_probabilities_with_perplexities(n, &knn, &perplexities, 2.0, 50, 1e-5)
                .expect("valid adaptive perplexity input");
        assert_sparse_p_bit_equal(&actual, &expected, "uniform_perplexities_vs_scalar");
    }

    #[test]
    fn test_compute_knn_probabilities_with_perplexities_variable_rows_and_clamping() {
        // Two nodes with identical non-degenerate distance rows should reflect different target perplexities.
        let base_row = vec![(2, 0.1), (3, 0.5), (4, 1.0), (5, 2.0), (6, 4.0), (7, 8.0)];
        let mut row1 = base_row.clone();
        row1[0].0 = 2; // neighbors 2..7 for both 0 and 1
        let knn: Vec<Vec<(usize, f64)>> = vec![
            base_row,                 // row 0: m=6, low perplexity (clamped from 0.5 -> 1.0)
            row1,                     // row 1: m=6, higher perplexity 4.5
            vec![(0, 0.1), (1, 0.2)], // row 2: m=2, perplexity 100.0 clamped to 1.99
            vec![(0, 0.5)],           // row 3: m=1
            vec![(0, 1.0), (1, 1.0)], // row 4: m=2, equal distances -> uniform
            vec![],                   // row 5: m=0 (still receives symmetric entries)
            vec![(0, 4.0), (1, 4.0)], // row 6: m=2
            vec![(0, 8.0), (1, 8.0)], // row 7: m=2
        ];
        let n = knn.len();
        let perplexities = vec![0.5, 4.5, 100.0, 1.5, 3.0, 2.0, 1.2, 2.5];
        let sparse_p =
            compute_knn_probabilities_with_perplexities(n, &knn, &perplexities, 2.0, 50, 1e-5)
                .expect("valid heterogeneous perplexities");

        assert_eq!(sparse_p.len(), n);
        let mut total_sum = 0.0;
        for (i, row) in sparse_p.iter().enumerate() {
            let mut prev = None;
            for &(j, p_ij) in row {
                assert_ne!(i, j);
                assert!(p_ij.is_finite() && p_ij > 0.0);
                if let Some(p) = prev {
                    assert!(p < j);
                }
                prev = Some(j);
                let p_ji = sparse_p[j]
                    .iter()
                    .find(|&&(col, _)| col == i)
                    .map(|&(_, v)| v)
                    .expect("missing symmetric entry");
                assert!((p_ij - p_ji).abs() < 1e-12);
                total_sum += p_ij;
            }
        }
        // 7 of 8 rows are non-empty, so total probability sum is 7 / 8.
        assert!((total_sum - 7.0 / 8.0).abs() < 1e-6);

        // Node 0 had lower perplexity (0.5 -> 1.0) than node 1 (4.5) on identical neighbor distances 2..7,
        // so node 0 places strictly higher probability mass on closest neighbor 2 than node 1 does.
        let p02 = sparse_p[0].iter().find(|&&(j, _)| j == 2).unwrap().1;
        let p12 = sparse_p[1].iter().find(|&&(j, _)| j == 2).unwrap().1;
        assert!(p02 > p12, "expected p02 ({p02}) > p12 ({p12})");
    }

    #[test]
    fn test_knn_from_csr_and_csr_probabilities_equivalence_and_validation() {
        let indptr = vec![0, 3, 4, 4, 7];
        let indices = vec![1, 2, 1, 0, 3, 2, 1];
        let distances = vec![0.4, 1.2, 0.4, 0.4, 0.0, 0.8, 1.5];
        let perplexities = vec![1.5, 2.0, 1.0, 2.5];

        let knn = knn_from_csr(4, &indptr, &indices, &distances).expect("valid CSR");
        let via_knn =
            compute_knn_probabilities_with_perplexities(4, &knn, &perplexities, 2.0, 50, 1e-5)
                .expect("valid knn");
        let via_csr = compute_knn_probabilities_from_csr(
            4,
            &indptr,
            &indices,
            &distances,
            &perplexities,
            2.0,
            50,
            1e-5,
        )
        .expect("valid csr");
        assert_sparse_p_bit_equal(&via_csr, &via_knn, "csr_vs_knn");

        // Edge cases: n = 0 and n = 1
        assert!(
            compute_knn_probabilities_from_csr::<f64>(0, &[0], &[], &[], &[], 2.0, 50, 1e-5)
                .expect("n=0 valid")
                .is_empty()
        );
        assert_eq!(
            compute_knn_probabilities_from_csr::<f64>(1, &[0, 0], &[], &[], &[1.0], 2.0, 50, 1e-5)
                .expect("n=1 valid"),
            vec![Vec::<(usize, f64)>::new()]
        );

        // Validation errors (must fail cleanly even for n < 2)
        assert!(knn_from_csr::<f64>(4, &indptr[..4], &indices, &distances).is_err());
        assert!(knn_from_csr::<f64>(4, &[1, 3, 4, 4, 7], &indices, &distances).is_err());
        assert!(knn_from_csr::<f64>(4, &[0, 4, 3, 4, 7], &indices, &distances).is_err());
        assert!(knn_from_csr::<f64>(4, &indptr, &indices[..6], &distances).is_err());
        assert!(knn_from_csr::<f64>(4, &indptr, &indices, &distances[..6]).is_err());
        assert!(knn_from_csr::<f64>(4, &[0, 10_000, 10_000, 10_000, 10_000], &[], &[]).is_err());

        let mut bad_idx = indices.clone();
        bad_idx[0] = 4;
        assert!(knn_from_csr::<f64>(4, &indptr, &bad_idx, &distances).is_err());

        for bad_d in [-0.1, f64::NAN, f64::INFINITY] {
            let mut bad_dists = distances.clone();
            bad_dists[0] = bad_d;
            assert!(knn_from_csr::<f64>(4, &indptr, &indices, &bad_dists).is_err());
        }

        // Perplexity validation errors (including n = 1)
        assert!(compute_knn_probabilities_from_csr(
            4,
            &indptr,
            &indices,
            &distances,
            &perplexities[..3],
            2.0,
            50,
            1e-5
        )
        .is_err());
        for bad_p in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(compute_knn_probabilities_from_csr::<f64>(
                1,
                &[0, 0],
                &[],
                &[],
                &[bad_p],
                2.0,
                50,
                1e-5
            )
            .is_err());
        }
        assert!(compute_knn_probabilities_with_perplexities::<f64>(
            2,
            &[vec![]],
            &[1.0, 1.0],
            2.0,
            50,
            1e-5
        )
        .is_err());
    }
}
