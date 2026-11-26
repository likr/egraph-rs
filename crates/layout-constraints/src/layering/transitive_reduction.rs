use std::collections::VecDeque;

use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar, VisitStamp};

/// Computes the exact transitive reduction of a directed acyclic [`CsrGraph<W>`].
///
/// Retains every directed edge `(u, v)` for which no directed path $u \leadsto w \to v$ of
/// length $\ge 2$ exists, preserving `num_nodes` and original edge weights (when `graph` is
/// weighted). Rejects undirected graphs with [`CoreError::UnsupportedOperation`] and cyclic
/// graphs with [`CoreError::InvalidInput`].
pub fn transitive_reduction<W: Scalar + Copy>(graph: &CsrGraph<W>) -> CoreResult<CsrGraph<W>> {
    if !graph.is_directed() {
        return Err(CoreError::unsupported(
            "transitive_reduction requires a directed CsrGraph",
        ));
    }
    let n = graph.num_nodes();
    let weights = graph.weights();
    if n == 0 {
        let empty_w = weights.map(|_| Vec::<W>::new());
        return CsrGraph::from_edges(0, &[], empty_w.as_deref(), true, false);
    }

    // Step 1: Compute Kahn topological order and reject any cycle (including self-loops).
    let indptr = graph.indptr();
    let indices = graph.indices();
    let mut in_deg = vec![0u32; n];
    for &v in indices {
        in_deg[v as usize] += 1;
    }

    let mut queue = VecDeque::new();
    for (u, &deg) in in_deg.iter().enumerate() {
        if deg == 0 {
            queue.push_back(u as u32);
        }
    }

    let mut topo_pos = vec![0u32; n];
    let mut visited_count = 0usize;
    while let Some(u) = queue.pop_front() {
        topo_pos[u as usize] = visited_count as u32;
        visited_count += 1;
        for &v in graph.neighbors(u as usize) {
            let vi = v as usize;
            in_deg[vi] -= 1;
            if in_deg[vi] == 0 {
                queue.push_back(v);
            }
        }
    }

    if visited_count != n {
        return Err(CoreError::invalid_input(
            "transitive_reduction requires an acyclic directed graph (DAG)",
        ));
    }

    // Step 2: For each vertex u, sort outgoing edges by increasing topological position of v.
    // An edge (u, v) is redundant iff v is already reachable from an earlier out-neighbor of u.
    let mut reduced_edges: Vec<(u32, u32)> = Vec::new();
    let mut reduced_weights: Option<Vec<W>> = weights.map(|_| Vec::new());

    let mut stamp = VisitStamp::new(n);
    let mut stack: Vec<u32> = Vec::with_capacity(n.min(1024));
    let mut nbrs: Vec<(u32, u32, usize)> = Vec::new();

    for u in 0..n {
        let start = indptr[u] as usize;
        let end = indptr[u + 1] as usize;
        let deg = end - start;
        if deg == 0 {
            continue;
        }
        let u_u32 = u as u32;
        if deg == 1 {
            reduced_edges.push((u_u32, indices[start]));
            if let (Some(w_src), Some(w_dst)) = (weights, reduced_weights.as_mut()) {
                w_dst.push(w_src[start]);
            }
            continue;
        }

        nbrs.clear();
        for (offset, &v) in indices[start..end].iter().enumerate() {
            let pos = start + offset;
            nbrs.push((topo_pos[v as usize], v, pos));
        }
        nbrs.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let max_target_topo = nbrs.last().unwrap().0;

        stamp.clear();
        let mut prev_v: Option<u32> = None;
        for &(_tpos, v, pos) in &nbrs {
            if prev_v == Some(v) {
                continue;
            }
            prev_v = Some(v);

            if stamp.is_visited(v as usize) {
                continue;
            }

            reduced_edges.push((u_u32, v));
            if let (Some(w_src), Some(w_dst)) = (weights, reduced_weights.as_mut()) {
                w_dst.push(w_src[pos]);
            }

            if stamp.visit(v as usize) {
                stack.clear();
                stack.push(v);
                while let Some(cur) = stack.pop() {
                    for &nxt in graph.neighbors(cur as usize) {
                        let ni = nxt as usize;
                        if topo_pos[ni] <= max_target_topo && stamp.visit(ni) {
                            stack.push(nxt);
                        }
                    }
                }
            }
        }
    }

    CsrGraph::from_edges(n, &reduced_edges, reduced_weights.as_deref(), true, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use egraph_core::DeterministicRng;

    fn reachability_matrix(g: &CsrGraph<f64>) -> Vec<Vec<bool>> {
        let n = g.num_nodes();
        let mut reach = vec![vec![false; n]; n];
        let mut stack = Vec::new();
        for (s, row) in reach.iter_mut().enumerate() {
            stack.clear();
            for &v in g.neighbors(s) {
                if !row[v as usize] {
                    row[v as usize] = true;
                    stack.push(v);
                }
            }
            while let Some(u) = stack.pop() {
                for &v in g.neighbors(u as usize) {
                    if !row[v as usize] {
                        row[v as usize] = true;
                        stack.push(v);
                    }
                }
            }
        }
        reach
    }

    #[test]
    fn test_transitive_reduction_chain_and_diamond_with_weights() {
        // Chain 0 -> 1 -> 2 -> 3 plus shortcuts 0 -> 2, 0 -> 3, 1 -> 3.
        let g = CsrGraph::<f64>::from_edges(
            4,
            &[(0, 1), (1, 2), (2, 3), (0, 2), (0, 3), (1, 3)],
            Some(&[1.5, 2.5, 3.5, 9.0, 9.0, 9.0]),
            true,
            false,
        )
        .unwrap();

        let tr = transitive_reduction(&g).unwrap();
        assert_eq!(tr.num_nodes(), 4);
        assert_eq!(tr.edges(), vec![(0, 1), (1, 2), (2, 3)]);
        assert_eq!(tr.weights().unwrap(), &[1.5, 2.5, 3.5]);

        // Diamond 0 -> 1 -> 3, 0 -> 2 -> 3, plus transitive shortcut 0 -> 3.
        let diamond = CsrGraph::<f64>::from_edges(
            4,
            &[(0, 1), (0, 2), (1, 3), (2, 3), (0, 3)],
            None,
            true,
            false,
        )
        .unwrap();
        let tr_d = transitive_reduction(&diamond).unwrap();
        assert_eq!(tr_d.edges(), vec![(0, 1), (0, 2), (1, 3), (2, 3)]);
        assert!(tr_d.weights().is_none());
    }

    #[test]
    fn test_transitive_reduction_random_dags_reachability_and_minimality() {
        let mut rng = DeterministicRng::from_seed_u64(2026);
        for trial in 0..8 {
            let n = 25usize;
            let prob = 0.25 + (trial as f64) * 0.05;
            let mut edges = Vec::new();
            let mut weights = Vec::new();
            for u in 0..n {
                for v in (u + 1)..n {
                    if rng.next_f64() < prob {
                        edges.push((u as u32, v as u32));
                        weights.push((u * 100 + v) as f64);
                    }
                }
            }
            let g = CsrGraph::<f64>::from_edges(n, &edges, Some(&weights), true, false).unwrap();
            let tr = transitive_reduction(&g).unwrap();

            // 1. Reachability matrix is identical.
            assert_eq!(reachability_matrix(&g), reachability_matrix(&tr));

            // 2. Every retained edge has matching original weight and is essential (removing it changes reachability).
            let tr_edges = tr.edges();
            assert!(tr_edges.len() <= edges.len());
            for &(u, v) in &tr_edges {
                let orig_w = g.edge_weight(u as usize, v as usize).unwrap();
                let tr_w = tr.edge_weight(u as usize, v as usize).unwrap();
                assert_eq!(orig_w, tr_w);

                let pruned_edges: Vec<(u32, u32)> =
                    tr_edges.iter().copied().filter(|&e| e != (u, v)).collect();
                let pruned =
                    CsrGraph::<f64>::from_edges(n, &pruned_edges, None, true, false).unwrap();
                assert_ne!(
                    reachability_matrix(&tr),
                    reachability_matrix(&pruned),
                    "edge ({u}, {v}) was redundant"
                );
            }
        }
    }

    #[test]
    fn test_transitive_reduction_rejects_undirected_cyclic_and_handles_empty() {
        let undir = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, false, false).unwrap();
        assert!(matches!(
            transitive_reduction(&undir),
            Err(CoreError::UnsupportedOperation(_))
        ));

        let cyclic =
            CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2), (2, 0)], None, true, false).unwrap();
        assert!(matches!(
            transitive_reduction(&cyclic),
            Err(CoreError::InvalidInput(_))
        ));

        let empty_weighted = CsrGraph::<f64>::from_edges(0, &[], Some(&[]), true, false).unwrap();
        let tr_empty = transitive_reduction(&empty_weighted).unwrap();
        assert_eq!(tr_empty.num_nodes(), 0);
        assert!(tr_empty.is_weighted());
        assert!(tr_empty.is_directed());
    }
}
