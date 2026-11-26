use std::collections::VecDeque;

use egraph_core::{CsrGraph, Scalar};

use super::super::*;

pub(super) const KARATE_CSV: &str = include_str!("../../../../dataset/src/data/karate.csv");
pub(super) const LESMIS_CSV: &str = include_str!("../../../../dataset/src/data/lesmis.csv");
pub(super) const BUS_1138_CSV: &str = include_str!("../../../../dataset/src/data/1138_bus.csv");

pub(super) fn parse_test_csv(input: &str) -> CsrGraph<f64> {
    let mut tokens = input.split_ascii_whitespace();
    let header = tokens.next().unwrap();
    let mut hparts = header.split(',');
    let n: usize = hparts.next().unwrap().parse().unwrap();
    let m: usize = hparts.next().unwrap().parse().unwrap();
    let mut edges = Vec::with_capacity(m);
    for row in tokens {
        let mut parts = row.split(',');
        let u: u32 = parts.next().unwrap().parse().unwrap();
        let v: u32 = parts.next().unwrap().parse().unwrap();
        edges.push((u, v));
    }
    assert_eq!(edges.len(), m);
    CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
}

pub(super) fn make_two_cliques_barbell(clique_size: usize) -> CsrGraph<f64> {
    let n = clique_size * 2;
    let mut edges = Vec::new();
    for i in 0..clique_size {
        for j in (i + 1)..clique_size {
            edges.push((i as u32, j as u32));
            edges.push(((clique_size + i) as u32, (clique_size + j) as u32));
        }
    }
    edges.push(((clique_size - 1) as u32, clique_size as u32));
    CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
}

pub(super) fn make_ring_of_cliques(num_cliques: usize, clique_size: usize) -> CsrGraph<f64> {
    let mut edges = Vec::new();
    for c in 0..num_cliques {
        let base = (c * clique_size) as u32;
        for i in 0..clique_size as u32 {
            for j in (i + 1)..clique_size as u32 {
                edges.push((base + i, base + j));
            }
        }
        let next_base = (((c + 1) % num_cliques) * clique_size) as u32;
        edges.push((base + clique_size as u32 - 1, next_base));
    }
    CsrGraph::from_edges(num_cliques * clique_size, &edges, None, false, false).unwrap()
}

pub(super) fn oracle_quality_and_modularity<W: Scalar>(
    graph: &CsrGraph<W>,
    labels: &[u32],
    num_communities: usize,
    objective: LeidenObjective,
    resolution: f64,
) -> (f64, f64) {
    let n = graph.num_nodes();
    let mut two_e_c = vec![0.0_f64; num_communities];
    let mut k_c = vec![0.0_f64; num_communities];
    let mut n_c = vec![0usize; num_communities];
    let mut two_m = 0.0_f64;

    for u in 0..n {
        let cu = labels[u] as usize;
        n_c[cu] += 1;
        let row_start = graph.indptr()[u] as usize;
        let row_end = graph.indptr()[u + 1] as usize;
        for pos in row_start..row_end {
            let v = graph.indices()[pos] as usize;
            let w = graph.weights().map_or(1.0, |ws| ws[pos].to_f64());
            k_c[cu] += w;
            two_m += w;
            if labels[v] as usize == cu {
                two_e_c[cu] += w;
            }
        }
    }

    let modularity = if two_m <= 0.0 {
        0.0
    } else {
        let mut q = 0.0_f64;
        for c in 0..num_communities {
            q += two_e_c[c] / two_m - (k_c[c] / two_m).powi(2);
        }
        q
    };

    let quality = match objective {
        LeidenObjective::Modularity => {
            if two_m <= 0.0 {
                0.0
            } else {
                let mut q_gamma = 0.0_f64;
                for c in 0..num_communities {
                    q_gamma += two_e_c[c] / two_m - resolution * (k_c[c] / two_m).powi(2);
                }
                q_gamma
            }
        }
        LeidenObjective::Cpm => {
            let mut h = 0.0_f64;
            for c in 0..num_communities {
                let sz = n_c[c] as f64;
                h += 0.5 * two_e_c[c] - resolution * (sz * (sz - 1.0) * 0.5);
            }
            h
        }
    };

    (quality, modularity)
}

pub(super) fn assert_canonical_and_connected<W: Scalar>(graph: &CsrGraph<W>, res: &LeidenResult) {
    let n = graph.num_nodes();
    assert_eq!(res.labels.len(), n);
    if n == 0 {
        assert_eq!(res.num_communities, 0);
        return;
    }

    // Verify canonical 0..num_communities order of first appearance.
    let mut next_expected = 0u32;
    let mut seen = vec![false; res.num_communities];
    for &l in &res.labels {
        let idx = l as usize;
        assert!(idx < res.num_communities);
        if !seen[idx] {
            assert_eq!(l, next_expected);
            seen[idx] = true;
            next_expected += 1;
        }
    }
    assert_eq!(next_expected as usize, res.num_communities);

    // Verify every community induces a single connected component via BFS over positive edges.
    let mut visited = vec![false; n];
    let mut comm_started = vec![false; res.num_communities];
    let mut queue = VecDeque::new();
    for start in 0..n {
        if visited[start] {
            continue;
        }
        let c = res.labels[start] as usize;
        assert!(
            !comm_started[c],
            "community {c} is disconnected (multiple components)"
        );
        comm_started[c] = true;
        visited[start] = true;
        queue.push_back(start as u32);
        while let Some(u_u32) = queue.pop_front() {
            let u = u_u32 as usize;
            let row_start = graph.indptr()[u] as usize;
            let row_end = graph.indptr()[u + 1] as usize;
            for pos in row_start..row_end {
                if let Some(ws) = graph.weights()
                    && ws[pos].to_f64() <= 0.0
                {
                    continue;
                }
                let v = graph.indices()[pos] as usize;
                if !visited[v] && res.labels[v] as usize == c {
                    visited[v] = true;
                    queue.push_back(v as u32);
                }
            }
        }
    }
}
