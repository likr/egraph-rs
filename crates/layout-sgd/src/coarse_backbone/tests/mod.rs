mod backends_and_normalization;
mod overlap_and_scaling;

use egraph_core::CsrGraph;

pub(super) fn make_three_cluster_chain(big_size: usize) -> CsrGraph<f64> {
    // Cluster A: 6-clique {0..6}, Cluster B: big_size-clique {6..6+big_size},
    // Cluster C: 6-clique {6+big_size..12+big_size}, linked only by A-B and B-C bridges.
    let small = 6usize;
    let n = 2 * small + big_size;
    let mut edges = Vec::new();
    for i in 0..small {
        for j in (i + 1)..small {
            edges.push((i as u32, j as u32));
        }
    }
    for i in 0..big_size {
        for j in (i + 1)..big_size {
            edges.push(((small + i) as u32, (small + j) as u32));
        }
    }
    let c0 = small + big_size;
    for i in 0..small {
        for j in (i + 1)..small {
            edges.push(((c0 + i) as u32, (c0 + j) as u32));
        }
    }
    edges.push(((small - 1) as u32, small as u32));
    edges.push(((small + big_size - 1) as u32, c0 as u32));
    CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
}

pub(super) fn make_unequal_clusters_graph(
    n_big: usize,
    big_size: usize,
    n_small: usize,
    small_size: usize,
    topology: &str,
) -> CsrGraph<f64> {
    let mut offsets = Vec::with_capacity(n_big + n_small);
    let mut cur = 0usize;
    for _ in 0..n_big {
        offsets.push((cur, big_size));
        cur += big_size;
    }
    for _ in 0..n_small {
        offsets.push((cur, small_size));
        cur += small_size;
    }
    let n_total = cur;
    let mut edges = Vec::new();
    for &(off, sz) in &offsets {
        for i in 0..sz {
            for j in (i + 1)..sz {
                edges.push(((off + i) as u32, (off + j) as u32));
            }
        }
    }
    match topology {
        "chain" => {
            for c in 0..(offsets.len().saturating_sub(1)) {
                let u = offsets[c].0 + offsets[c].1 - 1;
                let v = offsets[c + 1].0;
                edges.push((u as u32, v as u32));
            }
        }
        "star_chain" => {
            // Star over the big clusters centered at cluster 0, then chain through small clusters.
            for c in 1..n_big {
                let u = offsets[0].0 + (c % offsets[0].1);
                let v = offsets[c].0;
                edges.push((u as u32, v as u32));
            }
            if n_big > 0 && n_small > 0 {
                let u = offsets[0].0 + offsets[0].1 - 1;
                let v = offsets[n_big].0;
                edges.push((u as u32, v as u32));
                for c in n_big..(offsets.len() - 1) {
                    let a = offsets[c].0 + offsets[c].1 - 1;
                    let b = offsets[c + 1].0;
                    edges.push((a as u32, b as u32));
                }
            }
        }
        "mixed_2comp" => {
            // Component 0: star over big clusters; Component 1: chain over small clusters.
            for c in 1..n_big {
                let u = offsets[0].0 + (c % offsets[0].1);
                let v = offsets[c].0;
                edges.push((u as u32, v as u32));
            }
            for c in n_big..(offsets.len().saturating_sub(1)) {
                let a = offsets[c].0 + offsets[c].1 - 1;
                let b = offsets[c + 1].0;
                edges.push((a as u32, b as u32));
            }
        }
        "star_all" => {
            // Star over all remaining clusters centered at cluster 0.
            for c in 1..offsets.len() {
                let u = offsets[0].0 + (c % offsets[0].1);
                let v = offsets[c].0;
                edges.push((u as u32, v as u32));
            }
        }
        "disconnected" => {}
        other => panic!("unknown topology: {other}"),
    }
    CsrGraph::from_edges(n_total, &edges, None, false, false).unwrap()
}
