mod directed_and_errors;
mod parity_and_boundaries;
mod regressions;

use egraph_core::{CsrGraph, Layout2d, Scalar};

use crate::extract_unique_edges;

use super::*;

pub(super) fn brute_gg_edges(layout: &Layout2d<f64>) -> Vec<(u32, u32)> {
    let n = layout.len();
    let xs = layout.x();
    let ys = layout.y();
    let mut out = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let d_ij_sq = (xs[i] - xs[j]).powi(2) + (ys[i] - ys[j]).powi(2);
            if d_ij_sq <= 0.0 {
                continue;
            }
            let mut ok = true;
            for r in 0..n {
                if r == i || r == j {
                    continue;
                }
                let d_ir_sq = (xs[i] - xs[r]).powi(2) + (ys[i] - ys[r]).powi(2);
                let d_jr_sq = (xs[j] - xs[r]).powi(2) + (ys[j] - ys[r]).powi(2);
                if d_ir_sq + d_jr_sq <= d_ij_sq * (1.0 + 1e-12) {
                    ok = false;
                    break;
                }
            }
            if ok {
                out.push((i as u32, j as u32));
            }
        }
    }
    out
}

pub(super) fn brute_rng_edges(layout: &Layout2d<f64>) -> Vec<(u32, u32)> {
    let n = layout.len();
    let xs = layout.x();
    let ys = layout.y();
    let mut out = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let d_ij_sq = (xs[i] - xs[j]).powi(2) + (ys[i] - ys[j]).powi(2);
            if d_ij_sq <= 0.0 {
                continue;
            }
            let mut ok = true;
            for r in 0..n {
                if r == i || r == j {
                    continue;
                }
                let d_ir_sq = (xs[i] - xs[r]).powi(2) + (ys[i] - ys[r]).powi(2);
                let d_jr_sq = (xs[j] - xs[r]).powi(2) + (ys[j] - ys[r]).powi(2);
                let thresh = d_ij_sq * (1.0 - 1e-12);
                if d_ir_sq < thresh && d_jr_sq < thresh {
                    ok = false;
                    break;
                }
            }
            if ok {
                out.push((i as u32, j as u32));
            }
        }
    }
    out
}

pub(super) fn brute_gabriel_graph_property<W>(graph: &CsrGraph<W>, layout: &Layout2d<f64>) -> f64 {
    let n = layout.len();
    let xs = layout.x();
    let ys = layout.y();
    let mut sum = 0.0_f64;
    for (u, v) in extract_unique_edges(graph) {
        let ui = u as usize;
        let vi = v as usize;
        let r = 0.5 * (xs[ui] - xs[vi]).hypot(ys[ui] - ys[vi]);
        if r <= 0.0 {
            continue;
        }
        let cx = 0.5 * (xs[ui] + xs[vi]);
        let cy = 0.5 * (ys[ui] + ys[vi]);
        for i in 0..n {
            if i == ui || i == vi {
                continue;
            }
            let d = (xs[i] - cx).hypot(ys[i] - cy);
            if d < r {
                let viol = r - d;
                sum += viol * viol;
            }
        }
    }
    sum
}

pub(super) fn brute_node_resolution(layout: &Layout2d<f64>) -> f64 {
    let n = layout.len();
    if n <= 1 {
        return 0.0;
    }
    let xs = layout.x();
    let ys = layout.y();
    let mut d_max_sq = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let d_sq = (xs[i] - xs[j]).powi(2) + (ys[i] - ys[j]).powi(2);
            d_max_sq = d_max_sq.max(d_sq);
        }
    }
    if d_max_sq <= 0.0 {
        return (n as f64) * ((n - 1) as f64) * 0.5;
    }
    let thresh = d_max_sq.sqrt() / (n as f64).sqrt();
    let mut sum = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = (xs[i] - xs[j]).hypot(ys[i] - ys[j]);
            if d < thresh {
                let hinge = 1.0 - d / thresh;
                sum += hinge * hinge;
            }
        }
    }
    sum
}

pub(super) fn assert_proximity_determinism_and_tree_parity<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) {
    let tree = build_kdtree_f64(layout).unwrap();

    let gg_prop_1 = gabriel_graph_property(graph, layout).unwrap();
    let gg_prop_2 = gabriel_graph_property(graph, layout).unwrap();
    let gg_prop_tree = gabriel_graph_property_with_tree(graph, layout, &tree).unwrap();
    assert_eq!(gg_prop_1.to_bits(), gg_prop_2.to_bits());
    assert_eq!(gg_prop_1.to_bits(), gg_prop_tree.to_bits());

    let nr_1 = node_resolution(layout).unwrap();
    let nr_2 = node_resolution(layout).unwrap();
    let nr_tree = node_resolution_with_tree(layout, &tree).unwrap();
    assert_eq!(nr_1.to_bits(), nr_2.to_bits());
    assert_eq!(nr_1.to_bits(), nr_tree.to_bits());

    let rng_edges_1 = relative_neighborhood_edges(layout).unwrap();
    let rng_edges_tree = relative_neighborhood_edges_with_tree(layout, &tree).unwrap();
    assert_eq!(rng_edges_1, rng_edges_tree);

    let gg_edges_1 = gabriel_graph_edges(layout).unwrap();
    let gg_edges_tree = gabriel_graph_edges_with_tree(layout, &tree).unwrap();
    assert_eq!(gg_edges_1, gg_edges_tree);

    let s_rng_1 = relative_neighborhood_preservation(graph, layout).unwrap();
    let s_rng_2 = relative_neighborhood_preservation(graph, layout).unwrap();
    let s_rng_tree = relative_neighborhood_preservation_with_tree(graph, layout, &tree).unwrap();
    assert_eq!(s_rng_1.to_bits(), s_rng_2.to_bits());
    assert_eq!(s_rng_1.to_bits(), s_rng_tree.to_bits());

    let s_gg_1 = gabriel_graph_preservation(graph, layout).unwrap();
    let s_gg_2 = gabriel_graph_preservation(graph, layout).unwrap();
    let s_gg_tree = gabriel_graph_preservation_with_tree(graph, layout, &tree).unwrap();
    assert_eq!(s_gg_1.to_bits(), s_gg_2.to_bits());
    assert_eq!(s_gg_1.to_bits(), s_gg_tree.to_bits());
}
