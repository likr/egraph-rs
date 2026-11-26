use egraph_core::{CoreResult, CsrGraph, Scalar};

use super::simmelian::{
    WEIGHT_TIE_EPS, build_simple_undirected_graph, compute_simmelian_prefix_jaccard,
    count_quadrangles,
};
use super::{BackboneConfig, BackboneResult};

struct DisjointSet {
    parent: Vec<u32>,
    size: Vec<u32>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            size: vec![1u32; n],
        }
    }

    fn find(&mut self, mut x: u32) -> u32 {
        let mut root = x;
        while self.parent[root as usize] != root {
            root = self.parent[root as usize];
        }
        while self.parent[x as usize] != root {
            let next = self.parent[x as usize];
            self.parent[x as usize] = root;
            x = next;
        }
        root
    }

    fn union(&mut self, a: u32, b: u32) {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.size[ra as usize] < self.size[rb as usize] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb as usize] = ra;
        self.size[ra as usize] += self.size[rb as usize];
    }
}

/// Extracts the Quadrilateral Simmelian Backbone (Nocaj et al. 2015) from `graph`.
///
/// Counts edge quadrangles via Chiba–Nishizeki $O(m\,\alpha(G))$, normalizes $Q(u, v)$,
/// applies tie-invariant Simmelian prefix-Jaccard reweighting $w(u, v)$, computes the
/// bucketed Union of Maximum Spanning Trees (`UMST`), and merges `E_UMST` with the top
/// `config.retain_ratio` fraction of edges.
pub fn extract_quadrilateral_backbone<W: Scalar>(
    graph: &CsrGraph<W>,
    config: &BackboneConfig,
) -> CoreResult<BackboneResult> {
    config.validate()?;
    let simple = build_simple_undirected_graph(graph)?;
    let n = simple.num_nodes;
    let m = simple.edges.len();

    if m == 0 {
        let backbone_csr = CsrGraph::from_edges(n, &[], None, false, false)?;
        return Ok(BackboneResult {
            simmelian_weights: Vec::new(),
            umst_edges: Vec::new(),
            backbone_edges: Vec::new(),
            edge_weights: Vec::new(),
            backbone_csr,
        });
    }

    let q_edge = count_quadrangles(&simple);
    let simmelian_weights = compute_simmelian_prefix_jaccard(&simple, &q_edge);

    // Sort edge indices descending by Simmelian weight (tie-break deterministically by edge ID).
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_unstable_by(|&a, &b| {
        simmelian_weights[b]
            .total_cmp(&simmelian_weights[a])
            .then_with(|| a.cmp(&b))
    });

    // Bucketed Kruskal UMST (Nocaj et al. 2015 Algorithm 2).
    let mut dsu = DisjointSet::new(n);
    let mut in_umst = vec![false; m];
    let mut to_union: Vec<(u32, u32)> = Vec::new();
    let mut b_start = 0usize;
    while b_start < m {
        let anchor_w = simmelian_weights[order[b_start]];
        let mut b_end = b_start + 1;
        while b_end < m && (anchor_w - simmelian_weights[order[b_end]]).abs() <= WEIGHT_TIE_EPS {
            b_end += 1;
        }
        to_union.clear();
        for &e_idx in &order[b_start..b_end] {
            let (u, v) = simple.edges[e_idx];
            if dsu.find(u) != dsu.find(v) {
                in_umst[e_idx] = true;
                to_union.push((u, v));
            }
        }
        for &(u, v) in &to_union {
            dsu.union(u, v);
        }
        b_start = b_end;
    }

    let mut in_backbone = in_umst.clone();
    if config.retain_ratio > 0.0 {
        let k = ((config.retain_ratio * (m as f64)).ceil() as usize).clamp(1, m);
        let w_thr = simmelian_weights[order[k - 1]];
        for &e_idx in &order {
            if simmelian_weights[e_idx] >= w_thr - WEIGHT_TIE_EPS {
                in_backbone[e_idx] = true;
            } else {
                break;
            }
        }
    }

    let mut umst_edges = Vec::new();
    let mut backbone_edges = Vec::new();
    let mut edge_weights = Vec::with_capacity(m);
    let mut csr_edges = Vec::new();
    let mut csr_weights = Vec::new();
    let mut all_unit_weight = true;

    for (e_idx, &pair) in simple.edges.iter().enumerate() {
        if in_umst[e_idx] {
            umst_edges.push(pair);
        }
        if in_backbone[e_idx] {
            backbone_edges.push(pair);
        }
        let w = if in_umst[e_idx] {
            config.umst_weight_scale
        } else if in_backbone[e_idx] {
            1.0
        } else {
            config.non_backbone_weight_scale
        };
        edge_weights.push(w);

        let include_in_csr =
            (in_backbone[e_idx] || config.non_backbone_weight_scale > 0.0) && w > 0.0;
        if include_in_csr {
            csr_edges.push(pair);
            csr_weights.push(w);
            if w != 1.0 {
                all_unit_weight = false;
            }
        }
    }

    let backbone_csr = if all_unit_weight {
        CsrGraph::from_edges(n, &csr_edges, None, false, false)?
    } else {
        CsrGraph::from_edges(n, &csr_edges, Some(&csr_weights), false, false)?
    };

    Ok(BackboneResult {
        simmelian_weights,
        umst_edges,
        backbone_edges,
        edge_weights,
        backbone_csr,
    })
}
