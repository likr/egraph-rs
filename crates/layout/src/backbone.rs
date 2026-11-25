//! Quadrilateral Simmelian Backbone extraction and layout solver (Nocaj, Ortmann & Brandes, 2015).
//!
//! Implements the hairball-untangling graph simplification pipeline from Nocaj et al. (2015):
//! 1. **Chiba–Nishizeki $O(m\,\alpha(G))$ quadrangle counting**: counts 4-cycles $q(u, v)$ containing
//!    each canonical undirected simple edge $(u, v)$.
//! 2. **Normalized quadrilateral embeddedness**:
//!    $$Q(u, v) = \frac{q(u, v)}{\sqrt{q(u)\,q(v)}}, \qquad q(v) = \sum_{w \in N(v)} q(v, w).$$
//! 3. **Tie-invariant Simmelian prefix-Jaccard reweighting**: ranks each vertex's neighbors by
//!    $Q(u, \cdot)$ descending (expanding ties symmetrically) and sets $w(u, v)$ to the maximum
//!    prefix Jaccard similarity.
//! 4. **Bucketed Union of Maximum Spanning Trees (`UMST`)**: processes equal-weight edge buckets
//!    before committing Union-Find merges so all edges belonging to any MST are preserved.
//! 5. **Backbone assembly & layout**: combines $E_{\text{UMST}}$ with top-`retain_ratio` edges and
//!    runs cooperative [`StressMajorizationSolver`] or [`SgdSolver`] via [`IterativeSolver`].

use egraph_core::{
    CoreError, CoreResult, CsrGraph, IterativeSolver, Layout2d, Scalar, SolverStatus, VisitStamp,
};
use egraph_linalg::{BfsWorkspace, DijkstraWorkspace};

use crate::sgd::{SgdScheduleConfig, SgdSolver};
use crate::stress_majorization::{StressMajorizationConfig, StressMajorizationSolver};

const WEIGHT_TIE_EPS: f64 = 1e-12;

/// Layout backend used by [`BackboneSolver`] on the extracted backbone graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BackboneLayoutMode {
    /// Gansner-Koren-North (2004) Stress Majorization (Nocaj et al. 2015 default).
    #[default]
    StressMajorization,
    /// Full-APSP Stochastic Gradient Descent (`SgdSolver::full`).
    Sgd,
}

/// Configuration for Quadrilateral Simmelian Backbone extraction and layout.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BackboneConfig {
    /// Fraction of top-ranked edges retained beyond `E_UMST`, in `[0.0, 1.0]` (`default: 0.2`).
    pub retain_ratio: f64,
    /// Layout edge weight scale assigned to `E_UMST` edges, finite `>= 0.0` (`default: 1.0`).
    pub umst_weight_scale: f64,
    /// Layout edge weight scale assigned to non-backbone edges, finite `>= 0.0` (`default: 0.0`,
    /// omitting non-backbone edges as in Nocaj et al. 2015 §3.2).
    pub non_backbone_weight_scale: f64,
    /// Layout optimization backend (`default: BackboneLayoutMode::StressMajorization`).
    pub layout_mode: BackboneLayoutMode,
    /// Maximum layout optimization iterations (`default: 100`).
    pub iterations: usize,
    /// Deterministic RNG seed used when `layout_mode == BackboneLayoutMode::Sgd` (`default: 42`).
    pub seed: u64,
}

impl Default for BackboneConfig {
    fn default() -> Self {
        Self {
            retain_ratio: 0.2,
            umst_weight_scale: 1.0,
            non_backbone_weight_scale: 0.0,
            layout_mode: BackboneLayoutMode::StressMajorization,
            iterations: 100,
            seed: 42,
        }
    }
}

impl BackboneConfig {
    /// Validates all backbone extraction and layout hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.retain_ratio.is_finite() {
            return Err(CoreError::non_finite("BackboneConfig::retain_ratio"));
        }
        if !(0.0..=1.0).contains(&self.retain_ratio) {
            return Err(CoreError::invalid_input(format!(
                "retain_ratio must be in [0.0, 1.0], got {}",
                self.retain_ratio
            )));
        }
        if !self.umst_weight_scale.is_finite() {
            return Err(CoreError::non_finite("BackboneConfig::umst_weight_scale"));
        }
        if self.umst_weight_scale < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "umst_weight_scale must be >= 0.0, got {}",
                self.umst_weight_scale
            )));
        }
        if !self.non_backbone_weight_scale.is_finite() {
            return Err(CoreError::non_finite(
                "BackboneConfig::non_backbone_weight_scale",
            ));
        }
        if self.non_backbone_weight_scale < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "non_backbone_weight_scale must be >= 0.0, got {}",
                self.non_backbone_weight_scale
            )));
        }
        Ok(())
    }
}

/// Output of [`extract_quadrilateral_backbone`].
///
/// Canonical edge order: `simmelian_weights` and `edge_weights` are indexed identically to the
/// graph's unique undirected simple edges `(u, v)` with `u < v` sorted lexicographically by
/// `(u, v)`. Both `umst_edges` and `backbone_edges` follow the same `(u, v)` (`u < v`) order.
#[derive(Debug, Clone)]
pub struct BackboneResult {
    /// Simmelian prefix-Jaccard weights $w(u, v) \in [0, 1]$ aligned with canonical undirected
    /// simple edges `(u, v)` (`u < v`) sorted lexicographically.
    pub simmelian_weights: Vec<f64>,
    /// Canonical undirected edges `(u, v)` (`u < v`) belonging to the Union of Maximum Spanning Trees.
    pub umst_edges: Vec<(u32, u32)>,
    /// Canonical undirected edges `(u, v)` (`u < v`) in $E_{\text{UMST}} \cup E_{\text{threshold}}$.
    pub backbone_edges: Vec<(u32, u32)>,
    /// Assigned layout edge weights aligned with `simmelian_weights`: `umst_weight_scale` for
    /// `E_UMST` edges, `1.0` for non-UMST backbone edges, and `non_backbone_weight_scale` for
    /// remaining edges.
    pub edge_weights: Vec<f64>,
    /// Undirected CSR representation of the active backbone graph (unweighted when all active
    /// edges have unit weight `1.0`, weighted otherwise; omits zero-weight edges).
    pub backbone_csr: CsrGraph<f64>,
}

struct SimpleUndirectedGraph {
    num_nodes: usize,
    edges: Vec<(u32, u32)>,
    adj_indptr: Vec<u32>,
    adj_indices: Vec<u32>,
    adj_edge_id: Vec<u32>,
}

fn build_simple_undirected_graph<W: Scalar>(
    graph: &CsrGraph<W>,
) -> CoreResult<SimpleUndirectedGraph> {
    let n = graph.num_nodes();
    if u32::try_from(n).is_err() {
        return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
    }
    if let Some(ws) = graph.weights() {
        for &w in ws {
            let wf = w.to_f64();
            if !wf.is_finite() {
                return Err(CoreError::non_finite("backbone::graph_weights"));
            }
            if wf < 0.0 {
                return Err(CoreError::invalid_input(
                    "graph edge weights must be non-negative",
                ));
            }
        }
    }

    let mut edges: Vec<(u32, u32)> = Vec::with_capacity(graph.num_edges());
    if !graph.is_directed() {
        for u in 0..n {
            let u32_u = u as u32;
            for &v_u32 in graph.neighbors(u) {
                if u32_u < v_u32 {
                    edges.push((u32_u, v_u32));
                }
            }
        }
        edges.sort_unstable();
        edges.dedup();
    } else {
        for u in 0..n {
            let u32_u = u as u32;
            for &v_u32 in graph.neighbors(u) {
                if u32_u != v_u32 {
                    let pair = if u32_u < v_u32 {
                        (u32_u, v_u32)
                    } else {
                        (v_u32, u32_u)
                    };
                    edges.push(pair);
                }
            }
        }
        edges.sort_unstable();
        edges.dedup();
    }

    let m = edges.len();
    let two_m = m
        .checked_mul(2)
        .ok_or_else(|| CoreError::invalid_csr("backbone half-edge count overflow"))?;
    if u32::try_from(two_m).is_err() {
        return Err(CoreError::invalid_csr("half-edge count exceeds u32::MAX"));
    }

    let mut deg = vec![0u32; n];
    for &(u, v) in &edges {
        deg[u as usize] += 1;
        deg[v as usize] += 1;
    }

    let mut adj_indptr = Vec::with_capacity(n + 1);
    adj_indptr.push(0u32);
    let mut acc = 0u32;
    for &d in &deg {
        acc += d;
        adj_indptr.push(acc);
    }

    let mut cursor = adj_indptr[..n].to_vec();
    let mut adj_indices = vec![0u32; two_m];
    let mut adj_edge_id = vec![0u32; two_m];

    for (e_idx, &(u, v)) in edges.iter().enumerate() {
        let e_u32 = e_idx as u32;
        let pu = cursor[u as usize] as usize;
        adj_indices[pu] = v;
        adj_edge_id[pu] = e_u32;
        cursor[u as usize] += 1;

        let pv = cursor[v as usize] as usize;
        adj_indices[pv] = u;
        adj_edge_id[pv] = e_u32;
        cursor[v as usize] += 1;
    }

    Ok(SimpleUndirectedGraph {
        num_nodes: n,
        edges,
        adj_indptr,
        adj_indices,
        adj_edge_id,
    })
}

/// Chiba–Nishizeki $O(m\,\alpha(G))$ quadrangle ($C_4$) counting per edge.
fn count_quadrangles(g: &SimpleUndirectedGraph) -> Vec<u64> {
    let n = g.num_nodes;
    let m = g.edges.len();
    let mut q_edge = vec![0u64; m];
    if n < 4 || m < 4 {
        return q_edge;
    }

    // Sort vertices by non-increasing degree (tie-break deterministically by vertex ID).
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_unstable_by(|&a, &b| {
        let da = g.adj_indptr[a as usize + 1] - g.adj_indptr[a as usize];
        let db = g.adj_indptr[b as usize + 1] - g.adj_indptr[b as usize];
        db.cmp(&da).then_with(|| a.cmp(&b))
    });

    let mut rank = vec![0u32; n];
    for (r, &v) in order.iter().enumerate() {
        rank[v as usize] = r as u32;
    }

    // P[c] stores (b, e_ab, e_bc) for common neighbors b of a and c with rank(b), rank(c) > rank(a).
    let mut buckets: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n];
    let mut touched_c: Vec<usize> = Vec::new();

    for &a_u32 in &order {
        let a = a_u32 as usize;
        let rank_a = rank[a];
        let a_start = g.adj_indptr[a] as usize;
        let a_end = g.adj_indptr[a + 1] as usize;
        if a_end - a_start < 2 {
            continue;
        }

        for pos_ab in a_start..a_end {
            let b = g.adj_indices[pos_ab] as usize;
            if rank[b] <= rank_a {
                continue;
            }
            let e_ab = g.adj_edge_id[pos_ab];
            let b_start = g.adj_indptr[b] as usize;
            let b_end = g.adj_indptr[b + 1] as usize;

            for pos_bc in b_start..b_end {
                let c = g.adj_indices[pos_bc] as usize;
                if c != a && rank[c] > rank_a {
                    if buckets[c].is_empty() {
                        touched_c.push(c);
                    }
                    buckets[c].push((e_ab, g.adj_edge_id[pos_bc]));
                }
            }
        }

        for &c in &touched_c {
            let list = &mut buckets[c];
            let k = list.len();
            if k >= 2 {
                let delta = (k - 1) as u64;
                for &(e_ab, e_bc) in list.iter() {
                    q_edge[e_ab as usize] += delta;
                    q_edge[e_bc as usize] += delta;
                }
            }
            list.clear();
        }
        touched_c.clear();
    }

    q_edge
}

#[derive(Debug, Clone, Copy)]
struct RankedNeighbor {
    nbr: u32,
    q_norm: f64,
    tie_start: usize,
    tie_end: usize,
}

fn compute_simmelian_prefix_jaccard(g: &SimpleUndirectedGraph, q_edge: &[u64]) -> Vec<f64> {
    let n = g.num_nodes;
    let m = g.edges.len();
    if m == 0 {
        return Vec::new();
    }

    let mut q_node = vec![0u128; n];
    for (e_idx, &(u, v)) in g.edges.iter().enumerate() {
        let qv = q_edge[e_idx] as u128;
        q_node[u as usize] += qv;
        q_node[v as usize] += qv;
    }

    let mut q_norm = vec![0.0_f64; m];
    for (e_idx, &(u, v)) in g.edges.iter().enumerate() {
        let q_uv = q_edge[e_idx];
        if q_uv > 0 {
            let qu = q_node[u as usize] as f64;
            let qv = q_node[v as usize] as f64;
            let denom = qu.sqrt() * qv.sqrt();
            if denom > 0.0 && denom.is_finite() {
                q_norm[e_idx] = (q_uv as f64) / denom;
            }
        }
    }

    // Build per-vertex Q-ranked neighbor lists with tie-closed group bounds.
    let mut ranked_lists: Vec<Vec<RankedNeighbor>> = Vec::with_capacity(n);
    let mut max_deg = 0usize;

    for u in 0..n {
        let start = g.adj_indptr[u] as usize;
        let end = g.adj_indptr[u + 1] as usize;
        let deg = end - start;
        max_deg = max_deg.max(deg);
        let mut list = Vec::with_capacity(deg);
        for pos in start..end {
            list.push(RankedNeighbor {
                nbr: g.adj_indices[pos],
                q_norm: q_norm[g.adj_edge_id[pos] as usize],
                tie_start: 0,
                tie_end: 0,
            });
        }
        list.sort_unstable_by(|a, b| {
            b.q_norm
                .total_cmp(&a.q_norm)
                .then_with(|| a.nbr.cmp(&b.nbr))
        });

        let mut grp_start = 0usize;
        while grp_start < deg {
            let anchor = list[grp_start].q_norm;
            let mut grp_end = grp_start + 1;
            while grp_end < deg && (anchor - list[grp_end].q_norm).abs() <= WEIGHT_TIE_EPS {
                grp_end += 1;
            }
            for item in &mut list[grp_start..grp_end] {
                item.tie_start = grp_start;
                item.tie_end = grp_end;
            }
            grp_start = grp_end;
        }
        ranked_lists.push(list);
    }

    // Compute exact tie-invariant maximum prefix Jaccard in O(m * Delta(G)) without hash maps.
    let mut simmelian_weights = vec![0.0_f64; m];
    let mut in_u = VisitStamp::new(n);
    let mut u_tie_start = vec![0usize; n];
    let mut enter_at = vec![0usize; max_deg + 1];

    for u in 0..n {
        let list_u = &ranked_lists[u];
        let deg_u = list_u.len();
        if deg_u == 0 {
            continue;
        }
        in_u.clear();
        for item in list_u {
            let w = item.nbr as usize;
            in_u.mark(w);
            u_tie_start[w] = item.tie_start;
        }

        let start = g.adj_indptr[u] as usize;
        let end = g.adj_indptr[u + 1] as usize;
        for pos in start..end {
            let v = g.adj_indices[pos] as usize;
            if u >= v {
                continue;
            }
            let e_idx = g.adj_edge_id[pos] as usize;
            let list_v = &ranked_lists[v];
            let deg_v = list_v.len();
            let limit = deg_u.min(deg_v);
            if limit == 0 {
                continue;
            }

            enter_at[..=limit].fill(0);
            for item_v in list_v {
                let w = item_v.nbr as usize;
                if in_u.is_visited(w) {
                    let ku = u_tie_start[w] + 1;
                    let kv = item_v.tie_start + 1;
                    let k_enter = ku.max(kv);
                    if k_enter <= limit {
                        enter_at[k_enter] += 1;
                    }
                }
            }

            let mut intersect = 0usize;
            let mut best = 0.0_f64;
            for k in 1..=limit {
                intersect += enter_at[k];
                let size_a = list_u[k - 1].tie_end;
                let size_b = list_v[k - 1].tie_end;
                let union = size_a + size_b - intersect;
                if union > 0 {
                    let jaccard = (intersect as f64) / (union as f64);
                    if jaccard > best {
                        best = jaccard;
                    }
                }
            }
            simmelian_weights[e_idx] = best;
        }
    }

    simmelian_weights
}

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

#[derive(Debug, Clone)]
enum BackboneInnerSolver<S: Scalar> {
    StressMajorization(Box<StressMajorizationSolver<S>>),
    Sgd(Box<SgdSolver<S>>),
}

/// Cooperative layout solver running Stress Majorization or SGD on the Quadrilateral Simmelian
/// Backbone extracted from a graph.
#[derive(Debug, Clone)]
pub struct BackboneSolver<S: Scalar = f64> {
    config: BackboneConfig,
    num_nodes: usize,
    backbone: BackboneResult,
    inner: BackboneInnerSolver<S>,
}

fn build_apsp_distance_matrix(backbone_csr: &CsrGraph<f64>) -> CoreResult<Vec<f64>> {
    let n = backbone_csr.num_nodes();
    let total = n
        .checked_mul(n)
        .ok_or_else(|| CoreError::invalid_input("APSP distance matrix overflow"))?;
    let mut dists = vec![f64::INFINITY; total];
    if n == 0 {
        return Ok(dists);
    }

    if !backbone_csr.is_weighted() {
        let mut bfs = BfsWorkspace::new(n);
        for u in 0..n {
            bfs.bfs_from(backbone_csr, u, &mut dists[u * n..(u + 1) * n])?;
        }
        return Ok(dists);
    }

    // Convert edge strengths w > 0 into shortest-path edge lengths len = 1 / w.
    let orig_w = backbone_csr.weights().unwrap_or(&[]);
    let mut inv_w = Vec::with_capacity(orig_w.len());
    for &w in orig_w {
        let len = 1.0 / w;
        if !len.is_finite() || len <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "backbone edge weight {w} produces non-finite inverse edge length"
            )));
        }
        inv_w.push(len);
    }
    let length_graph = CsrGraph::from_csr(
        backbone_csr.indptr(),
        backbone_csr.indices(),
        Some(&inv_w),
        false,
        false,
    )?;
    let mut dijk = DijkstraWorkspace::new(n);
    for u in 0..n {
        dijk.dijkstra_from(&length_graph, u, &mut dists[u * n..(u + 1) * n])?;
    }
    Ok(dists)
}

impl<S: Scalar> BackboneSolver<S> {
    /// Extracts the quadrilateral Simmelian backbone from `graph` and initializes the layout solver.
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: BackboneConfig) -> CoreResult<Self> {
        config.validate()?;
        let n = graph.num_nodes();
        let backbone = extract_quadrilateral_backbone(graph, &config)?;

        let inner = match config.layout_mode {
            BackboneLayoutMode::StressMajorization => {
                let sm_cfg = StressMajorizationConfig {
                    max_iterations: config.iterations,
                    ..Default::default()
                };
                let solver = if !backbone.backbone_csr.is_weighted() {
                    StressMajorizationSolver::from_graph(&backbone.backbone_csr, sm_cfg)?
                } else {
                    let dists = build_apsp_distance_matrix(&backbone.backbone_csr)?;
                    StressMajorizationSolver::from_distance_matrix(n, &dists, sm_cfg)?
                };
                BackboneInnerSolver::StressMajorization(Box::new(solver))
            }
            BackboneLayoutMode::Sgd => {
                let sched = SgdScheduleConfig {
                    t_max: config.iterations,
                    ..Default::default()
                };
                let solver = if !backbone.backbone_csr.is_weighted() {
                    SgdSolver::full(&backbone.backbone_csr, sched, config.seed)?
                } else {
                    let dists = build_apsp_distance_matrix(&backbone.backbone_csr)?;
                    let mut src = Vec::new();
                    let mut tgt = Vec::new();
                    let mut dij = Vec::new();
                    let mut wij = Vec::new();
                    for i in 0..n {
                        for j in (i + 1)..n {
                            let d = dists[i * n + j];
                            if d.is_finite() && d > 0.0 {
                                let w = 1.0 / (d * d);
                                let d_s = S::from_f64(d);
                                let w_s = S::from_f64(w);
                                if d_s.is_finite()
                                    && d_s > S::ZERO
                                    && w_s.is_finite()
                                    && w_s > S::ZERO
                                {
                                    src.push(i as u32);
                                    tgt.push(j as u32);
                                    dij.push(d_s);
                                    wij.push(w_s);
                                }
                            }
                        }
                    }
                    let dji = dij.clone();
                    let wji = wij.clone();
                    SgdSolver::from_pairs(n, src, tgt, dij, dji, wij, wji, sched, config.seed)?
                };
                BackboneInnerSolver::Sgd(Box::new(solver))
            }
        };

        Ok(Self {
            config,
            num_nodes: n,
            backbone,
            inner,
        })
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &BackboneConfig {
        &self.config
    }

    /// Returns the expected number of vertices in [`Layout2d`].
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the extracted [`BackboneResult`].
    #[inline]
    pub fn backbone_result(&self) -> &BackboneResult {
        &self.backbone
    }
}

impl<S: Scalar> IterativeSolver<S> for BackboneSolver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        match &mut self.inner {
            BackboneInnerSolver::StressMajorization(s) => s.step(layout),
            BackboneInnerSolver::Sgd(s) => s.step(layout),
        }
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        match &self.inner {
            BackboneInnerSolver::StressMajorization(s) => s.status(),
            BackboneInnerSolver::Sgd(s) => s.status(),
        }
    }

    #[inline]
    fn iteration(&self) -> usize {
        match &self.inner {
            BackboneInnerSolver::StressMajorization(s) => s.iteration(),
            BackboneInnerSolver::Sgd(s) => s.iteration(),
        }
    }

    #[inline]
    fn total_iterations(&self) -> usize {
        match &self.inner {
            BackboneInnerSolver::StressMajorization(s) => s.total_iterations(),
            BackboneInnerSolver::Sgd(s) => s.total_iterations(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egraph_core::DeterministicRng;

    fn seeded_layout<S: Scalar>(n: usize, scale: f64, seed: u64) -> Layout2d<S> {
        let mut rng = DeterministicRng::from_seed_u64(seed);
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            x.push(S::from_f64((rng.next_f64() * 2.0 - 1.0) * scale));
            y.push(S::from_f64((rng.next_f64() * 2.0 - 1.0) * scale));
        }
        Layout2d::from_xy(x, y).unwrap()
    }

    fn brute_force_quadrangles(n: usize, edges: &[(u32, u32)]) -> Vec<u64> {
        let mut adj = vec![vec![false; n]; n];
        let mut edge_id = vec![vec![usize::MAX; n]; n];
        for (idx, &(u, v)) in edges.iter().enumerate() {
            adj[u as usize][v as usize] = true;
            adj[v as usize][u as usize] = true;
            edge_id[u as usize][v as usize] = idx;
            edge_id[v as usize][u as usize] = idx;
        }
        let mut counts = vec![0u64; edges.len()];
        // Enumerate all 4-subsets {a, b, c, d} and each of the 3 possible 4-cycle orders.
        for a in 0..n {
            for b in (a + 1)..n {
                for c in (b + 1)..n {
                    for d in (c + 1)..n {
                        let cycles = [[a, b, c, d], [a, b, d, c], [a, c, b, d]];
                        for cyc in cycles {
                            if adj[cyc[0]][cyc[1]]
                                && adj[cyc[1]][cyc[2]]
                                && adj[cyc[2]][cyc[3]]
                                && adj[cyc[3]][cyc[0]]
                            {
                                counts[edge_id[cyc[0]][cyc[1]]] += 1;
                                counts[edge_id[cyc[1]][cyc[2]]] += 1;
                                counts[edge_id[cyc[2]][cyc[3]]] += 1;
                                counts[edge_id[cyc[3]][cyc[0]]] += 1;
                            }
                        }
                    }
                }
            }
        }
        counts
    }

    fn brute_force_simmelian_prefix_jaccard(g: &SimpleUndirectedGraph, q_edge: &[u64]) -> Vec<f64> {
        let n = g.num_nodes;
        let m = g.edges.len();
        let mut q_node = vec![0u128; n];
        for (idx, &(u, v)) in g.edges.iter().enumerate() {
            q_node[u as usize] += q_edge[idx] as u128;
            q_node[v as usize] += q_edge[idx] as u128;
        }
        let mut q_norm = vec![0.0_f64; m];
        for (idx, &(u, v)) in g.edges.iter().enumerate() {
            if q_edge[idx] > 0 {
                let denom = (q_node[u as usize] as f64).sqrt() * (q_node[v as usize] as f64).sqrt();
                if denom > 0.0 {
                    q_norm[idx] = (q_edge[idx] as f64) / denom;
                }
            }
        }
        let mut lists: Vec<Vec<(u32, f64)>> = Vec::with_capacity(n);
        for u in 0..n {
            let s = g.adj_indptr[u] as usize;
            let e = g.adj_indptr[u + 1] as usize;
            let mut items: Vec<(u32, f64)> = (s..e)
                .map(|p| (g.adj_indices[p], q_norm[g.adj_edge_id[p] as usize]))
                .collect();
            items.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            lists.push(items);
        }
        let tie_prefix = |list: &[(u32, f64)], k: usize| -> Vec<u32> {
            let mut grp_start = 0usize;
            while grp_start < list.len() {
                let anchor = list[grp_start].1;
                let mut grp_end = grp_start + 1;
                while grp_end < list.len() && (anchor - list[grp_end].1).abs() <= WEIGHT_TIE_EPS {
                    grp_end += 1;
                }
                if k <= grp_end {
                    return list[..grp_end].iter().map(|x| x.0).collect();
                }
                grp_start = grp_end;
            }
            list.iter().map(|x| x.0).collect()
        };
        let mut out = vec![0.0_f64; m];
        for (idx, &(u, v)) in g.edges.iter().enumerate() {
            let lu = &lists[u as usize];
            let lv = &lists[v as usize];
            let limit = lu.len().min(lv.len());
            let mut best = 0.0_f64;
            for k in 1..=limit {
                let au = tie_prefix(lu, k);
                let bv = tie_prefix(lv, k);
                let inter = au.iter().filter(|x| bv.contains(x)).count();
                let union = au.len() + bv.len() - inter;
                if union > 0 {
                    best = best.max((inter as f64) / (union as f64));
                }
            }
            out[idx] = best;
        }
        out
    }

    #[test]
    fn test_quadrangle_counts_known_graphs_and_brute_force_oracle() {
        // 1. C_4: each edge belongs to 1 quadrangle.
        let c4 =
            CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3), (3, 0)], None, false, false)
                .unwrap();
        let s_c4 = build_simple_undirected_graph(&c4).unwrap();
        assert_eq!(count_quadrangles(&s_c4), vec![1, 1, 1, 1]);

        // 2. K_4: each of the 6 edges belongs to 2 quadrangles.
        let k4_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        let k4 = CsrGraph::<f64>::from_edges(4, &k4_edges, None, false, false).unwrap();
        let s_k4 = build_simple_undirected_graph(&k4).unwrap();
        assert_eq!(count_quadrangles(&s_k4), vec![2; 6]);

        // 3. K_{2, 4}: each of the 8 edges belongs to 3 quadrangles.
        let mut k24_edges = Vec::new();
        for b in 2..6u32 {
            k24_edges.push((0, b));
            k24_edges.push((1, b));
        }
        let k24 = CsrGraph::<f64>::from_edges(6, &k24_edges, None, false, false).unwrap();
        let s_k24 = build_simple_undirected_graph(&k24).unwrap();
        assert_eq!(count_quadrangles(&s_k24), vec![3; 8]);

        // 4. Randomized dense & sparse graphs vs brute-force C_4 oracle.
        let mut rng = DeterministicRng::from_seed_u64(1337);
        for n in [6usize, 9, 12] {
            for p in [0.25, 0.55, 0.8] {
                let mut edges = Vec::new();
                for u in 0..n as u32 {
                    for v in (u + 1)..n as u32 {
                        if rng.next_f64() < p {
                            edges.push((u, v));
                        }
                    }
                }
                let g = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();
                let sg = build_simple_undirected_graph(&g).unwrap();
                let fast = count_quadrangles(&sg);
                let oracle = brute_force_quadrangles(n, &sg.edges);
                assert_eq!(fast, oracle, "C_4 mismatch for n={n}, p={p}");
                let fast_w = compute_simmelian_prefix_jaccard(&sg, &fast);
                let oracle_w = brute_force_simmelian_prefix_jaccard(&sg, &oracle);
                for (fw, ow) in fast_w.iter().zip(oracle_w.iter()) {
                    assert!(
                        (fw - ow).abs() < 1e-12,
                        "prefix-Jaccard mismatch: fast={fw}, oracle={ow}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_simmelian_weights_and_tie_invariant_umst_symmetry() {
        // K_4: all 6 edges are structurally identical -> identical Simmelian weights (0.5) and all 6 in UMST.
        let k4_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        let k4 = CsrGraph::<f64>::from_edges(4, &k4_edges, None, false, false).unwrap();
        let res = extract_quadrilateral_backbone(
            &k4,
            &BackboneConfig {
                retain_ratio: 0.0,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(res.simmelian_weights.len(), 6);
        for &w in &res.simmelian_weights {
            assert!((w - 0.5).abs() < 1e-12, "expected 0.5, got {w}");
        }
        assert_eq!(res.umst_edges.len(), 6);
        assert_eq!(res.backbone_edges.len(), 6);
        assert_eq!(res.edge_weights, vec![1.0; 6]);
        assert!(!res.backbone_csr.is_weighted());
        assert_eq!(res.backbone_csr.num_edges(), 6);

        // Two K_4 cliques bridged by (3, 4): UMST must retain connectivity via (3, 4) and all symmetric clique edges.
        let mut barbell = Vec::new();
        for u in 0..4u32 {
            for v in (u + 1)..4u32 {
                barbell.push((u, v));
            }
        }
        for u in 4..8u32 {
            for v in (u + 1)..8u32 {
                barbell.push((u, v));
            }
        }
        barbell.push((3, 4));
        let g = CsrGraph::<f64>::from_edges(8, &barbell, None, false, false).unwrap();
        let b_res = extract_quadrilateral_backbone(
            &g,
            &BackboneConfig {
                retain_ratio: 0.0,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(b_res.umst_edges.contains(&(3, 4)));
        // Backbone CSR is connected (1 component).
        assert_eq!(
            b_res.backbone_csr.largest_connected_component_nodes().len(),
            8
        );
    }

    #[test]
    fn test_retain_ratio_and_custom_edge_weights() {
        // Planted 2-clique + 2 cross edges (bridge-like noisy hairball).
        let mut edges = Vec::new();
        for u in 0..6u32 {
            for v in (u + 1)..6u32 {
                edges.push((u, v));
            }
        }
        for u in 6..12u32 {
            for v in (u + 1)..12u32 {
                edges.push((u, v));
            }
        }
        edges.push((0, 6));
        edges.push((1, 7));
        let g = CsrGraph::<f64>::from_edges(12, &edges, None, false, false).unwrap();
        let m = g.num_edges();

        // retain_ratio = 0.2, umst_weight_scale = 2.0, non_backbone_weight_scale = 0.1
        let cfg = BackboneConfig {
            retain_ratio: 0.2,
            umst_weight_scale: 2.0,
            non_backbone_weight_scale: 0.1,
            ..Default::default()
        };
        let res = extract_quadrilateral_backbone(&g, &cfg).unwrap();
        assert_eq!(res.simmelian_weights.len(), m);
        assert_eq!(res.edge_weights.len(), m);
        assert!(res.backbone_edges.len() >= res.umst_edges.len());
        assert!(res.backbone_csr.is_weighted());
        assert_eq!(res.backbone_csr.num_edges(), m);
        assert_eq!(
            res.backbone_csr.largest_connected_component_nodes().len(),
            12
        );

        // Intra-cluster edges must strictly dominate inter-cluster edges in Simmelian weight.
        let simple = build_simple_undirected_graph(&g).unwrap();
        let mut min_intra = f64::INFINITY;
        let mut max_inter = 0.0_f64;
        for (idx, &(u, v)) in simple.edges.iter().enumerate() {
            let same = (u < 6) == (v < 6);
            let w = res.simmelian_weights[idx];
            if same {
                min_intra = min_intra.min(w);
            } else {
                max_inter = max_inter.max(w);
            }
        }
        assert!(
            min_intra > max_inter,
            "expected intra ({min_intra}) > inter ({max_inter})"
        );

        // retain_ratio = 1.0 keeps all edges in backbone_edges.
        let full_res = extract_quadrilateral_backbone(
            &g,
            &BackboneConfig {
                retain_ratio: 1.0,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(full_res.backbone_edges.len(), m);
    }

    #[test]
    fn test_backbone_solver_step_vs_run_bit_exact_modes_and_dtypes() {
        let mut edges = Vec::new();
        for u in 0..5u32 {
            for v in (u + 1)..5u32 {
                edges.push((u, v));
            }
        }
        for u in 5..10u32 {
            for v in (u + 1)..10u32 {
                edges.push((u, v));
            }
        }
        edges.push((4, 5));
        let g = CsrGraph::<f64>::from_edges(10, &edges, None, false, false).unwrap();

        for mode in [
            BackboneLayoutMode::StressMajorization,
            BackboneLayoutMode::Sgd,
        ] {
            let cfg = BackboneConfig {
                retain_ratio: 0.25,
                umst_weight_scale: 1.5,
                non_backbone_weight_scale: 0.2,
                layout_mode: mode,
                iterations: 25,
                seed: 77,
            };

            // f64 bit-exact check
            let mut s_run = BackboneSolver::<f64>::new(&g, cfg).unwrap();
            let mut s_step = BackboneSolver::<f64>::new(&g, cfg).unwrap();
            let mut l_run = seeded_layout::<f64>(10, 1.0, 99);
            let mut l_step = l_run.clone();

            assert_eq!(s_run.total_iterations(), 25);
            assert!(s_run.run(&mut l_run).unwrap().is_finished());
            assert!(l_run.is_all_finite());
            let _ = s_step.run_steps(&mut l_step, 5).unwrap();
            while !s_step.status().is_finished() {
                s_step.step(&mut l_step).unwrap();
            }
            assert_eq!(s_run.status(), s_step.status());
            assert_eq!(s_run.iteration(), s_step.iteration());
            for i in 0..10 {
                assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
                assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
            }

            // f32 bit-exact check
            let mut s32_run = BackboneSolver::<f32>::new(&g, cfg).unwrap();
            let mut s32_step = BackboneSolver::<f32>::new(&g, cfg).unwrap();
            let mut l32_run = seeded_layout::<f32>(10, 1.0, 99);
            let mut l32_step = l32_run.clone();
            s32_run.run(&mut l32_run).unwrap();
            while !s32_step.status().is_finished() {
                s32_step.step(&mut l32_step).unwrap();
            }
            assert!(l32_run.is_all_finite());
            for i in 0..10 {
                assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
                assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
            }
        }
    }

    #[test]
    fn test_backbone_degenerate_directed_and_validation_errors() {
        let cfg = BackboneConfig::default();

        // Empty graph
        let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        let res0 = extract_quadrilateral_backbone(&empty, &cfg).unwrap();
        assert!(res0.simmelian_weights.is_empty());
        assert!(res0.backbone_edges.is_empty());
        let mut s0 = BackboneSolver::<f64>::new(&empty, cfg).unwrap();
        let mut l0 = Layout2d::<f64>::zeros(0);
        assert!(s0.run(&mut l0).unwrap().is_finished());

        // Singleton graph
        let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
        let mut s1 = BackboneSolver::<f64>::new(&single, cfg).unwrap();
        let mut l1 = Layout2d::from_xy(vec![2.0], vec![-3.0]).unwrap();
        assert!(s1.run(&mut l1).unwrap().is_finished());
        assert_eq!(l1.x(), &[2.0]);
        assert_eq!(l1.y(), &[-3.0]);

        // Edgeless + disconnected + isolated vertices
        let disc = CsrGraph::<f64>::from_edges(5, &[(0, 1), (2, 3)], None, false, false).unwrap();
        let mut s_disc = BackboneSolver::<f64>::new(&disc, cfg).unwrap();
        let mut l_disc = seeded_layout::<f64>(5, 1.0, 5);
        assert!(s_disc.run(&mut l_disc).unwrap().is_finished());
        assert!(l_disc.is_all_finite());

        // Directed graph with reciprocal edges and self-loops matches undirected counterpart.
        let undir = CsrGraph::<f64>::from_edges(
            4,
            &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
            None,
            false,
            false,
        )
        .unwrap();
        let dir = CsrGraph::<f64>::from_edges(
            4,
            &[
                (0, 0),
                (0, 1),
                (1, 0),
                (0, 2),
                (0, 3),
                (1, 2),
                (2, 1),
                (1, 3),
                (2, 3),
                (3, 3),
            ],
            None,
            true,
            true,
        )
        .unwrap();
        let r_u = extract_quadrilateral_backbone(&undir, &cfg).unwrap();
        let r_d = extract_quadrilateral_backbone(&dir, &cfg).unwrap();
        assert_eq!(r_u.simmelian_weights, r_d.simmelian_weights);
        assert_eq!(r_u.umst_edges, r_d.umst_edges);
        assert_eq!(r_u.backbone_edges, r_d.backbone_edges);

        // iterations == 0 completes immediately without touching layout.
        let mut s_zero = BackboneSolver::<f64>::new(
            &undir,
            BackboneConfig {
                iterations: 0,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(s_zero.status(), SolverStatus::Completed);
        let mut l_zero = seeded_layout::<f64>(4, 1.0, 1);
        let before = l_zero.clone();
        assert_eq!(s_zero.run(&mut l_zero).unwrap(), SolverStatus::Completed);
        assert_eq!(l_zero, before);

        // Invalid config checks
        for bad in [
            BackboneConfig {
                retain_ratio: f64::NAN,
                ..Default::default()
            },
            BackboneConfig {
                retain_ratio: -0.1,
                ..Default::default()
            },
            BackboneConfig {
                retain_ratio: 1.1,
                ..Default::default()
            },
            BackboneConfig {
                umst_weight_scale: -1.0,
                ..Default::default()
            },
            BackboneConfig {
                umst_weight_scale: f64::INFINITY,
                ..Default::default()
            },
            BackboneConfig {
                non_backbone_weight_scale: -0.1,
                ..Default::default()
            },
            BackboneConfig {
                non_backbone_weight_scale: f64::NAN,
                ..Default::default()
            },
        ] {
            assert!(extract_quadrilateral_backbone(&undir, &bad).is_err());
        }

        // Negative input weights rejected
        let neg_w =
            CsrGraph::from_csr(&[0, 1, 2], &[1, 0], Some(&[-1.0, -1.0]), false, false).unwrap();
        assert!(extract_quadrilateral_backbone(&neg_w, &cfg).is_err());

        // Layout dimension mismatch rejected
        let mut wrong_l = Layout2d::<f64>::zeros(2);
        assert!(matches!(
            s_disc.step(&mut wrong_l),
            Err(CoreError::DimensionMismatch { .. })
        ));
    }
}
