use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, Scalar, SolverStatus};
use egraph_neighborhood::KnnGraph;
use egraph_spatial_fft::{Fft2dWorkspace, IbFft2d};
use egraph_spatial_trees::ArenaQuadtree2d;

use super::config::{FieldRepulsionBackend, TFdpConfig};
use super::quadtree::FieldQuadtree;
use super::solver::{FieldLayoutSolver, FieldSolverMode};

impl<S: Scalar> FieldLayoutSolver<S> {
    /// Constructs a `t-FDP` [`FieldLayoutSolver`] directly from a [`CsrGraph`].
    pub fn t_fdp<W: Scalar>(graph: &CsrGraph<W>, config: TFdpConfig) -> CoreResult<Self> {
        config.validate()?;
        let n = graph.num_nodes();
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_csr("n_nodes exceeds u32::MAX"));
        }
        if let Some(ws) = graph.weights() {
            for &w in ws {
                let wf = w.to_f64();
                if !wf.is_finite() {
                    return Err(CoreError::non_finite("FieldLayoutSolver::t_fdp::weights"));
                }
                if wf < 0.0 {
                    return Err(CoreError::invalid_input(format!(
                        "graph edge weights must be >= 0.0, got {wf}"
                    )));
                }
            }
        }

        let (indptr, indices, edge_weights) = extract_symmetric_csr_edges(graph);
        Self::build_t_fdp_inner(n, indptr, indices, edge_weights, config)
    }

    /// Constructs a `t-FDP` [`FieldLayoutSolver`] from a validated [`KnnGraph`], converting
    /// neighbor distances $d_{ij} \ge 0$ to symmetric weights $1 / (1 + d_{ij})$.
    pub fn t_fdp_from_knn(knn: &KnnGraph, config: TFdpConfig) -> CoreResult<Self> {
        config.validate()?;
        let n = knn.num_nodes();
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_csr("n_nodes exceeds u32::MAX"));
        }

        let mut pairs: Vec<(u32, u32, f32)> = Vec::with_capacity(knn.nnz() * 2);
        for u in 0..n {
            let u_u32 = u as u32;
            let (nbrs, dists) = knn.row(u);
            for (&v, &d) in nbrs.iter().zip(dists.iter()) {
                let v_usize = v as usize;
                if v_usize == u || v_usize >= n {
                    continue;
                }
                if !d.is_finite() {
                    return Err(CoreError::non_finite(
                        "FieldLayoutSolver::t_fdp_from_knn::distances",
                    ));
                }
                if d < 0.0 {
                    return Err(CoreError::invalid_input(format!(
                        "KnnGraph distances must be >= 0.0, got {d}"
                    )));
                }
                let w = (1.0 / (1.0 + d)) as f32;
                pairs.push((u_u32, v, w));
                pairs.push((v, u_u32, w));
            }
        }

        let (indptr, indices, edge_weights) = symmetrize_pair_list(n, pairs);
        Self::build_t_fdp_inner(n, indptr, indices, edge_weights, config)
    }

    fn build_t_fdp_inner(
        n: usize,
        indptr: Vec<usize>,
        indices: Vec<u32>,
        edge_weights: Vec<f32>,
        config: TFdpConfig,
    ) -> CoreResult<Self> {
        let ibfft = if config.backend == FieldRepulsionBackend::IbFft {
            Some(IbFft2d::new(config.ibfft)?)
        } else {
            None
        };
        let total_iterations = config.total_iterations;
        let status = if total_iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };
        let buf_len = if config.backend == FieldRepulsionBackend::IbFft {
            n
        } else {
            0
        };
        let total_w: f64 = edge_weights.iter().map(|&w| w as f64).sum();
        let avg_deg = if n > 0 {
            (total_w / (n as f64)).max(1.0)
        } else {
            1.0
        };
        let rep_scale = if n > 1 {
            avg_deg / ((n - 1) as f64)
        } else {
            1.0
        };

        Ok(Self {
            mode: FieldSolverMode::TFdp { config, rep_scale },
            backend: config.backend,
            num_nodes: n,
            indptr,
            indices,
            edge_weights,
            hub_eta: Vec::new(),
            attr_x: vec![0.0; n],
            attr_y: vec![0.0; n],
            vel_x: vec![0.0; n],
            vel_y: vec![0.0; n],
            next_x: vec![0.0; n],
            next_y: vec![0.0; n],
            phi_1: vec![S::ZERO; buf_len],
            rep_x: vec![S::ZERO; n],
            rep_y: vec![S::ZERO; n],
            fft_ws: Fft2dWorkspace::new(),
            ibfft,
            quadtree: ArenaQuadtree2d::new(),
            field_quadtree: FieldQuadtree::new(),
            rng: DeterministicRng::from_seed_u64(config.seed),
            iteration: 0,
            total_iterations,
            status,
        })
    }
}

fn extract_symmetric_csr_edges<W: Scalar>(graph: &CsrGraph<W>) -> (Vec<usize>, Vec<u32>, Vec<f32>) {
    let n = graph.num_nodes();
    if n == 0 {
        return (vec![0], Vec::new(), Vec::new());
    }
    let mut pairs: Vec<(u32, u32, f32)> = Vec::with_capacity(if graph.is_directed() {
        graph.nnz() * 2
    } else {
        graph.nnz()
    });
    for u in 0..n {
        let u_u32 = u as u32;
        let nbrs = graph.neighbors(u);
        let w_opt = graph.neighbor_weights(u);
        for (idx, &v) in nbrs.iter().enumerate() {
            let v_usize = v as usize;
            if v_usize == u || v_usize >= n {
                continue;
            }
            let w = w_opt.map_or(1.0, |ws| ws[idx].to_f64());
            if w <= 0.0 {
                continue;
            }
            let w_f32 = w as f32;
            pairs.push((u_u32, v, w_f32));
            if graph.is_directed() {
                pairs.push((v, u_u32, w_f32));
            }
        }
    }
    symmetrize_pair_list(n, pairs)
}

fn symmetrize_pair_list(
    n: usize,
    mut pairs: Vec<(u32, u32, f32)>,
) -> (Vec<usize>, Vec<u32>, Vec<f32>) {
    if pairs.is_empty() {
        return (vec![0; n + 1], Vec::new(), Vec::new());
    }
    pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    let mut deg = vec![0usize; n];
    let mut write_idx = 0usize;
    let mut pos = 0usize;
    let len = pairs.len();
    while pos < len {
        let (u, v, first_w) = pairs[pos];
        let mut sum_w = first_w as f64;
        let mut count = 1usize;
        pos += 1;
        while pos < len && pairs[pos].0 == u && pairs[pos].1 == v {
            sum_w += pairs[pos].2 as f64;
            count += 1;
            pos += 1;
        }
        let avg_w = (sum_w / (count as f64)) as f32;
        if avg_w > 0.0 {
            pairs[write_idx] = (u, v, avg_w);
            write_idx += 1;
            deg[u as usize] += 1;
        }
    }
    pairs.truncate(write_idx);

    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0);
    let mut total = 0usize;
    for &d in &deg {
        total += d;
        indptr.push(total);
    }
    drop(deg);
    let mut cursor = indptr[..n].to_vec();
    let mut indices = vec![0u32; total];
    let mut weights = vec![0.0_f32; total];
    for (u, v, w) in pairs {
        let p = cursor[u as usize];
        indices[p] = v;
        weights[p] = w;
        cursor[u as usize] += 1;
    }
    (indptr, indices, weights)
}
