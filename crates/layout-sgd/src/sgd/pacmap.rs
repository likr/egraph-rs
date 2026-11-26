use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, Layout2d, Scalar, SolverStatus,
};
use egraph_linalg::{BfsWorkspace, select_maxmin_sp_pivots};
use egraph_neighborhood::DogVaradhanLandmarks;

use super::config::{PacmapGraphConfig, SgdScheduleConfig, SgdScheduleKind};
use super::graph_sampling::{
    build_symmetric_csr_from_sorted_pairs, extract_canonical_undirected_edges,
};
use super::nd_variants::PacmapNdScratch;
use super::pacmap_pairs::{LandmarkTable, collect_pacmap_tier_pairs};
use super::pairs::{MAX_PACKED_NODES, validate_graph_weights};

#[derive(Debug, Clone)]
pub(super) struct PacmapState {
    pub(super) config: PacmapGraphConfig,
    pub(super) nb_pairs: Vec<(u32, u32)>,
    pub(super) nb_weights: Vec<f64>,
    pub(super) mn_pairs: Vec<(u32, u32)>,
    pub(super) fp_pairs: Vec<(u32, u32)>,
    grad_x: Vec<f64>,
    grad_y: Vec<f64>,
    m_x: Vec<f64>,
    m_y: Vec<f64>,
    v_x: Vec<f64>,
    v_y: Vec<f64>,
    pub(super) nd_scratch: Option<PacmapNdScratch>,
}

pub(super) enum PacmapExternalLandmarks<'a> {
    Dog(&'a DogVaradhanLandmarks),
    Distances {
        data: &'a [f64],
        num_landmarks: usize,
    },
}

pub(super) struct BuiltPacmap {
    pub(super) num_nodes: usize,
    pub(super) num_springs: usize,
    pub(super) num_hinges: usize,
    pub(super) state: PacmapState,
    pub(super) schedule_cfg: SgdScheduleConfig,
    pub(super) status: SolverStatus,
    pub(super) rng: DeterministicRng,
}

fn validate_pacmap_external_landmarks(
    n: usize,
    ext_landmarks: Option<&PacmapExternalLandmarks<'_>>,
) -> CoreResult<()> {
    let Some(ext) = ext_landmarks else {
        return Ok(());
    };
    match ext {
        PacmapExternalLandmarks::Dog(landmarks) => {
            if landmarks.num_nodes() != n {
                return Err(CoreError::dimension_mismatch(
                    n,
                    landmarks.num_nodes(),
                    "SgdSolver::pacmap_graph_with_landmarks::landmarks",
                ));
            }
            let k_lm = landmarks.num_landmarks();
            if n > 0 && k_lm == 0 {
                return Err(CoreError::invalid_input(
                    "landmarks must have at least 1 landmark when num_nodes > 0",
                ));
            }
            let expected = n
                .checked_mul(k_lm)
                .ok_or_else(|| CoreError::invalid_input("landmark table size overflow"))?;
            if landmarks.d_dog().len() != expected {
                return Err(CoreError::dimension_mismatch(
                    expected,
                    landmarks.d_dog().len(),
                    "DogVaradhanLandmarks::d_dog",
                ));
            }
            if landmarks.d_dog().iter().any(|v| !v.is_finite()) {
                return Err(CoreError::non_finite("DogVaradhanLandmarks::d_dog"));
            }
            if let Some(dc) = landmarks.d_coarse() {
                if dc.len() != expected {
                    return Err(CoreError::dimension_mismatch(
                        expected,
                        dc.len(),
                        "DogVaradhanLandmarks::d_coarse",
                    ));
                }
                if dc.iter().any(|v| !v.is_finite()) {
                    return Err(CoreError::non_finite("DogVaradhanLandmarks::d_coarse"));
                }
            }
            for &lm in landmarks.landmarks() {
                if (lm as usize) >= n {
                    return Err(CoreError::index_out_of_bounds(
                        lm as usize,
                        n,
                        "DogVaradhanLandmarks::landmarks",
                    ));
                }
            }
        }
        PacmapExternalLandmarks::Distances {
            data,
            num_landmarks,
        } => {
            let k_lm = *num_landmarks;
            if n > 0 && k_lm == 0 {
                return Err(CoreError::invalid_input(
                    "num_landmarks must be >= 1 when num_nodes > 0",
                ));
            }
            let expected = n
                .checked_mul(k_lm)
                .ok_or_else(|| CoreError::invalid_input("landmark table size overflow"))?;
            if data.len() != expected {
                return Err(CoreError::dimension_mismatch(
                    expected,
                    data.len(),
                    "SgdSolver::pacmap_graph_with_landmark_distances::landmark_distances",
                ));
            }
            for &v in *data {
                if !v.is_finite() {
                    return Err(CoreError::non_finite(
                        "SgdSolver::pacmap_graph_with_landmark_distances::landmark_distances",
                    ));
                }
                if v < 0.0 {
                    return Err(CoreError::invalid_input(format!(
                        "landmark_distances must be >= 0.0, got {v}"
                    )));
                }
            }
        }
    }
    Ok(())
}

pub(super) fn build_pacmap_state<W: Scalar>(
    graph: &CsrGraph<W>,
    config: PacmapGraphConfig,
    ext_landmarks: Option<PacmapExternalLandmarks<'_>>,
) -> CoreResult<BuiltPacmap> {
    config.validate()?;
    validate_graph_weights(graph, "SgdSolver::pacmap_graph")?;

    let n = graph.num_nodes();
    if n > MAX_PACKED_NODES {
        return Err(CoreError::invalid_input(format!(
            "num_nodes ({n}) exceeds packed u32 index limit {MAX_PACKED_NODES}"
        )));
    }
    validate_pacmap_external_landmarks(n, ext_landmarks.as_ref())?;

    let total_iters = config.total_iterations();
    let schedule_cfg = SgdScheduleConfig {
        iterations: total_iters,
        eps: 0.01,
        kind: SgdScheduleKind::Constant,
        step_scale: config.learning_rate,
    };
    let status = if total_iters == 0 {
        SolverStatus::Completed
    } else {
        SolverStatus::Initialized
    };

    if n <= 1 {
        return Ok(BuiltPacmap {
            num_nodes: n,
            num_springs: 0,
            num_hinges: 0,
            state: PacmapState {
                config,
                nb_pairs: Vec::new(),
                nb_weights: Vec::new(),
                mn_pairs: Vec::new(),
                fp_pairs: Vec::new(),
                grad_x: Vec::new(),
                grad_y: Vec::new(),
                m_x: Vec::new(),
                m_y: Vec::new(),
                v_x: Vec::new(),
                v_y: Vec::new(),
                nd_scratch: None,
            },
            schedule_cfg,
            status,
            rng: DeterministicRng::from_seed_u64(config.seed),
        });
    }

    let canonical_edges = extract_canonical_undirected_edges(graph);
    let n_canon_edges = canonical_edges.len();
    let mut u1 = Vec::with_capacity(n_canon_edges);
    let mut v1 = Vec::with_capacity(n_canon_edges);
    for &(u, v, _) in &canonical_edges {
        u1.push(u);
        v1.push(v);
    }
    let (adj_indptr, adj_indices) = build_symmetric_csr_from_sorted_pairs(n, &u1, &v1);
    let sym_graph = CsrGraph::<f64>::from_csr(&adj_indptr, &adj_indices, None, false, false)?;

    // Optional per-node weighted 1-hop adjacency when input graph carries weights,
    // tie-breaking equal weights by ascending neighbor degree then vertex ID.
    let weighted_rows: Option<Vec<Vec<(f64, u32)>>> = if graph.is_weighted() {
        let mut rows = vec![Vec::new(); n];
        for &(u, v, w) in &canonical_edges {
            rows[u as usize].push((w, v));
            rows[v as usize].push((w, u));
        }
        for row in &mut rows {
            row.sort_unstable_by(|a, b| {
                a.0.total_cmp(&b.0)
                    .then_with(|| {
                        sym_graph
                            .degree(a.1 as usize)
                            .cmp(&sym_graph.degree(b.1 as usize))
                    })
                    .then_with(|| a.1.cmp(&b.1))
            });
        }
        Some(rows)
    } else {
        None
    };
    drop(canonical_edges);
    drop(u1);
    drop(v1);

    let n_nb = config.n_neighbors.min(n - 1);
    let n_mn = ((config.mn_ratio * (config.n_neighbors as f64)).floor() as usize).min(n - 1);
    let n_fp = ((config.fp_ratio * (config.n_neighbors as f64)).floor() as usize).min(n - 1);

    // Use supplied external landmark distances or compute compact BFS landmark distances.
    let lm_table = if n_mn == 0 {
        LandmarkTable::None
    } else {
        match ext_landmarks {
            Some(PacmapExternalLandmarks::Dog(landmarks)) => LandmarkTable::RowMajor {
                data: landmarks.d_dog(),
                num_landmarks: landmarks.num_landmarks(),
            },
            Some(PacmapExternalLandmarks::Distances {
                data,
                num_landmarks,
            }) => LandmarkTable::RowMajor {
                data,
                num_landmarks,
            },
            None => {
                let k_lm = config.num_landmarks.min(n);
                let landmarks = select_maxmin_sp_pivots(&sym_graph, k_lm, config.seed, 2)?;
                let num_lm = landmarks.len();
                let mut lm_dists = vec![f64::INFINITY; num_lm * n];
                let mut bfs = BfsWorkspace::new(n);
                for (idx, &lm) in landmarks.iter().enumerate() {
                    bfs.bfs_from(
                        &sym_graph,
                        lm as usize,
                        &mut lm_dists[idx * n..(idx + 1) * n],
                    )?;
                }
                LandmarkTable::ColMajor {
                    data: lm_dists,
                    num_landmarks: num_lm,
                }
            }
        }
    };

    let mut rng = DeterministicRng::from_seed_u64(config.seed);
    let tiers = collect_pacmap_tier_pairs(
        &sym_graph,
        &adj_indptr,
        weighted_rows.as_deref(),
        &lm_table,
        n,
        n_canon_edges,
        n_nb,
        n_mn,
        n_fp,
        config.keep_all_1hop,
        config.max_1hop_per_node,
        &mut rng,
    );

    let mut nb_pairs = tiers.nb_pairs;
    nb_pairs.sort_unstable();
    nb_pairs.dedup();
    let nb_weights: Vec<f64> = if config.degree_weight_beta > 0.0 {
        let beta = config.degree_weight_beta;
        let deg_scale: Vec<f64> = (0..n)
            .map(|u| (sym_graph.degree(u).max(1) as f64).powf(-beta))
            .collect();
        nb_pairs
            .iter()
            .map(|&(u, v)| deg_scale[u as usize] * deg_scale[v as usize])
            .collect()
    } else {
        Vec::new()
    };

    let n_springs = nb_pairs.len() + tiers.mn_pairs.len();
    let n_hinges = tiers.fp_pairs.len();

    Ok(BuiltPacmap {
        num_nodes: n,
        num_springs: n_springs,
        num_hinges: n_hinges,
        state: PacmapState {
            config,
            nb_pairs,
            nb_weights,
            mn_pairs: tiers.mn_pairs,
            fp_pairs: tiers.fp_pairs,
            grad_x: vec![0.0; n],
            grad_y: vec![0.0; n],
            m_x: vec![0.0; n],
            m_y: vec![0.0; n],
            v_x: vec![0.0; n],
            v_y: vec![0.0; n],
            nd_scratch: None,
        },
        schedule_cfg,
        status,
        rng,
    })
}

pub(super) fn step_pacmap_state<S: Scalar>(
    st: &mut PacmapState,
    num_nodes: usize,
    iter: usize,
    layout: &mut Layout2d<S>,
) {
    let total = st.config.total_iterations();
    if num_nodes <= 1 || total == 0 || iter >= total {
        return;
    }

    let [p1_iters, p2_iters, _] = st.config.phase_iterations;
    let (w_nb, w_mn, w_fp) = if iter < p1_iters {
        let p1 = if p1_iters <= 1 {
            0.0
        } else {
            (iter as f64) / ((p1_iters - 1) as f64)
        };
        (2.0, 1000.0 * (1.0 - p1) + 3.0 * p1, 1.0)
    } else if iter < p1_iters + p2_iters {
        (3.0, 3.0, 1.0)
    } else {
        (1.0, 0.0, 1.0)
    };

    st.grad_x.fill(0.0);
    st.grad_y.fill(0.0);

    let (xs, ys) = layout.xy_mut();
    if w_nb > 0.0 {
        let scale_nb = 20.0 * w_nb;
        let has_weights = !st.nb_weights.is_empty();
        for (idx, &(u, v)) in st.nb_pairs.iter().enumerate() {
            let ui = u as usize;
            let vi = v as usize;
            let dx = xs[ui].to_f64() - xs[vi].to_f64();
            let dy = ys[ui].to_f64() - ys[vi].to_f64();
            let d2 = dx * dx + dy * dy;
            if d2.is_finite() {
                let denom = 10.0 + d2;
                let pair_w = if has_weights { st.nb_weights[idx] } else { 1.0 };
                let coeff = (scale_nb * pair_w) / (denom * denom);
                if coeff.is_finite() {
                    let gx = coeff * dx;
                    let gy = coeff * dy;
                    st.grad_x[ui] += gx;
                    st.grad_y[ui] += gy;
                    st.grad_x[vi] -= gx;
                    st.grad_y[vi] -= gy;
                }
            }
        }
    }
    if w_mn > 0.0 {
        let scale_mn = 20_000.0 * w_mn;
        for &(u, v) in &st.mn_pairs {
            let ui = u as usize;
            let vi = v as usize;
            let dx = xs[ui].to_f64() - xs[vi].to_f64();
            let dy = ys[ui].to_f64() - ys[vi].to_f64();
            let d2 = dx * dx + dy * dy;
            if d2.is_finite() {
                let denom = 10_000.0 + d2;
                let coeff = scale_mn / (denom * denom);
                if coeff.is_finite() {
                    let gx = coeff * dx;
                    let gy = coeff * dy;
                    st.grad_x[ui] += gx;
                    st.grad_y[ui] += gy;
                    st.grad_x[vi] -= gx;
                    st.grad_y[vi] -= gy;
                }
            }
        }
    }
    if w_fp > 0.0 {
        let scale_fp = -2.0 * w_fp;
        for &(u, v) in &st.fp_pairs {
            let ui = u as usize;
            let vi = v as usize;
            let dx = xs[ui].to_f64() - xs[vi].to_f64();
            let dy = ys[ui].to_f64() - ys[vi].to_f64();
            let d2 = dx * dx + dy * dy;
            if d2.is_finite() {
                let denom = 1.0 + d2;
                let coeff = scale_fp / (denom * denom);
                if coeff.is_finite() {
                    let gx = coeff * dx;
                    let gy = coeff * dy;
                    st.grad_x[ui] += gx;
                    st.grad_y[ui] += gy;
                    st.grad_x[vi] -= gx;
                    st.grad_y[vi] -= gy;
                }
            }
        }
    }

    // Zero-allocation Adam update matching PaCMAP's (beta1=0.9, beta2=0.999) schedule.
    const BETA1: f64 = 0.9;
    const BETA2: f64 = 0.999;
    const ADAM_EPS: f64 = 1e-7;
    let t_step = (iter + 1) as i32;
    let bc1 = (1.0 - BETA1.powi(t_step)).max(1e-12);
    let bc2 = (1.0 - BETA2.powi(t_step)).max(1e-12);
    let lr_t = st.config.learning_rate * bc2.sqrt() / bc1;

    for u in 0..num_nodes {
        let gx = st.grad_x[u];
        let gy = st.grad_y[u];
        st.m_x[u] = BETA1 * st.m_x[u] + (1.0 - BETA1) * gx;
        st.m_y[u] = BETA1 * st.m_y[u] + (1.0 - BETA1) * gy;
        st.v_x[u] = BETA2 * st.v_x[u] + (1.0 - BETA2) * gx * gx;
        st.v_y[u] = BETA2 * st.v_y[u] + (1.0 - BETA2) * gy * gy;

        let step_x = lr_t * st.m_x[u] / (st.v_x[u].sqrt() + ADAM_EPS);
        let step_y = lr_t * st.m_y[u] / (st.v_y[u].sqrt() + ADAM_EPS);
        if step_x.is_finite() {
            let nx = xs[u] - S::from_f64(step_x);
            if nx.is_finite() {
                xs[u] = nx;
            }
        }
        if step_y.is_finite() {
            let ny = ys[u] - S::from_f64(step_y);
            if ny.is_finite() {
                ys[u] = ny;
            }
        }
    }
}
