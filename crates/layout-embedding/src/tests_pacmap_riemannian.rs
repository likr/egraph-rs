use egraph_core::{
    CoreError, CoreResult, IterativeSolver, Layout2d, LayoutStorage, ManifoldLayout2d, MetricSpace,
    RiemannianLayout, SolverStatus,
};

use crate::test_support::{ALL_MANIFOLDS, circle_layout, grid_graph};
use crate::{PacmapGraphConfig, PacmapGraphSolver};

/// Custom `RiemannianLayout` wrapper introducing per-vertex SO(2) tangent-frame rotations and
/// tracking `exp_map_and_transport` calls to verify Riemannian Adam frame invariance.
#[derive(Debug, Clone)]
struct RotatedTangentLayout {
    inner: ManifoldLayout2d<f64>,
    angles: Vec<f64>,
    twist_per_step: f64,
    exp_and_transport_calls: usize,
}

impl RotatedTangentLayout {
    fn new(inner: ManifoldLayout2d<f64>, angles: Vec<f64>, twist_per_step: f64) -> Self {
        Self {
            inner,
            angles,
            twist_per_step,
            exp_and_transport_calls: 0,
        }
    }

    #[inline]
    fn rotate(v: [f64; 2], angle: f64) -> [f64; 2] {
        let (s, c) = angle.sin_cos();
        [c * v[0] - s * v[1], s * v[0] + c * v[1]]
    }
}

impl LayoutStorage for RotatedTangentLayout {
    type Scalar = f64;

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn validate_finite(&self) -> CoreResult<()> {
        self.inner.validate_finite()
    }
}

impl MetricSpace for RotatedTangentLayout {
    fn distance(&self, i: usize, j: usize) -> f64 {
        self.inner.distance(i, j)
    }
}

impl RiemannianLayout for RotatedTangentLayout {
    fn tangent_dim(&self) -> usize {
        2
    }

    fn log_map_into(&self, i: usize, j: usize, out: &mut [f64]) -> CoreResult<()> {
        let mut raw = [0.0_f64; 2];
        self.inner.log_map_into(i, j, &mut raw)?;
        let local = Self::rotate(raw, -self.angles[i]);
        out[0] = local[0];
        out[1] = local[1];
        Ok(())
    }

    fn exp_map(&mut self, i: usize, delta: &[f64]) -> CoreResult<()> {
        let world_delta = Self::rotate([delta[0], delta[1]], self.angles[i]);
        self.inner.exp_map(i, &world_delta)
    }

    fn parallel_transport_into(&self, i: usize, delta: &[f64], vec: &mut [f64]) -> CoreResult<()> {
        let world_delta = Self::rotate([delta[0], delta[1]], self.angles[i]);
        let mut world_vec = Self::rotate([vec[0], vec[1]], self.angles[i]);
        self.inner
            .parallel_transport_into(i, &world_delta, &mut world_vec)?;
        let local = Self::rotate(world_vec, -self.angles[i]);
        vec[0] = local[0];
        vec[1] = local[1];
        Ok(())
    }

    fn exp_map_and_transport(
        &mut self,
        i: usize,
        delta: &[f64],
        vec: &mut [f64],
    ) -> CoreResult<()> {
        self.exp_and_transport_calls += 1;
        let world_delta = Self::rotate([delta[0], delta[1]], self.angles[i]);
        let mut world_vec = Self::rotate([vec[0], vec[1]], self.angles[i]);
        self.inner
            .exp_map_and_transport(i, &world_delta, &mut world_vec)?;
        // Rotate the local frame after stepping to verify atomic momentum transport.
        self.angles[i] += self.twist_per_step;
        let local = Self::rotate(world_vec, -self.angles[i]);
        vec[0] = local[0];
        vec[1] = local[1];
        Ok(())
    }
}

#[test]
fn test_pacmap_graph_so2_tangent_frame_invariance_and_exp_map_and_transport() {
    let grid = grid_graph(3, 4);
    let n = grid.num_nodes();
    let cfg = PacmapGraphConfig::new()
        .with_n_neighbors(6)
        .with_phase_iterations([6, 6, 8])
        .with_learning_rate(0.5)
        .with_num_landmarks(6)
        .with_seed(42);

    // 1. Global non-axis-aligned SO(2) rotation (37 deg) equivariance on Layout2d.
    let theta = 37.0_f64.to_radians();
    let (st, ct) = theta.sin_cos();
    let base_l = circle_layout::<f64>(n, 1.0);
    let rot_x: Vec<f64> = (0..n)
        .map(|i| ct * base_l.x()[i] - st * base_l.y()[i])
        .collect();
    let rot_y: Vec<f64> = (0..n)
        .map(|i| st * base_l.x()[i] + ct * base_l.y()[i])
        .collect();
    let mut l_base = base_l.clone();
    let mut l_rot = Layout2d::from_xy(rot_x, rot_y).unwrap();
    PacmapGraphSolver::<Layout2d<f64>>::new(&grid, cfg.clone())
        .unwrap()
        .run(&mut l_base)
        .unwrap();
    PacmapGraphSolver::<Layout2d<f64>>::new(&grid, cfg.clone())
        .unwrap()
        .run(&mut l_rot)
        .unwrap();
    for i in 0..n {
        let rx = ct * l_base.x()[i] - st * l_base.y()[i];
        let ry = st * l_base.x()[i] + ct * l_base.y()[i];
        assert!(
            (rx - l_rot.x()[i]).abs() < 1e-11 && (ry - l_rot.y()[i]).abs() < 1e-11,
            "Euclidean 37-deg SO(2) equivariance violated at {i}: ({rx}, {ry}) vs ({}, {})",
            l_rot.x()[i],
            l_rot.y()[i]
        );
    }

    // 2. Per-vertex SO(2) tangent-frame rotation + dynamic frame twist across all 4 manifolds.
    let per_node_angles: Vec<f64> = (0..n).map(|i| 0.37 * (i as f64) + 0.19).collect();
    for kind in ALL_MANIFOLDS {
        let init_m = ManifoldLayout2d::<f64>::init(kind, n, 13, 0.35).unwrap();
        let mut ref_layout = init_m.clone();
        let mut rot_layout = RotatedTangentLayout::new(init_m, per_node_angles.clone(), 0.23);

        PacmapGraphSolver::<ManifoldLayout2d<f64>>::new(&grid, cfg.clone())
            .unwrap()
            .run(&mut ref_layout)
            .unwrap();
        PacmapGraphSolver::<RotatedTangentLayout>::new(&grid, cfg.clone())
            .unwrap()
            .run(&mut rot_layout)
            .unwrap();

        assert_eq!(
            rot_layout.exp_and_transport_calls,
            n * cfg.total_iterations(),
            "{kind:?}: expected exp_map_and_transport to be called for every vertex/step"
        );
        // Multi-step SO(2) tangent-frame rotation (20 steps) amplifies 1-2 ULP `sin_cos` +
        // 2x2 rotation rounding to ~1e-10 on Torus2d (`lr=0.5` on unit period `[0, 1)^2`),
        // differing slightly between FMA (`aarch64`) and non-FMA/glibc `libm` (`x86_64`).
        // Genuine non-invariance produces O(1e-2..1e0) drift, so `1e-8` gives ~100x FMA headroom.
        for i in 0..n {
            assert!(
                (ref_layout.x()[i] - rot_layout.inner.x()[i]).abs() < 1e-8
                    && (ref_layout.y()[i] - rot_layout.inner.y()[i]).abs() < 1e-8,
                "{kind:?} tangent-frame SO(2) invariance failed at node {i}: {:?} vs {:?}",
                (ref_layout.x()[i], ref_layout.y()[i]),
                (rot_layout.inner.x()[i], rot_layout.inner.y()[i]),
            );
        }
    }

    // 3. Breakdown transitions solver to SolverStatus::Failed without corrupting iteration().
    #[derive(Debug, Clone)]
    struct FailingTransportLayout {
        inner: Layout2d<f64>,
        fail_after: usize,
        calls: usize,
    }
    impl LayoutStorage for FailingTransportLayout {
        type Scalar = f64;
        fn len(&self) -> usize {
            self.inner.len()
        }
        fn validate_finite(&self) -> CoreResult<()> {
            self.inner.validate_finite()
        }
    }
    impl MetricSpace for FailingTransportLayout {
        fn distance(&self, i: usize, j: usize) -> f64 {
            self.inner.distance(i, j)
        }
    }
    impl RiemannianLayout for FailingTransportLayout {
        fn tangent_dim(&self) -> usize {
            2
        }
        fn log_map_into(&self, i: usize, j: usize, out: &mut [f64]) -> CoreResult<()> {
            self.inner.log_map_into(i, j, out)
        }
        fn exp_map(&mut self, i: usize, delta: &[f64]) -> CoreResult<()> {
            self.inner.exp_map(i, delta)
        }
        fn parallel_transport_into(
            &self,
            i: usize,
            delta: &[f64],
            vec: &mut [f64],
        ) -> CoreResult<()> {
            self.inner.parallel_transport_into(i, delta, vec)
        }
        fn exp_map_and_transport(
            &mut self,
            i: usize,
            delta: &[f64],
            vec: &mut [f64],
        ) -> CoreResult<()> {
            self.calls += 1;
            if self.calls > self.fail_after {
                return Err(CoreError::numerical_breakdown("injected breakdown"));
            }
            self.inner.exp_map_and_transport(i, delta, vec)
        }
    }

    let mut fail_target = FailingTransportLayout {
        inner: circle_layout::<f64>(n, 1.0),
        fail_after: n + 2,
        calls: 0,
    };
    let mut fail_solver = PacmapGraphSolver::<FailingTransportLayout>::new(&grid, cfg).unwrap();
    assert_eq!(
        fail_solver.step(&mut fail_target).unwrap(),
        SolverStatus::Running
    );
    assert!(matches!(
        fail_solver.step(&mut fail_target),
        Err(CoreError::NumericalBreakdown(_))
    ));
    assert_eq!(fail_solver.status(), SolverStatus::Failed);
    assert!(fail_solver.status().is_failed());
    assert_eq!(fail_solver.iteration(), 1);
    assert_eq!(
        fail_solver.step(&mut fail_target).unwrap(),
        SolverStatus::Failed
    );
    assert_eq!(fail_solver.iteration(), 1);
}
