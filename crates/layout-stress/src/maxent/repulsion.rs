use egraph_core::{
    CoreError, CoreResult, EuclideanLayout, EuclideanStorageRef, Layout2d, Layout3d, LayoutNd,
    LayoutStorage, RepulsionBackend, Scalar,
};
use egraph_spatial_trees::{ArenaOctree3d, ArenaQuadtree2d};

/// Exact $O(N^2)$ pairwise repulsion backend for [`MaxentStressSolver`](super::MaxentStressSolver).
#[non_exhaustive]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Exact {}

impl Exact {
    /// Creates a new [`Exact`] repulsion backend.
    #[inline]
    pub fn new() -> Self {
        Self {}
    }
}

/// Barnes-Hut spatial-tree repulsion backend (`ArenaQuadtree2d` on [`Layout2d`], `ArenaOctree3d`
/// on [`Layout3d`]) for [`MaxentStressSolver`](super::MaxentStressSolver).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct BarnesHut {
    theta: f64,
}

impl Default for BarnesHut {
    #[inline]
    fn default() -> Self {
        Self { theta: 1.2 }
    }
}

impl BarnesHut {
    /// Creates a [`BarnesHut`] repulsion backend with opening angle `theta >= 0.0`.
    #[inline]
    pub fn new(theta: f64) -> Self {
        Self { theta }
    }

    /// Sets the Barnes-Hut opening angle `theta >= 0.0`.
    #[inline]
    pub fn with_theta(mut self, theta: f64) -> Self {
        self.theta = theta;
        self
    }

    /// Returns the configured Barnes-Hut opening angle `theta`.
    #[inline]
    pub fn theta(&self) -> f64 {
        self.theta
    }

    /// Validates Barnes-Hut parameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.theta.is_finite() {
            return Err(CoreError::non_finite("BarnesHut::theta"));
        }
        if self.theta < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "BarnesHut::theta must be >= 0.0, got {}",
                self.theta
            )));
        }
        Ok(())
    }
}

/// Open repulsion-backend contract for [`MaxentStressSolver`](super::MaxentStressSolver).
///
/// External crates (such as `egraph-surface`) can implement this trait for custom layouts and
/// repulsion backends without modifying `egraph-layout-stress`.
pub trait MaxentRepulsion<L: LayoutStorage + ?Sized> {
    /// Reusable per-solver repulsion evaluator (e.g. spatial tree + scratch buffers).
    type Evaluator: RepulsionBackend<L> + Clone;

    /// Validates backend-specific hyperparameters before solver construction.
    fn validate(&self) -> CoreResult<()> {
        Ok(())
    }

    /// Constructs the reusable repulsion evaluator for `num_nodes` vertices, repulsion exponent
    /// `q`, and softening squared distance `eps_sq`.
    fn build_evaluator(&self, num_nodes: usize, q: f64, eps_sq: f64)
    -> CoreResult<Self::Evaluator>;
}

/// Associates each [`LayoutStorage`] target type with its default [`MaxentRepulsion`] backend.
pub trait DefaultMaxentBackend: LayoutStorage {
    /// Default repulsion backend selected by [`MaxentStressSolver`](super::MaxentStressSolver)
    /// when `B` is not explicitly specified.
    type Backend: MaxentRepulsion<Self>;
}

impl<S: Scalar> DefaultMaxentBackend for Layout2d<S> {
    type Backend = BarnesHut;
}

impl<S: Scalar> DefaultMaxentBackend for Layout3d<S> {
    type Backend = BarnesHut;
}

impl<S: Scalar> DefaultMaxentBackend for LayoutNd<S> {
    type Backend = Exact;
}

/// Internal `ArenaQuadtree2d` repulsion evaluator for [`BarnesHut`] on [`Layout2d`].
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct Quadtree2dMaxentEvaluator {
    quadtree: ArenaQuadtree2d<f64>,
    theta: f64,
    q: f64,
    eps_sq: f64,
    x_cur: Vec<f64>,
    y_cur: Vec<f64>,
}

impl Quadtree2dMaxentEvaluator {
    fn new(num_nodes: usize, theta: f64, q: f64, eps_sq: f64) -> Self {
        Self {
            quadtree: ArenaQuadtree2d::new(),
            theta,
            q,
            eps_sq,
            x_cur: vec![0.0; num_nodes],
            y_cur: vec![0.0; num_nodes],
        }
    }
}

impl<S: Scalar> RepulsionBackend<Layout2d<S>> for Quadtree2dMaxentEvaluator {
    fn accumulate_repulsion(
        &mut self,
        layout: &Layout2d<S>,
        forces: &mut [f64],
    ) -> CoreResult<Option<f64>> {
        let n = layout.len();
        let expected = n
            .checked_mul(2)
            .ok_or_else(|| CoreError::invalid_input("Maxent repulsion buffer size overflow"))?;
        if forces.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                forces.len(),
                "Quadtree2dMaxentEvaluator::accumulate_repulsion",
            ));
        }
        if self.x_cur.len() != n {
            self.x_cur.resize(n, 0.0);
            self.y_cur.resize(n, 0.0);
        }
        let xs = layout.x();
        let ys = layout.y();
        for i in 0..n {
            self.x_cur[i] = xs[i].to_f64();
            self.y_cur[i] = ys[i].to_f64();
        }
        self.quadtree.build_xy(&self.x_cur, &self.y_cur)?;
        self.quadtree
            .compute_maxent_repulsion_flat(self.theta, self.q, self.eps_sq, forces)?;
        Ok(None)
    }
}

/// Internal `ArenaOctree3d` repulsion evaluator for [`BarnesHut`] on [`Layout3d`].
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct Octree3dMaxentEvaluator {
    octree: ArenaOctree3d<f64>,
    theta: f64,
    q: f64,
    eps_sq: f64,
    x_cur: Vec<f64>,
    y_cur: Vec<f64>,
    z_cur: Vec<f64>,
}

impl Octree3dMaxentEvaluator {
    fn new(num_nodes: usize, theta: f64, q: f64, eps_sq: f64) -> Self {
        Self {
            octree: ArenaOctree3d::new(),
            theta,
            q,
            eps_sq,
            x_cur: vec![0.0; num_nodes],
            y_cur: vec![0.0; num_nodes],
            z_cur: vec![0.0; num_nodes],
        }
    }
}

impl<S: Scalar> RepulsionBackend<Layout3d<S>> for Octree3dMaxentEvaluator {
    fn accumulate_repulsion(
        &mut self,
        layout: &Layout3d<S>,
        forces: &mut [f64],
    ) -> CoreResult<Option<f64>> {
        let n = layout.len();
        let expected = n
            .checked_mul(3)
            .ok_or_else(|| CoreError::invalid_input("Maxent repulsion buffer size overflow"))?;
        if forces.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                forces.len(),
                "Octree3dMaxentEvaluator::accumulate_repulsion",
            ));
        }
        if self.x_cur.len() != n {
            self.x_cur.resize(n, 0.0);
            self.y_cur.resize(n, 0.0);
            self.z_cur.resize(n, 0.0);
        }
        let xs = layout.x();
        let ys = layout.y();
        let zs = layout.z();
        for i in 0..n {
            self.x_cur[i] = xs[i].to_f64();
            self.y_cur[i] = ys[i].to_f64();
            self.z_cur[i] = zs[i].to_f64();
        }
        self.octree
            .build_xyz(&self.x_cur, &self.y_cur, &self.z_cur)?;
        self.octree
            .compute_maxent_repulsion_flat(self.theta, self.q, self.eps_sq, forces)?;
        Ok(None)
    }
}

/// Internal exact $O(N^2)$ repulsion evaluator for [`Exact`] on any [`EuclideanLayout`].
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct ExactMaxentEvaluator {
    q: f64,
    eps_sq: f64,
    cur_flat: Vec<f64>,
}

impl ExactMaxentEvaluator {
    fn new(q: f64, eps_sq: f64) -> Self {
        Self {
            q,
            eps_sq,
            cur_flat: Vec::new(),
        }
    }
}

impl<L: EuclideanLayout> RepulsionBackend<L> for ExactMaxentEvaluator {
    fn accumulate_repulsion(&mut self, layout: &L, forces: &mut [f64]) -> CoreResult<Option<f64>> {
        let n = layout.len();
        let d = layout.dims();
        let expected = n
            .checked_mul(d)
            .ok_or_else(|| CoreError::invalid_input("Maxent repulsion buffer size overflow"))?;
        if forces.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                forces.len(),
                "ExactMaxentEvaluator::accumulate_repulsion",
            ));
        }
        if self.cur_flat.len() != expected {
            self.cur_flat.resize(expected, 0.0);
        }

        match layout.as_euclidean_storage() {
            EuclideanStorageRef::Soa2d { x, y } => {
                for i in 0..n {
                    self.cur_flat[i * 2] = x[i].to_f64();
                    self.cur_flat[i * 2 + 1] = y[i].to_f64();
                }
            }
            EuclideanStorageRef::Soa3d { x, y, z } => {
                for i in 0..n {
                    self.cur_flat[i * 3] = x[i].to_f64();
                    self.cur_flat[i * 3 + 1] = y[i].to_f64();
                    self.cur_flat[i * 3 + 2] = z[i].to_f64();
                }
            }
            EuclideanStorageRef::FlatNd { coords, .. } => {
                for (dst, &src) in self.cur_flat.iter_mut().zip(coords.iter()) {
                    *dst = src.to_f64();
                }
            }
        }

        forces.fill(0.0);
        let is_q_zero = self.q == 0.0;
        let exp = 0.5 * (self.q + 2.0);
        let eps_sq = self.eps_sq;

        for i in 0..n {
            let base_i = i * d;
            for j in (i + 1)..n {
                let base_j = j * d;
                let mut dist_sq = 0.0_f64;
                for a in 0..d {
                    let diff = self.cur_flat[base_i + a] - self.cur_flat[base_j + a];
                    dist_sq += diff * diff;
                }
                let d2 = dist_sq + eps_sq;
                let mult = if is_q_zero {
                    1.0 / d2
                } else {
                    1.0 / d2.powf(exp)
                };
                for a in 0..d {
                    let fa = mult * (self.cur_flat[base_i + a] - self.cur_flat[base_j + a]);
                    forces[base_i + a] += fa;
                    forces[base_j + a] -= fa;
                }
            }
        }

        Ok(None)
    }
}

impl<L: EuclideanLayout> MaxentRepulsion<L> for Exact {
    type Evaluator = ExactMaxentEvaluator;

    #[inline]
    fn build_evaluator(
        &self,
        _num_nodes: usize,
        q: f64,
        eps_sq: f64,
    ) -> CoreResult<Self::Evaluator> {
        Ok(ExactMaxentEvaluator::new(q, eps_sq))
    }
}

impl<S: Scalar> MaxentRepulsion<Layout2d<S>> for BarnesHut {
    type Evaluator = Quadtree2dMaxentEvaluator;

    fn validate(&self) -> CoreResult<()> {
        BarnesHut::validate(self)
    }

    #[inline]
    fn build_evaluator(
        &self,
        num_nodes: usize,
        q: f64,
        eps_sq: f64,
    ) -> CoreResult<Self::Evaluator> {
        self.validate()?;
        Ok(Quadtree2dMaxentEvaluator::new(
            num_nodes, self.theta, q, eps_sq,
        ))
    }
}

impl<S: Scalar> MaxentRepulsion<Layout3d<S>> for BarnesHut {
    type Evaluator = Octree3dMaxentEvaluator;

    fn validate(&self) -> CoreResult<()> {
        BarnesHut::validate(self)
    }

    #[inline]
    fn build_evaluator(
        &self,
        num_nodes: usize,
        q: f64,
        eps_sq: f64,
    ) -> CoreResult<Self::Evaluator> {
        self.validate()?;
        Ok(Octree3dMaxentEvaluator::new(
            num_nodes, self.theta, q, eps_sq,
        ))
    }
}
