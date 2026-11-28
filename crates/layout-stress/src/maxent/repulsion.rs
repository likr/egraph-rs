use egraph_core::{
    CoreError, CoreResult, EuclideanLayout, EuclideanStorageRef, Layout2d, Layout3d,
    RepulsionBackend, Scalar,
};
use egraph_spatial_trees::{ArenaOctree3d, ArenaQuadtree2d};

#[derive(Debug, Clone)]
pub(super) struct Quadtree2dMaxentBackend {
    quadtree: ArenaQuadtree2d<f64>,
    theta: f64,
    q: f64,
    eps_sq: f64,
    x_cur: Vec<f64>,
    y_cur: Vec<f64>,
}

impl Quadtree2dMaxentBackend {
    pub(super) fn new(num_nodes: usize, theta: f64, q: f64, eps_sq: f64) -> Self {
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

impl<S: Scalar> RepulsionBackend<Layout2d<S>> for Quadtree2dMaxentBackend {
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
                "Quadtree2dMaxentBackend::accumulate_repulsion",
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

#[derive(Debug, Clone)]
pub(super) struct Octree3dMaxentBackend {
    octree: ArenaOctree3d<f64>,
    theta: f64,
    q: f64,
    eps_sq: f64,
    x_cur: Vec<f64>,
    y_cur: Vec<f64>,
    z_cur: Vec<f64>,
}

impl Octree3dMaxentBackend {
    pub(super) fn new(num_nodes: usize, theta: f64, q: f64, eps_sq: f64) -> Self {
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

impl<S: Scalar> RepulsionBackend<Layout3d<S>> for Octree3dMaxentBackend {
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
                "Octree3dMaxentBackend::accumulate_repulsion",
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

#[derive(Debug, Clone)]
pub(super) struct ExactMaxentBackend {
    q: f64,
    eps_sq: f64,
    cur_flat: Vec<f64>,
}

impl ExactMaxentBackend {
    pub(super) fn new(q: f64, eps_sq: f64) -> Self {
        Self {
            q,
            eps_sq,
            cur_flat: Vec::new(),
        }
    }
}

impl<L: EuclideanLayout> RepulsionBackend<L> for ExactMaxentBackend {
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
                "ExactMaxentBackend::accumulate_repulsion",
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
