use egraph_core::{
    CoreError, CoreResult, EuclideanLayout, EuclideanStorageRef, RepulsionBackend, Scalar,
};
use egraph_spatial_trees::{ArenaOctree3d, ArenaQuadtree2d};

/// Weighted Coulomb repulsion backend routing via [`EuclideanStorageRef`]:
/// - `Soa2d` (`Layout2d`) -> [`ArenaQuadtree2d`] or exact 2D
/// - `Soa3d` (`Layout3d`) -> [`ArenaOctree3d`] or exact 3D
/// - `FlatNd` (`LayoutNd`) -> exact N-D only (Barnes-Hut fails fast with [`CoreError::UnsupportedOperation`])
#[derive(Debug, Clone)]
pub struct CoulombRepulsionBackend<S: Scalar> {
    use_barnes_hut: bool,
    theta: f64,
    eps_sq: f64,
    masses: Vec<f64>,
    quadtree: ArenaQuadtree2d<S>,
    octree: ArenaOctree3d<S>,
}

impl<S: Scalar> CoulombRepulsionBackend<S> {
    /// Creates a weighted Coulomb repulsion backend for `masses`.
    pub fn new(
        use_barnes_hut: bool,
        theta: f64,
        eps_sq: f64,
        masses: Vec<f64>,
    ) -> CoreResult<Self> {
        if !theta.is_finite() || theta < 0.0 {
            return Err(CoreError::invalid_input("theta must be finite and >= 0.0"));
        }
        if !eps_sq.is_finite() || eps_sq <= 0.0 {
            return Err(CoreError::invalid_input("eps_sq must be finite and > 0.0"));
        }
        Ok(Self {
            use_barnes_hut,
            theta,
            eps_sq,
            masses,
            quadtree: ArenaQuadtree2d::new(),
            octree: ArenaOctree3d::new(),
        })
    }

    /// Updates the softening squared-distance parameter (`eps_sq > 0`).
    pub fn set_eps_sq(&mut self, eps_sq: f64) -> CoreResult<()> {
        if !eps_sq.is_finite() || eps_sq <= 0.0 {
            return Err(CoreError::invalid_input("eps_sq must be finite and > 0.0"));
        }
        self.eps_sq = eps_sq;
        Ok(())
    }

    /// Returns the per-node mass weights configured on this backend.
    pub fn masses(&self) -> &[f64] {
        &self.masses
    }
}

impl<S: Scalar, L: EuclideanLayout<Scalar = S> + ?Sized> RepulsionBackend<L>
    for CoulombRepulsionBackend<S>
{
    fn accumulate_repulsion(&mut self, layout: &L, forces: &mut [f64]) -> CoreResult<Option<f64>> {
        let n = layout.len();
        let dims = layout.dims();
        let expected = n
            .checked_mul(dims)
            .ok_or_else(|| CoreError::invalid_input("repulsion force buffer size overflow"))?;
        if forces.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                forces.len(),
                "CoulombRepulsionBackend::accumulate_repulsion",
            ));
        }
        if self.masses.len() != n {
            return Err(CoreError::dimension_mismatch(
                self.masses.len(),
                n,
                "CoulombRepulsionBackend::accumulate_repulsion::masses",
            ));
        }
        if n < 2 {
            return Ok(None);
        }

        let theta_sq = self.theta * self.theta;
        let eps_sq = self.eps_sq;

        match layout.as_euclidean_storage() {
            EuclideanStorageRef::Soa2d { x, y } => {
                if self.use_barnes_hut {
                    self.quadtree.build_weighted_xy(x, y, &self.masses)?;
                    for i in 0..n {
                        let (fx, fy) = self.quadtree.eval_point_coulomb_raw(i, theta_sq, eps_sq)?;
                        let mi = self.masses[i];
                        forces[2 * i] += mi * fx;
                        forces[2 * i + 1] += mi * fy;
                    }
                } else {
                    for i in 0..n {
                        let xi = x[i].to_f64();
                        let yi = y[i].to_f64();
                        let mi = self.masses[i];
                        let mut acc_x = 0.0_f64;
                        let mut acc_y = 0.0_f64;
                        for j in (i + 1)..n {
                            let dx = xi - x[j].to_f64();
                            let dy = yi - y[j].to_f64();
                            let mult = (mi * self.masses[j]) / (dx * dx + dy * dy + eps_sq);
                            let fx = mult * dx;
                            let fy = mult * dy;
                            acc_x += fx;
                            acc_y += fy;
                            forces[2 * j] -= fx;
                            forces[2 * j + 1] -= fy;
                        }
                        forces[2 * i] += acc_x;
                        forces[2 * i + 1] += acc_y;
                    }
                }
                Ok(None)
            }
            EuclideanStorageRef::Soa3d { x, y, z } => {
                if self.use_barnes_hut {
                    self.octree.build_weighted_xyz(x, y, z, &self.masses)?;
                    for i in 0..n {
                        let (fx, fy, fz) =
                            self.octree.eval_point_coulomb_raw(i, theta_sq, eps_sq)?;
                        let mi = self.masses[i];
                        forces[3 * i] += mi * fx;
                        forces[3 * i + 1] += mi * fy;
                        forces[3 * i + 2] += mi * fz;
                    }
                } else {
                    for i in 0..n {
                        let xi = x[i].to_f64();
                        let yi = y[i].to_f64();
                        let zi = z[i].to_f64();
                        let mi = self.masses[i];
                        let mut acc_x = 0.0_f64;
                        let mut acc_y = 0.0_f64;
                        let mut acc_z = 0.0_f64;
                        for j in (i + 1)..n {
                            let dx = xi - x[j].to_f64();
                            let dy = yi - y[j].to_f64();
                            let dz = zi - z[j].to_f64();
                            let mult =
                                (mi * self.masses[j]) / (dx * dx + dy * dy + dz * dz + eps_sq);
                            let fx = mult * dx;
                            let fy = mult * dy;
                            let fz = mult * dz;
                            acc_x += fx;
                            acc_y += fy;
                            acc_z += fz;
                            forces[3 * j] -= fx;
                            forces[3 * j + 1] -= fy;
                            forces[3 * j + 2] -= fz;
                        }
                        forces[3 * i] += acc_x;
                        forces[3 * i + 1] += acc_y;
                        forces[3 * i + 2] += acc_z;
                    }
                }
                Ok(None)
            }
            EuclideanStorageRef::FlatNd {
                coords,
                num_nodes,
                dims,
            } => {
                if self.use_barnes_hut {
                    return Err(CoreError::unsupported(
                        "Barnes-Hut repulsion is not supported on LayoutNd; use Layout3d for 3D Barnes-Hut, Layout2d for 2D, or Exact for arbitrary N-D",
                    ));
                }
                for i in 0..num_nodes {
                    let base_i = i * dims;
                    let mi = self.masses[i];
                    for (j, &mj) in self.masses.iter().enumerate().take(num_nodes).skip(i + 1) {
                        let base_j = j * dims;
                        let mut dist_sq = 0.0_f64;
                        for axis in 0..dims {
                            let delta =
                                coords[base_i + axis].to_f64() - coords[base_j + axis].to_f64();
                            dist_sq += delta * delta;
                        }
                        let mult = (mi * mj) / (dist_sq + eps_sq);
                        for axis in 0..dims {
                            let f = mult
                                * (coords[base_i + axis].to_f64() - coords[base_j + axis].to_f64());
                            forces[base_i + axis] += f;
                            forces[base_j + axis] -= f;
                        }
                    }
                }
                Ok(None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use egraph_core::{CoreError, Layout2d, Layout3d, LayoutNd, RepulsionBackend};

    use super::CoulombRepulsionBackend;

    #[test]
    fn test_coulomb_repulsion_backend_routes_2d_3d_and_rejects_nd_barnes_hut() {
        let l2 = Layout2d::from_xy(vec![0.0, 1.5, -0.5], vec![0.0, 0.5, 1.0]).unwrap();
        let l3 = Layout3d::from_xyz(
            vec![0.0, 1.5, -0.5],
            vec![0.0, 0.5, 1.0],
            vec![0.0, 0.2, -0.8],
        )
        .unwrap();
        let lnd = LayoutNd::from_flat(3, 4, &[0.0; 12]).unwrap();
        let masses = vec![1.0, 2.0, 1.5];

        let mut bh_2d =
            CoulombRepulsionBackend::<f64>::new(true, 0.0, 1e-6, masses.clone()).unwrap();
        let mut ex_2d =
            CoulombRepulsionBackend::<f64>::new(false, 0.0, 1e-6, masses.clone()).unwrap();
        let mut f_bh2 = vec![0.0; 6];
        let mut f_ex2 = vec![0.0; 6];
        bh_2d.accumulate_repulsion(&l2, &mut f_bh2).unwrap();
        ex_2d.accumulate_repulsion(&l2, &mut f_ex2).unwrap();
        for (a, b) in f_bh2.iter().zip(f_ex2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }

        let mut bh_3d =
            CoulombRepulsionBackend::<f64>::new(true, 0.0, 1e-6, masses.clone()).unwrap();
        let mut ex_3d =
            CoulombRepulsionBackend::<f64>::new(false, 0.0, 1e-6, masses.clone()).unwrap();
        let mut f_bh3 = vec![0.0; 9];
        let mut f_ex3 = vec![0.0; 9];
        bh_3d.accumulate_repulsion(&l3, &mut f_bh3).unwrap();
        ex_3d.accumulate_repulsion(&l3, &mut f_ex3).unwrap();
        for (a, b) in f_bh3.iter().zip(f_ex3.iter()) {
            assert!((a - b).abs() < 1e-10);
        }

        let mut bh_nd =
            CoulombRepulsionBackend::<f64>::new(true, 0.5, 1e-6, masses.clone()).unwrap();
        let mut f_nd = vec![0.0; lnd.len() * lnd.dims()];
        assert!(matches!(
            bh_nd.accumulate_repulsion(&lnd, &mut f_nd),
            Err(CoreError::UnsupportedOperation(_))
        ));

        let mut ex_nd = CoulombRepulsionBackend::<f64>::new(false, 0.5, 1e-6, masses).unwrap();
        ex_nd.accumulate_repulsion(&lnd, &mut f_nd).unwrap();
        assert!(f_nd.iter().all(|v| v.is_finite()));
    }
}
