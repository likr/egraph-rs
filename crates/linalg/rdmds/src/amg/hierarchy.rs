use super::matrix::CsrMatrix;
use super::aggregation::{strength_of_connection, standard_aggregation, tentative_prolongator};
use super::prolongator::{estimate_spectral_radius, smoothed_prolongator};

#[derive(Debug, Clone)]
pub struct Level<S> {
    pub a: CsrMatrix<S>,
    pub p: CsrMatrix<S>,
    pub r: CsrMatrix<S>,
}

#[derive(Debug, Clone)]
pub struct Hierarchy<S> {
    pub levels: Vec<Level<S>>,
    pub coarsest_a: CsrMatrix<S>,
}

/// Builds the AMG hierarchy (Smoothed Aggregation)
pub fn build_hierarchy<S>(
    a_fine: CsrMatrix<S>,
    theta: S,
    max_levels: usize,
    max_coarse_size: usize,
) -> Hierarchy<S>
where
    S: Copy + num_traits::Float + num_traits::Zero + std::ops::AddAssign + std::ops::SubAssign + Default + std::cmp::PartialOrd + std::fmt::Debug,
{
    let mut levels = Vec::new();
    let mut current_a = a_fine;

    for _ in 0..max_levels {
        if current_a.rows <= max_coarse_size {
            break;
        }

        let strength = strength_of_connection(&current_a, theta);
        let (aggregates, num_aggregates) = standard_aggregation(&strength);

        if num_aggregates == 0 || num_aggregates == current_a.rows {
            // Cannot coarsen further
            break;
        }

        let p0 = tentative_prolongator(&aggregates, num_aggregates);
        let rho = estimate_spectral_radius(&current_a);

        let omega = if rho > S::zero() {
            let four = S::one() + S::one() + S::one() + S::one();
            let three = S::one() + S::one() + S::one();
            (four / three) / rho
        } else {
            S::one()
        };

        let p = smoothed_prolongator(&current_a, &p0, omega);
        let r = p.transpose();

        let a_coarse = r.mul_csr(&current_a).mul_csr(&p);

        let level = Level {
            a: current_a.clone(), // or we can avoid clone by taking ownership, but it's fine for now
            p,
            r,
        };

        levels.push(level);
        current_a = a_coarse;
    }

    Hierarchy {
        levels,
        coarsest_a: current_a,
    }
}
