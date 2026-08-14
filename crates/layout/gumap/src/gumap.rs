use crate::{fuzzy_simplicial_set::compute_fuzzy_simplicial_set, gumap_builder::GuMapBuilder};
use num_traits::Float;
use petgraph_drawing::{Drawing, DrawingEuclidean2d, DrawingIndex, DrawingValue};
use petgraph_linalg_kernel::Distance;
use rand::prelude::*;

/// GUMAP (Graph UMAP) layout algorithm.
///
/// Direct application of Uniform Manifold Approximation and Projection (UMAP)
/// to graph layout by constructing fuzzy simplicial sets from graph shortest path distances
/// and optimizing cross-entropy loss via stochastic gradient descent with negative sampling.
#[derive(Debug, Clone)]
pub struct GuMap<S> {
    pub k: usize,
    pub min_dist: S,
    pub spread: S,
    pub n_epochs: usize,
    pub negative_sample_rate: usize,
    pub learning_rate: S,
    pub a: S,
    pub b: S,
    pub sigma_iters: usize,
    pub sigma_tolerance: S,
}

impl<S> GuMap<S>
where
    S: DrawingValue + Float + std::iter::Sum + std::ops::AddAssign + Default,
{
    /// Creates a builder to configure and construct a `GuMap` layout instance.
    pub fn builder() -> GuMapBuilder<S> {
        GuMapBuilder::new()
    }

    /// Creates a `GuMap` instance with default hyperparameters.
    pub fn new() -> Self {
        GuMapBuilder::new().build().unwrap()
    }

    /// Executes the GUMAP layout algorithm on the drawing using the provided distance matrix and RNG.
    #[allow(clippy::needless_range_loop)]
    pub fn run<N, D, R>(
        &self,
        drawing: &mut DrawingEuclidean2d<N, S>,
        distance_matrix: &D,
        rng: &mut R,
    ) where
        N: Copy + Eq + std::hash::Hash + DrawingIndex,
        D: Distance<N, S> + ?Sized,
        R: Rng + ?Sized,
    {
        let n = drawing.len();
        if n < 2 {
            return;
        }

        // Extract or initialize coordinates
        let mut points: Vec<[S; 2]> = Vec::with_capacity(n);
        let mut is_all_zero = true;
        for i in 0..n {
            let entry = drawing.raw_entry(i);
            if entry.0 != S::zero() || entry.1 != S::zero() {
                is_all_zero = false;
            }
            points.push([entry.0, entry.1]);
        }

        if is_all_zero {
            for p in &mut points {
                let rx = S::from_f64(rng.gen_range(-10.0..10.0)).unwrap();
                let ry = S::from_f64(rng.gen_range(-10.0..10.0)).unwrap();
                p[0] = rx;
                p[1] = ry;
            }
        }

        // Step 2: Compute high-dimensional fuzzy simplicial set (symmetrized probabilities)
        let edges = compute_fuzzy_simplicial_set(
            distance_matrix,
            n,
            self.k,
            self.sigma_iters,
            self.sigma_tolerance,
        );

        if edges.is_empty() {
            return;
        }

        let two_s = S::from_f32(2.0).unwrap();
        let four_s = S::from_f32(4.0).unwrap();
        let neg_four_s = -four_s;
        let eps = S::from_f32(1e-4).unwrap();
        let two_ab = two_s * self.a * self.b;
        let two_b = two_s * self.b;
        let two_b_minus_two = two_b - two_s;

        // Step 4: SGD Optimization Loop with Negative Sampling and Linear Learning Rate Decay
        let n_epochs_s = S::from_usize(self.n_epochs).unwrap();

        for epoch in 0..self.n_epochs {
            let epoch_s = S::from_usize(epoch).unwrap();
            let alpha = self.learning_rate * (S::one() - epoch_s / n_epochs_s);

            for edge in &edges {
                // Probabilistic edge sampling based on weight
                let p_edge: f64 = edge.weight.to_f64().unwrap_or(1.0);
                if rng.gen::<f64>() > p_edge {
                    continue;
                }

                let i = edge.source;
                let j = edge.target;

                // 1. Attractive Force
                let dx = points[i][0] - points[j][0];
                let dy = points[i][1] - points[j][1];
                let dist_sq = dx * dx + dy * dy;
                let dist = dist_sq.sqrt().max(eps);

                let d_2b = dist.powf(two_b);
                let d_pow = if two_b_minus_two == S::zero() {
                    S::one()
                } else {
                    dist.powf(two_b_minus_two)
                };

                let grad_coeff = (-two_ab * d_pow / (S::one() + self.a * d_2b))
                    .max(neg_four_s)
                    .min(four_s);

                let delta_x = alpha * grad_coeff * dx;
                let delta_y = alpha * grad_coeff * dy;

                let new_ix = points[i][0] + delta_x;
                let new_iy = points[i][1] + delta_y;
                let new_jx = points[j][0] - delta_x;
                let new_jy = points[j][1] - delta_y;

                if new_ix.is_finite() {
                    points[i][0] = new_ix;
                }
                if new_iy.is_finite() {
                    points[i][1] = new_iy;
                }
                if new_jx.is_finite() {
                    points[j][0] = new_jx;
                }
                if new_jy.is_finite() {
                    points[j][1] = new_jy;
                }

                // 2. Repulsive Forces (Negative Sampling)
                for _ in 0..self.negative_sample_rate {
                    let k = rng.gen_range(0..n);
                    if k == i {
                        continue;
                    }

                    let r_dx = points[i][0] - points[k][0];
                    let r_dy = points[i][1] - points[k][1];
                    let r_dist_sq = r_dx * r_dx + r_dy * r_dy;
                    let r_dist = r_dist_sq.sqrt().max(eps);

                    let r_d_2b = r_dist.powf(two_b);
                    let rep_coeff = (two_b / ((r_dist_sq + eps) * (S::one() + self.a * r_d_2b)))
                        .max(S::zero())
                        .min(four_s);

                    let r_delta_x = alpha * rep_coeff * r_dx;
                    let r_delta_y = alpha * rep_coeff * r_dy;

                    let new_rix = points[i][0] + r_delta_x;
                    let new_riy = points[i][1] + r_delta_y;

                    if new_rix.is_finite() {
                        points[i][0] = new_rix;
                    }
                    if new_riy.is_finite() {
                        points[i][1] = new_riy;
                    }
                }
            }
        }

        // Store back to drawing
        for i in 0..n {
            drawing.raw_entry_mut(i).0 = points[i][0];
            drawing.raw_entry_mut(i).1 = points[i][1];
        }
    }
}

impl<S> Default for GuMap<S>
where
    S: DrawingValue + Float + std::iter::Sum + std::ops::AddAssign + Default,
{
    fn default() -> Self {
        Self::new()
    }
}
