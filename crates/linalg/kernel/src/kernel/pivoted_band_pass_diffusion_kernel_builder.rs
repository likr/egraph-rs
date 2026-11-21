use crate::kernel::band_pass_diffusion_kernel_builder::{
    dog_bandpass_terms, multiscale_heat_terms, validate_terms,
};
use crate::kernel::pivoted_diffusion_kernel_builder::{
    build_pivot_vectors_with, compute_hutchinson_diagonal_with, resolve_lambda_max,
};
use crate::*;
use ndarray::ScalarOperand;
use num_traits::Float;
use rand::Rng;

/// Builder for [`PivotedBandPassDiffusionKernel`].
pub struct PivotedBandPassDiffusionKernelBuilder<'a, S> {
    pub laplacian: &'a SparseSymmetricMatrix<S>,
    pub terms: Vec<(S, S)>,
    pub degree: usize,
    pub pivots: Vec<usize>,
    pub lambda_max: Option<S>,
    pub num_hutchinson: Option<usize>,
    /// Uniform rank-1 positive shift `+ off_diagonal_floor * 1 1^T` applied to all entries.
    pub off_diagonal_floor: S,
    error: Option<String>,
}

impl<'a, S> PivotedBandPassDiffusionKernelBuilder<'a, S>
where
    S: Float
        + std::iter::Sum
        + std::ops::AddAssign
        + Default
        + ScalarOperand
        + num_traits::FromPrimitive,
{
    pub fn new(
        laplacian: &'a SparseSymmetricMatrix<S>,
        terms: Vec<(S, S)>,
        degree: usize,
        pivots: Vec<usize>,
    ) -> Self {
        Self {
            laplacian,
            terms,
            degree,
            pivots,
            lambda_max: None,
            num_hutchinson: Some(32),
            off_diagonal_floor: S::zero(),
            error: None,
        }
    }

    /// Convenience constructor for a Difference-of-Gaussians pivoted kernel with uniform `beta_floor` shift.
    pub fn dog(
        laplacian: &'a SparseSymmetricMatrix<S>,
        t1: S,
        t2: S,
        alpha: S,
        beta_floor: S,
        degree: usize,
        pivots: Vec<usize>,
    ) -> Self {
        match dog_bandpass_terms(t1, t2, alpha, beta_floor) {
            Ok(terms) => Self::new(laplacian, terms, degree, pivots).off_diagonal_floor(beta_floor),
            Err(err) => Self {
                laplacian,
                terms: Vec::new(),
                degree,
                pivots,
                lambda_max: None,
                num_hutchinson: Some(32),
                off_diagonal_floor: S::zero(),
                error: Some(err),
            },
        }
    }

    /// Convenience constructor for a positive-coefficient multi-scale heat pivoted kernel.
    pub fn multiscale_heat(
        laplacian: &'a SparseSymmetricMatrix<S>,
        terms: Vec<(S, S)>,
        degree: usize,
        pivots: Vec<usize>,
    ) -> Self {
        match multiscale_heat_terms(&terms) {
            Ok(validated) => Self::new(laplacian, validated, degree, pivots),
            Err(err) => Self {
                laplacian,
                terms: Vec::new(),
                degree,
                pivots,
                lambda_max: None,
                num_hutchinson: Some(32),
                off_diagonal_floor: S::zero(),
                error: Some(err),
            },
        }
    }

    pub fn lambda_max(mut self, lambda_max: S) -> Self {
        self.lambda_max = Some(lambda_max);
        self
    }

    pub fn num_hutchinson(mut self, num_hutchinson: Option<usize>) -> Self {
        self.num_hutchinson = num_hutchinson;
        self
    }

    /// Sets the uniform rank-1 floor added to every kernel entry (`K_eff = K + floor * 1 1^T`).
    pub fn off_diagonal_floor(mut self, floor: S) -> Self {
        self.off_diagonal_floor = floor;
        self
    }

    pub fn build<R: Rng>(self, rng: &mut R) -> Result<PivotedBandPassDiffusionKernel<S>, String> {
        if let Some(err) = self.error {
            return Err(err);
        }
        validate_terms(&self.terms)?;
        if !self.off_diagonal_floor.is_finite() || self.off_diagonal_floor < S::zero() {
            return Err("off_diagonal_floor must be finite and >= 0".to_string());
        }

        let n = self.laplacian.dim();
        for &p in &self.pivots {
            if p >= n {
                return Err("Pivot index out of bounds".to_string());
            }
        }

        let lambda_max = resolve_lambda_max(self.laplacian, self.lambda_max, rng);

        let mut pivot_vectors = build_pivot_vectors_with(n, &self.pivots, |e| {
            chebyshev_bandpass_approximation_vec(
                self.laplacian,
                &self.terms,
                self.degree,
                lambda_max,
                e,
            )
        })?;

        let g_hat_0 = self
            .terms
            .iter()
            .map(|&(c, _)| c)
            .fold(S::zero(), |acc, c| acc + c);

        let mut diagonal = compute_hutchinson_diagonal_with(
            self.laplacian,
            self.num_hutchinson,
            g_hat_0,
            rng,
            |rademacher, u1| {
                chebyshev_bandpass_approximation_with_baseline(
                    self.laplacian,
                    &self.terms,
                    self.degree,
                    lambda_max,
                    rademacher,
                    u1,
                )
            },
        );

        // Apply uniform rank-1 floor after Chebyshev evaluation so both pivot columns and
        // diagonal estimates match `K_eff = K_bp + floor * 1 1^T` without a t=0 identity spike.
        if self.off_diagonal_floor > S::zero() {
            let floor = self.off_diagonal_floor;
            for vec in &mut pivot_vectors {
                vec.mapv_inplace(|x| x + floor);
            }
            if let Some(ref mut diag) = diagonal {
                diag.mapv_inplace(|x| x + floor);
            }
        }

        Ok(PivotedBandPassDiffusionKernel {
            pivots: self.pivots,
            vectors: pivot_vectors,
            diagonal,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::UnGraph;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn sample_graph_laplacian() -> SparseSymmetricMatrix<f64> {
        let mut g = UnGraph::<(), ()>::new_undirected();
        let nodes: Vec<_> = (0..6).map(|_| g.add_node(())).collect();
        for &(u, v) in &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)] {
            g.add_edge(nodes[u], nodes[v], ());
        }
        SparseSymmetricMatrix::symmetric_normalized_laplacian(&g, |_| 1.0)
    }

    #[test]
    fn test_bandpass_and_pivoted_consistency_and_hutchinson() {
        let lap = sample_graph_laplacian();
        let beta = 0.1;
        let alpha = 0.75;
        let mut rng1 = StdRng::seed_from_u64(42);
        let full = BandPassDiffusionKernelBuilder::dog(&lap, 0.4, 2.0, alpha, beta, 25)
            .lambda_max(2.0)
            .build(&mut rng1)
            .unwrap();

        let mut rng2 = StdRng::seed_from_u64(42);
        let pivoted =
            PivotedBandPassDiffusionKernelBuilder::dog(&lap, 0.4, 2.0, alpha, beta, 25, vec![0, 3])
                .lambda_max(2.0)
                .num_hutchinson(Some(512))
                .build(&mut rng2)
                .unwrap();

        for (p_idx, &p) in pivoted.pivots().iter().enumerate() {
            for j in 0..6 {
                assert!((pivoted.get_from_pivot(p_idx, j) - full.get_by_index(p, j)).abs() < 1e-8);
            }
        }

        // Verify off_diagonal_floor lifts off-diagonal entries uniformly relative to beta=0
        let mut rng3 = StdRng::seed_from_u64(42);
        let unshifted = BandPassDiffusionKernelBuilder::dog(&lap, 0.4, 2.0, alpha, 0.0, 25)
            .lambda_max(2.0)
            .build(&mut rng3)
            .unwrap();
        for i in 0..6 {
            for j in 0..6 {
                let expected = (1.0 - beta) * unshifted.get_by_index(i, j) + beta;
                assert!((full.get_by_index(i, j) - expected).abs() < 1e-10);
                assert!(full.get_by_index(i, j) > 0.0);
            }
        }

        // Hutchinson diagonal should closely approximate exact diagonal
        let mut mean_err = 0.0;
        for j in 0..6 {
            let est = pivoted.diagonal(j).expect("diagonal should be Some");
            let exact = full.get_by_index(j, j);
            mean_err += (est - exact).abs();
        }
        mean_err /= 6.0;
        assert!(
            mean_err < 0.08,
            "Hutchinson mean error too large: {mean_err}"
        );

        // Stationary vector action under K_eff = (1 - beta) * K_dog + beta * 1 1^T
        let u1 = lap.stationary_vector();
        let u1_sum = u1.sum();
        let ku1 = full.matrix.dot(&u1);
        for j in 0..6 {
            let expected = (1.0 - beta) * (1.0 - alpha) * u1[j] + beta * u1_sum;
            assert!((ku1[j] - expected).abs() < 1e-8);
        }
    }

    #[test]
    fn test_pivoted_diffusion_hutchinson_option() {
        let lap = sample_graph_laplacian();
        let mut rng = StdRng::seed_from_u64(7);
        let full = DiffusionKernelBuilder::new(&lap, 0.8, 25)
            .lambda_max(2.0)
            .build(&mut rng)
            .unwrap();

        let pk_none = PivotedDiffusionKernelBuilder::new(&lap, 0.8, 25, vec![0, 2])
            .lambda_max(2.0)
            .build(&mut rng)
            .unwrap();
        assert!(pk_none.diagonal(0).is_none());

        let pk_hutch = PivotedDiffusionKernelBuilder::new(&lap, 0.8, 25, vec![0, 2])
            .lambda_max(2.0)
            .num_hutchinson(Some(512))
            .build(&mut rng)
            .unwrap();
        for j in 0..6 {
            let est = pk_hutch.diagonal(j).unwrap();
            let exact = full.get_by_index(j, j);
            assert!(
                (est - exact).abs() < 0.15,
                "node {j}: est={est}, exact={exact}"
            );
        }
    }

    #[test]
    fn test_dog_supports_alpha_one_and_validates_params() {
        let lap = sample_graph_laplacian();
        let mut rng = StdRng::seed_from_u64(1);

        let terms = dog_bandpass_terms(0.5, 2.0, 1.0, 0.15).unwrap();
        assert_eq!(terms.len(), 2);
        assert!(terms.iter().all(|&(_, t)| t > 0.0));
        assert!((terms[0].0 - 0.85).abs() < 1e-12);
        assert!((terms[1].0 + 0.85).abs() < 1e-12);

        for &beta in &[0.0, 0.15] {
            let full = BandPassDiffusionKernelBuilder::dog(&lap, 0.5, 2.0, 1.0, beta, 25)
                .lambda_max(2.0)
                .build(&mut rng)
                .unwrap();
            let pivoted = PivotedBandPassDiffusionKernelBuilder::dog(
                &lap,
                0.5,
                2.0,
                1.0,
                beta,
                25,
                vec![0, 2],
            )
            .lambda_max(2.0)
            .build(&mut rng)
            .unwrap();
            for (p_idx, &p) in pivoted.pivots().iter().enumerate() {
                for j in 0..6 {
                    assert!(
                        (pivoted.get_from_pivot(p_idx, j) - full.get_by_index(p, j)).abs() < 1e-8
                    );
                }
            }
        }

        // Invalid parameters should be rejected cleanly
        assert!(dog_bandpass_terms(0.0, 2.0, 1.0, 0.1).is_err());
        assert!(dog_bandpass_terms(0.5, -1.0, 1.0, 0.1).is_err());
        assert!(dog_bandpass_terms(0.5, 2.0, -0.1, 0.1).is_err());
        assert!(dog_bandpass_terms(0.5, 2.0, 1.0, -0.01).is_err());
        assert!(dog_bandpass_terms(0.5, 2.0, 1.0, 1.0).is_err());
        assert!(dog_bandpass_terms(f64::NAN, 2.0, 1.0, 0.1).is_err());
    }

    #[test]
    fn test_multiscale_heat_2_and_3_scales() {
        let lap = sample_graph_laplacian();
        let mut rng = StdRng::seed_from_u64(13);

        for terms in [
            vec![(0.6, 0.5), (0.4, 2.0)],
            vec![(0.5, 0.5), (0.3, 2.0), (0.2, 8.0)],
        ] {
            let full = BandPassDiffusionKernelBuilder::multiscale_heat(&lap, terms.clone(), 25)
                .lambda_max(2.0)
                .build(&mut rng)
                .unwrap();
            let pivoted = PivotedBandPassDiffusionKernelBuilder::multiscale_heat(
                &lap,
                terms.clone(),
                25,
                vec![0, 3],
            )
            .lambda_max(2.0)
            .build(&mut rng)
            .unwrap();

            for (p_idx, &p) in pivoted.pivots().iter().enumerate() {
                for j in 0..6 {
                    let val = full.get_by_index(p, j);
                    assert!(val > 0.0, "multiscale heat entry ({p}, {j}) must be > 0");
                    assert!((pivoted.get_from_pivot(p_idx, j) - val).abs() < 1e-8);
                }
            }
        }

        assert!(
            BandPassDiffusionKernelBuilder::multiscale_heat(&lap, vec![], 20)
                .build(&mut rng)
                .is_err()
        );
        assert!(
            BandPassDiffusionKernelBuilder::multiscale_heat(&lap, vec![(-0.1, 0.5)], 20)
                .build(&mut rng)
                .is_err()
        );
        assert!(PivotedBandPassDiffusionKernelBuilder::multiscale_heat(
            &lap,
            vec![(0.5, 0.0)],
            20,
            vec![0]
        )
        .build(&mut rng)
        .is_err());
    }
}
