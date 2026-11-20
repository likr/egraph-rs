use crate::kernel::pivoted_diffusion_kernel_builder::resolve_lambda_max;
use crate::*;
use ndarray::{Array2, ScalarOperand};
use num_traits::Float;
use rand::Rng;

/// Constructs the 3-term DC-retained Difference-of-Gaussians (DoG) bandpass specification:
/// `[(1 - beta_floor)/(1 - alpha), t1], [-alpha*(1 - beta_floor)/(1 - alpha), t2], [beta_floor, 0]`.
pub fn dog_bandpass_terms<S: Float>(
    t1: S,
    t2: S,
    alpha: S,
    beta_floor: S,
) -> Result<Vec<(S, S)>, String> {
    if !t1.is_finite() || !t2.is_finite() || !alpha.is_finite() || !beta_floor.is_finite() {
        return Err("DoG parameters must be finite".to_string());
    }
    let denom = S::one() - alpha;
    if denom.abs() <= S::from(1e-12).unwrap() {
        return Err("alpha must not equal 1.0 in DC-retained DoG kernel".to_string());
    }
    let c1 = (S::one() - beta_floor) / denom;
    let c2 = -alpha * c1;
    Ok(vec![(c1, t1), (c2, t2), (beta_floor, S::zero())])
}

pub(crate) fn validate_terms<S: Float>(terms: &[(S, S)]) -> Result<(), String> {
    for &(c, t) in terms {
        if !c.is_finite() || !t.is_finite() {
            return Err("BandPass terms must be finite".to_string());
        }
    }
    Ok(())
}

/// Builder for [`BandPassDiffusionKernel`].
pub struct BandPassDiffusionKernelBuilder<'a, S> {
    pub laplacian: &'a SparseSymmetricMatrix<S>,
    pub terms: Vec<(S, S)>,
    pub degree: usize,
    pub lambda_max: Option<S>,
    error: Option<String>,
}

impl<'a, S> BandPassDiffusionKernelBuilder<'a, S>
where
    S: Float
        + std::iter::Sum
        + std::ops::AddAssign
        + Default
        + ScalarOperand
        + num_traits::FromPrimitive,
{
    pub fn new(laplacian: &'a SparseSymmetricMatrix<S>, terms: Vec<(S, S)>, degree: usize) -> Self {
        Self {
            laplacian,
            terms,
            degree,
            lambda_max: None,
            error: None,
        }
    }

    /// Convenience constructor for a DC-retained 3-term Difference-of-Gaussians kernel.
    pub fn dog(
        laplacian: &'a SparseSymmetricMatrix<S>,
        t1: S,
        t2: S,
        alpha: S,
        beta_floor: S,
        degree: usize,
    ) -> Self {
        match dog_bandpass_terms(t1, t2, alpha, beta_floor) {
            Ok(terms) => Self::new(laplacian, terms, degree),
            Err(err) => Self {
                laplacian,
                terms: Vec::new(),
                degree,
                lambda_max: None,
                error: Some(err),
            },
        }
    }

    pub fn lambda_max(mut self, lambda_max: S) -> Self {
        self.lambda_max = Some(lambda_max);
        self
    }

    pub fn build<R: Rng>(self, rng: &mut R) -> Result<BandPassDiffusionKernel<S>, String> {
        if let Some(err) = self.error {
            return Err(err);
        }
        validate_terms(&self.terms)?;

        let n = self.laplacian.dim();
        if n == 0 {
            return Ok(BandPassDiffusionKernel {
                n: 0,
                matrix: Array2::zeros((0, 0)),
            });
        }

        let lambda_max = resolve_lambda_max(self.laplacian, self.lambda_max, rng);

        let mut eye = Array2::zeros((n, n));
        for i in 0..n {
            eye[[i, i]] = S::one();
        }

        let matrix = chebyshev_bandpass_approximation(
            self.laplacian,
            &self.terms,
            self.degree,
            lambda_max,
            &eye,
        );

        Ok(BandPassDiffusionKernel { n, matrix })
    }
}
