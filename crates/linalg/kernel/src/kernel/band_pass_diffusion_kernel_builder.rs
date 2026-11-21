use crate::kernel::pivoted_diffusion_kernel_builder::resolve_lambda_max;
use crate::*;
use ndarray::{Array2, ScalarOperand};
use num_traits::Float;
use rand::Rng;

/// Constructs the 2-term Difference-of-Gaussians (DoG) bandpass specification:
/// `[(1 - beta_floor, t1), (-alpha * (1 - beta_floor), t2)]`.
///
/// The positive floor `beta_floor` is applied separately as a uniform rank-1 shift
/// (`+ beta_floor * 1 1^T`) via `off_diagonal_floor` rather than as `(beta_floor, 0.0)`,
/// because `exp(-0 * L) = I` only spikes the diagonal without lifting off-diagonal
/// similarities, and omitting `1 / (1 - alpha)` avoids the singularity at `alpha = 1.0`.
pub fn dog_bandpass_terms<S: Float>(
    t1: S,
    t2: S,
    alpha: S,
    beta_floor: S,
) -> Result<Vec<(S, S)>, String> {
    if !t1.is_finite() || !t2.is_finite() || !alpha.is_finite() || !beta_floor.is_finite() {
        return Err("DoG parameters must be finite".to_string());
    }
    if t1 <= S::zero() || t2 <= S::zero() {
        return Err("DoG diffusion times t1 and t2 must be > 0".to_string());
    }
    if alpha < S::zero() {
        return Err("DoG attenuation alpha must be >= 0".to_string());
    }
    if beta_floor < S::zero() || beta_floor >= S::one() {
        return Err("DoG beta_floor must satisfy 0.0 <= beta_floor < 1.0".to_string());
    }
    let scale = S::one() - beta_floor;
    Ok(vec![(scale, t1), (-alpha * scale, t2)])
}

/// Validates and returns positive-coefficient multi-scale heat kernel `(weight, t)` terms.
///
/// Enforces strictly positive finite weights and diffusion times (`w_m > 0`, `t_m > 0`)
/// so the resulting kernel remains strictly positive across connected components without
/// zero-time identity spikes.
pub fn multiscale_heat_terms<S: Float>(terms: &[(S, S)]) -> Result<Vec<(S, S)>, String> {
    if terms.is_empty() {
        return Err("Multiscale heat terms must not be empty".to_string());
    }
    for &(w, t) in terms {
        if !w.is_finite() || !t.is_finite() {
            return Err("Multiscale heat terms must be finite".to_string());
        }
        if w <= S::zero() {
            return Err("Multiscale heat term weights must be > 0".to_string());
        }
        if t <= S::zero() {
            return Err("Multiscale heat diffusion times must be > 0".to_string());
        }
    }
    Ok(terms.to_vec())
}

pub(crate) fn validate_terms<S: Float>(terms: &[(S, S)]) -> Result<(), String> {
    for &(c, t) in terms {
        if !c.is_finite() || !t.is_finite() {
            return Err("BandPass terms must be finite".to_string());
        }
        if t < S::zero() {
            return Err("BandPass diffusion times must be >= 0".to_string());
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
    /// Uniform rank-1 positive shift `+ off_diagonal_floor * 1 1^T` applied to all entries.
    pub off_diagonal_floor: S,
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
            off_diagonal_floor: S::zero(),
            error: None,
        }
    }

    /// Convenience constructor for a Difference-of-Gaussians kernel with uniform `beta_floor` shift.
    pub fn dog(
        laplacian: &'a SparseSymmetricMatrix<S>,
        t1: S,
        t2: S,
        alpha: S,
        beta_floor: S,
        degree: usize,
    ) -> Self {
        match dog_bandpass_terms(t1, t2, alpha, beta_floor) {
            Ok(terms) => Self::new(laplacian, terms, degree).off_diagonal_floor(beta_floor),
            Err(err) => Self {
                laplacian,
                terms: Vec::new(),
                degree,
                lambda_max: None,
                off_diagonal_floor: S::zero(),
                error: Some(err),
            },
        }
    }

    /// Convenience constructor for a positive-coefficient multi-scale heat kernel.
    pub fn multiscale_heat(
        laplacian: &'a SparseSymmetricMatrix<S>,
        terms: Vec<(S, S)>,
        degree: usize,
    ) -> Self {
        match multiscale_heat_terms(&terms) {
            Ok(validated) => Self::new(laplacian, validated, degree),
            Err(err) => Self {
                laplacian,
                terms: Vec::new(),
                degree,
                lambda_max: None,
                off_diagonal_floor: S::zero(),
                error: Some(err),
            },
        }
    }

    pub fn lambda_max(mut self, lambda_max: S) -> Self {
        self.lambda_max = Some(lambda_max);
        self
    }

    /// Sets the uniform rank-1 floor added to every kernel entry (`K_eff = K + floor * 1 1^T`).
    pub fn off_diagonal_floor(mut self, floor: S) -> Self {
        self.off_diagonal_floor = floor;
        self
    }

    pub fn build<R: Rng>(self, rng: &mut R) -> Result<BandPassDiffusionKernel<S>, String> {
        if let Some(err) = self.error {
            return Err(err);
        }
        validate_terms(&self.terms)?;
        if !self.off_diagonal_floor.is_finite() || self.off_diagonal_floor < S::zero() {
            return Err("off_diagonal_floor must be finite and >= 0".to_string());
        }

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

        let mut matrix = chebyshev_bandpass_approximation(
            self.laplacian,
            &self.terms,
            self.degree,
            lambda_max,
            &eye,
        );
        if self.off_diagonal_floor > S::zero() {
            matrix.mapv_inplace(|x| x + self.off_diagonal_floor);
        }

        Ok(BandPassDiffusionKernel { n, matrix })
    }
}
