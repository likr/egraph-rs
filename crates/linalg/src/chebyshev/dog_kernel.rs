use egraph_core::{CoreError, CoreResult};

use crate::csr_matrix::{CsrMatrix, SpectralMomentStats};
use crate::operator::LinearOperator;

use super::workspace::ChebyshevWorkspace;

/// Configuration for [`ChebyshevDogKernel`] implementing R1/R2 regularized Difference-of-Gaussians
/// with parameter-free `beta_auto` positivity floor and deterministic JL diagonal estimation.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct ChebyshevDogConfig {
    /// Optional short-range diffusion time $t_1$ (`None` uses R1 $1 / (1 + \sigma_\lambda)$).
    pub t1: Option<f64>,
    /// Optional long-range diffusion time $t_2$ (`None` uses R1 $4 / (1 + \sigma_\lambda)$).
    pub t2: Option<f64>,
    /// Optional DoG attenuation $\alpha$ (`None` uses R1 $\max(0, 1 - \sigma_\lambda)$).
    pub alpha: Option<f64>,
    /// Optional explicit uniform rank-1 floor $\beta \in [0, 1)$ (`None` computes `beta_auto`).
    pub beta_floor: Option<f64>,
    /// Optional explicit Chebyshev polynomial degree (`None` uses [`adaptive_chebyshev_degree`](crate::adaptive_chebyshev_degree)).
    pub degree: Option<usize>,
    /// Number of Core-First MaxMin shortest-path landmark columns to evaluate.
    pub n_landmarks: usize,
    /// Minimum degree threshold for Core-First MaxMin landmark selection.
    pub min_degree: usize,
    /// Johnson-Lindenstrauss projection dimension $d$ for exact PSD diagonal estimation.
    pub jl_dim: usize,
    /// Minimum spectral floor $\epsilon_{\mathrm{psd}}$ enforced during R2 PSD projection.
    pub eps_psd: f64,
    /// Spectral upper bound $\lambda_{\max}$ for $L_{\mathrm{sym}}$ (default `2.0`).
    pub lambda_max: f64,
    /// Explicit deterministic seed controlling landmark selection and JL Rademacher probes.
    pub seed: u64,
}

impl Default for ChebyshevDogConfig {
    fn default() -> Self {
        Self {
            t1: None,
            t2: None,
            alpha: None,
            beta_floor: None,
            degree: None,
            n_landmarks: 64,
            min_degree: 2,
            jl_dim: 64,
            eps_psd: 1e-4,
            lambda_max: 2.0,
            seed: 42,
        }
    }
}

impl ChebyshevDogConfig {
    /// Creates a new [`ChebyshevDogConfig`] initialized to [`Self::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the optional short-range diffusion time `t1`.
    pub fn with_t1(mut self, t1: Option<f64>) -> Self {
        self.t1 = t1;
        self
    }

    /// Sets the optional long-range diffusion time `t2`.
    pub fn with_t2(mut self, t2: Option<f64>) -> Self {
        self.t2 = t2;
        self
    }

    /// Sets the optional DoG attenuation `alpha`.
    pub fn with_alpha(mut self, alpha: Option<f64>) -> Self {
        self.alpha = alpha;
        self
    }

    /// Sets the optional explicit rank-1 positivity floor `beta_floor`.
    pub fn with_beta_floor(mut self, beta_floor: Option<f64>) -> Self {
        self.beta_floor = beta_floor;
        self
    }

    /// Sets the optional explicit Chebyshev polynomial `degree`.
    pub fn with_degree(mut self, degree: Option<usize>) -> Self {
        self.degree = degree;
        self
    }

    /// Sets the number of landmark columns `n_landmarks`.
    pub fn with_n_landmarks(mut self, n_landmarks: usize) -> Self {
        self.n_landmarks = n_landmarks;
        self
    }

    /// Sets the minimum degree threshold `min_degree` for landmark selection.
    pub fn with_min_degree(mut self, min_degree: usize) -> Self {
        self.min_degree = min_degree;
        self
    }

    /// Sets the Johnson-Lindenstrauss projection dimension `jl_dim`.
    pub fn with_jl_dim(mut self, jl_dim: usize) -> Self {
        self.jl_dim = jl_dim;
        self
    }

    /// Sets the minimum PSD spectral floor `eps_psd`.
    pub fn with_eps_psd(mut self, eps_psd: f64) -> Self {
        self.eps_psd = eps_psd;
        self
    }

    /// Sets the spectral upper bound `lambda_max`.
    pub fn with_lambda_max(mut self, lambda_max: f64) -> Self {
        self.lambda_max = lambda_max;
        self
    }

    /// Sets the deterministic PRNG `seed`.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// Pure $O(\mathrm{deg} \cdot (k_{\mathrm{lm}} + d_{\mathrm{JL}}) \cdot |E|)$ regularized DoG
/// diffusion kernel backed by [`ChebyshevWorkspace`] and deterministic JL embeddings.
#[derive(Debug, Clone, PartialEq)]
pub struct ChebyshevDogKernel {
    pub(super) num_nodes: usize,
    pub(super) t1: f64,
    pub(super) t2: f64,
    pub(super) alpha: f64,
    pub(super) degree: usize,
    pub(super) lambda_max: f64,
    pub(super) delta_psd: f64,
    pub(super) h_min: f64,
    pub(super) beta_auto: f64,
    pub(super) beta_eff: f64,
    pub(super) moment_stats: SpectralMomentStats,
    pub(super) terms_psd: Vec<(f64, f64)>,
    pub(super) coeffs_h: Vec<f64>,
    pub(super) landmarks: Vec<u32>,
    pub(super) landmark_kernel: Vec<f64>,
    pub(super) diag_estimate: Vec<f64>,
    pub(super) jl_embedding: Vec<f64>,
    pub(super) jl_dim: usize,
}

impl ChebyshevDogKernel {
    /// Returns the number of vertices $N$.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of evaluated landmark columns $k_{\mathrm{lm}}$.
    #[inline]
    pub fn num_landmarks(&self) -> usize {
        self.landmarks.len()
    }

    /// Returns the JL embedding column dimension $d$.
    #[inline]
    pub fn jl_dim(&self) -> usize {
        self.jl_dim
    }

    /// Returns the selected landmark vertex IDs.
    #[inline]
    pub fn landmarks(&self) -> &[u32] {
        &self.landmarks
    }

    /// Returns the row-major `(num_nodes, num_landmarks)` effective kernel block $K_{\mathrm{eff}}[:, P]$.
    #[inline]
    pub fn landmark_kernel_matrix(&self) -> &[f64] {
        &self.landmark_kernel
    }

    /// Returns the positive JL-based diagonal estimate $\mathrm{diag}(K_{\mathrm{eff}})$ of length `num_nodes`.
    #[inline]
    pub fn diag_estimate(&self) -> &[f64] {
        &self.diag_estimate
    }

    /// Returns the row-major `(num_nodes, jl_dim)` spectral square-root embedding $\sqrt{h_+}(L_{\mathrm{sym}}) Z$.
    #[inline]
    pub fn jl_embedding(&self) -> &[f64] {
        &self.jl_embedding
    }

    /// Returns the exact $O(|E|)$ spectral moment statistics used to parameterize R1 DoG.
    #[inline]
    pub fn moment_stats(&self) -> &SpectralMomentStats {
        &self.moment_stats
    }

    /// Returns the PSD-projected spectral filter terms.
    #[inline]
    pub fn terms_psd(&self) -> &[(f64, f64)] {
        &self.terms_psd
    }

    /// Returns the effective short-range diffusion time $t_1$.
    #[inline]
    pub fn t1(&self) -> f64 {
        self.t1
    }

    /// Returns the effective long-range diffusion time $t_2$.
    #[inline]
    pub fn t2(&self) -> f64 {
        self.t2
    }

    /// Returns the effective DoG attenuation $\alpha$.
    #[inline]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Returns the Chebyshev polynomial degree used.
    #[inline]
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Returns the PSD shift $\delta_{\mathrm{psd}}$ added during R2 projection.
    #[inline]
    pub fn delta_psd(&self) -> f64 {
        self.delta_psd
    }

    /// Returns the minimum sampled value $\min_\lambda h(\lambda)$ prior to PSD projection.
    #[inline]
    pub fn h_min(&self) -> f64 {
        self.h_min
    }

    /// Returns the parameter-free automatic positivity floor `beta_auto`.
    #[inline]
    pub fn beta_auto(&self) -> f64 {
        self.beta_auto
    }

    /// Returns the effective rank-1 positivity floor `beta_eff` actually applied.
    #[inline]
    pub fn beta_eff(&self) -> f64 {
        self.beta_eff
    }

    /// Applies the full regularized DoG kernel operator
    /// $(1 - \beta_{\mathrm{eff}}) h_+(L_{\mathrm{sym}}) V + \beta_{\mathrm{eff}} \mathbf{1}(\mathbf{1}^T V)$
    /// to row-major `(num_nodes, ncols)` matrix `v`.
    pub fn apply_batch(
        &self,
        l_sym: &CsrMatrix,
        v: &[f64],
        out: &mut [f64],
        ncols: usize,
        ws: &mut ChebyshevWorkspace,
    ) -> CoreResult<()> {
        if l_sym.num_nodes() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                l_sym.num_nodes(),
                "ChebyshevDogKernel::apply_batch::l_sym",
            ));
        }
        ws.apply_batch_with_lambda_max(l_sym, self.lambda_max, &self.coeffs_h, v, out, ncols)?;
        self.apply_rank_one_floor(v, out, ncols);
        Ok(())
    }

    /// Applies the full regularized DoG kernel operator using a matrix-free
    /// [`LinearOperator<f64>`] representing $L_{\mathrm{sym}}$.
    pub fn apply_batch_operator<O: LinearOperator<f64> + ?Sized>(
        &self,
        l_sym: &mut O,
        v: &[f64],
        out: &mut [f64],
        ncols: usize,
        ws: &mut ChebyshevWorkspace,
    ) -> CoreResult<()> {
        if l_sym.dim() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                l_sym.dim(),
                "ChebyshevDogKernel::apply_batch_operator::l_sym",
            ));
        }
        ws.apply_batch_operator_with_lambda_max(
            l_sym,
            self.lambda_max,
            &self.coeffs_h,
            v,
            out,
            ncols,
        )?;
        self.apply_rank_one_floor(v, out, ncols);
        Ok(())
    }

    fn apply_rank_one_floor(&self, v: &[f64], out: &mut [f64], ncols: usize) {
        if self.num_nodes == 0 || ncols == 0 {
            return;
        }
        let one_minus_beta = 1.0 - self.beta_eff;
        for c in 0..ncols {
            let mut col_sum = 0.0_f64;
            for i in 0..self.num_nodes {
                col_sum += v[i * ncols + c];
            }
            let shift = self.beta_eff * col_sum;
            for i in 0..self.num_nodes {
                let idx = i * ncols + c;
                out[idx] = one_minus_beta * out[idx] + shift;
            }
        }
    }

    /// Converts the landmark kernel block $K_{\mathrm{eff}}[:, P]$ into a row-major
    /// `(num_nodes, num_landmarks)` Normalized Varadhan Geodesic coordinate table
    /// $\left[-4 t_1 \ln \mathrm{clip}\!\left(\frac{K_{u p}}{\sqrt{K_{uu} K_{pp}}}, \epsilon_{\mathrm{sim}}, 1\right)\right]^p$.
    pub fn varadhan_landmark_table(
        &self,
        p_exp: f64,
        degree_normalize: bool,
        eps_sim: f64,
    ) -> CoreResult<Vec<f64>> {
        if !p_exp.is_finite() || p_exp <= 0.0 {
            return Err(CoreError::invalid_input("p_exp must be finite and > 0"));
        }
        if !eps_sim.is_finite() || eps_sim <= 0.0 || eps_sim > 1.0 {
            return Err(CoreError::invalid_input(
                "eps_sim must be finite and in (0, 1]",
            ));
        }
        let n = self.num_nodes;
        let k_lm = self.landmarks.len();
        if n == 0 || k_lm == 0 {
            return Ok(Vec::new());
        }

        let mut k_p_exact = Vec::with_capacity(k_lm);
        let mut max_kp = 1e-12_f64;
        for (col, &lm) in self.landmarks.iter().enumerate() {
            let val = self.landmark_kernel[(lm as usize) * k_lm + col].max(1e-12);
            if val > max_kp {
                max_kp = val;
            }
            k_p_exact.push(val);
        }
        let max_ku = self
            .diag_estimate
            .iter()
            .fold(1e-12_f64, |acc, &v| acc.max(v));
        let k_scale = max_ku.max(max_kp);

        let mut out = vec![0.0_f64; n * k_lm];
        for i in 0..n {
            let ku = self.diag_estimate[i].max(1e-12);
            let row_off = i * k_lm;
            for col in 0..k_lm {
                let denom = if degree_normalize {
                    (ku * k_p_exact[col]).sqrt().max(1e-12)
                } else {
                    k_scale
                };
                let sim = (self.landmark_kernel[row_off + col] / denom).clamp(eps_sim, 1.0);
                let log_val = (-4.0 * self.t1 * sim.ln()).max(0.0);
                out[row_off + col] = log_val.powf(p_exp);
            }
        }
        Ok(out)
    }
}
