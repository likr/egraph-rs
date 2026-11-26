//! Fused 2D field evaluation for tsNET C1 (Student-$t$ KL repulsion) and C2 (entropy repulsion).
//!
//! By evaluating C1 and C2 on a single shared [`IbFft2d`] particle grid, [`FitField2d`] halves
//! particle-grid forward 2D FFTs from `4` down to `2` and reduces total `fft_2d` calls per
//! iteration from `10` (unfused) to `8` (or `5` when entropy is disabled).

use egraph_core::{CoreError, CoreResult, Layout2d, Scalar};

use crate::fft::Fft2dWorkspace;
use crate::ibfft::{IbFft2d, IbFftGridSpec};

mod kernels;
#[cfg(test)]
mod tests;
mod types;

pub use types::{FitFieldConfig, FitFieldOutput, FitFieldStats};

/// Fused C1 (Student-$t$ KL) + C2 (entropy) 2D FFT field evaluator.
#[derive(Debug, Clone)]
pub struct FitField2d<S: Scalar = f64> {
    config: FitFieldConfig,
    ws: Fft2dWorkspace,
    ibfft: IbFft2d<S>,
    // Per-particle intermediate potentials in `f64` to prevent cancellation before normalization.
    phi_1_z: Vec<f64>,
    phi_1_k2: Vec<f64>,
    phi_x_k2: Vec<f64>,
    phi_y_k2: Vec<f64>,
    h1: Vec<f64>,
    h2: Vec<f64>,
    h3: Vec<f64>,
    last_stats: FitFieldStats,
}

impl<S: Scalar> Default for FitField2d<S> {
    fn default() -> Self {
        Self::new(FitFieldConfig::default()).expect("default FitFieldConfig is valid")
    }
}

impl<S: Scalar> FitField2d<S> {
    /// Constructs a new [`FitField2d`] workspace.
    pub fn new(config: FitFieldConfig) -> CoreResult<Self> {
        config.validate()?;
        let ibfft = IbFft2d::new(config.ibfft)?;
        Ok(Self {
            config,
            ws: Fft2dWorkspace::new(),
            ibfft,
            phi_1_z: Vec::new(),
            phi_1_k2: Vec::new(),
            phi_x_k2: Vec::new(),
            phi_y_k2: Vec::new(),
            h1: Vec::new(),
            h2: Vec::new(),
            h3: Vec::new(),
            last_stats: FitFieldStats::default(),
        })
    }

    /// Returns the active [`FitFieldConfig`].
    #[inline]
    pub fn config(&self) -> FitFieldConfig {
        self.config
    }

    /// Returns the adaptive FFT grid geometry computed by the most recent evaluation.
    #[inline]
    pub fn grid_spec(&self) -> Option<IbFftGridSpec> {
        self.ibfft.grid_spec()
    }

    /// Updates `epsilon_r` without reallocating workspaces.
    pub fn set_epsilon_r(&mut self, epsilon_r: f64) -> CoreResult<()> {
        if !epsilon_r.is_finite() || epsilon_r <= 0.0 {
            return Err(CoreError::invalid_input(
                "epsilon_r must be finite and strictly positive",
            ));
        }
        self.config.epsilon_r = epsilon_r;
        Ok(())
    }

    /// Returns telemetry from the most recent [`FitField2d::evaluate`] call.
    #[inline]
    pub fn last_stats(&self) -> FitFieldStats {
        self.last_stats
    }

    /// Returns cumulative 2D FFT calls executed by the owned [`Fft2dWorkspace`].
    #[inline]
    pub fn fft_2d_call_count(&self) -> u64 {
        self.ws.fft_2d_call_count()
    }

    /// Resets the owned [`Fft2dWorkspace`] call counter.
    #[inline]
    pub fn reset_fft_2d_call_count(&mut self) {
        self.ws.reset_fft_2d_call_count();
    }

    /// Evaluates fused C1 KL repulsion and (when enabled) C2 entropy repulsion on `layout`.
    pub fn evaluate(
        &mut self,
        layout: &Layout2d<S>,
        out: &mut FitFieldOutput<S>,
    ) -> CoreResult<()> {
        self.evaluate_xy(layout.x(), layout.y(), out)
    }

    /// Slice-based variant of [`FitField2d::evaluate`].
    pub fn evaluate_xy(
        &mut self,
        xs: &[S],
        ys: &[S],
        out: &mut FitFieldOutput<S>,
    ) -> CoreResult<()> {
        if xs.len() != ys.len() {
            return Err(CoreError::dimension_mismatch(
                xs.len(),
                ys.len(),
                "FitField2d::evaluate_xy",
            ));
        }
        let n = xs.len();
        out.resize_and_zero(n, self.config.compute_entropy);
        self.last_stats = FitFieldStats::default();

        let count_before = self.ws.fft_2d_call_count();
        let particle_fwds = self.ibfft.prepare_xy(&mut self.ws, xs, ys)?;
        if n < 2 {
            // Validate finiteness already occurred in `prepare_xy`.
            return Ok(());
        }

        let spec = self.ibfft.grid_spec().expect("spec must exist when n >= 2");
        self.resize_potentials(n);

        // --- C1: Student-t KL repulsion (1 kernel FFT + 2 inverse FFTs) ---
        self.build_kl_kernel_fft(&spec)?;
        self.convolve_c1(&spec)?;
        self.gather_c1_potentials(&spec);
        self.assemble_kl_forces(n, &spec, out);

        let mut kernel_fwds = 1u64;
        let mut inv_ffts = 2u64;

        // --- C2: Entropy repulsion reusing the exact same particle forward FFTs ---
        if self.config.compute_entropy {
            self.build_entropy_kernel_fft(&spec)?;
            self.convolve_c2(&spec)?;
            self.gather_c2_potentials(&spec);
            self.assemble_entropy_forces(out);
            kernel_fwds += 1;
            inv_ffts += 2;
        }

        let total_calls = self.ws.fft_2d_call_count() - count_before;
        let stats = FitFieldStats {
            particle_forward_ffts: particle_fwds,
            kernel_forward_ffts: kernel_fwds,
            inverse_ffts: inv_ffts,
            total_fft_2d_calls: total_calls,
        };
        out.stats = stats;
        self.last_stats = stats;
        Ok(())
    }

    fn resize_potentials(&mut self, n: usize) {
        self.phi_1_z.resize(n, 0.0);
        self.phi_1_k2.resize(n, 0.0);
        self.phi_x_k2.resize(n, 0.0);
        self.phi_y_k2.resize(n, 0.0);
        if self.config.compute_entropy {
            self.h1.resize(n, 0.0);
            self.h2.resize(n, 0.0);
            self.h3.resize(n, 0.0);
        }
    }
}
