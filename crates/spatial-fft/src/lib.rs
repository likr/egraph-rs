//! Fast Fourier Transform and grid-accelerated density-field kernels for `egraph-rs`.

pub mod fft;
pub mod fft_r2c;
pub mod fit_field;
pub mod ibfft;
pub mod screened_poisson;

pub use fft::{Complex64, Fft2dWorkspace};
pub use fft_r2c::{Complex32, R2cFft2dWorkspaceF32};
pub use fit_field::{FitField2d, FitFieldConfig, FitFieldOutput, FitFieldStats};
pub use ibfft::{IbFft2d, IbFftConfig, IbFftGridSpec, InterpolationOrder};
pub use screened_poisson::{
    DensityFieldKernel, ScreenedPoissonConfig, ScreenedPoissonField2d, ScreenedPoissonOutput,
    ScreenedPoissonStats,
};

/// Returns the crate version.
pub fn crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::crate_version;

    #[test]
    fn spatial_fft_crate_version_non_empty() {
        assert!(!crate_version().is_empty());
    }
}
