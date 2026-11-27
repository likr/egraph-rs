//! Fast Fourier Transform and grid-accelerated density-field kernels for `egraph-rs`.

mod fft;
mod fft_r2c;
mod fit_field;
mod ibfft;
mod screened_poisson;

pub use fft::{Complex64, Fft2dWorkspace};
pub use fft_r2c::{Complex32, R2cFft2dWorkspaceF32};
pub use fit_field::{FitField2d, FitFieldConfig, FitFieldOutput, FitFieldStats};
pub use ibfft::{IbFft2d, IbFftConfig, IbFftGridSpec, InterpolationOrder};
pub use screened_poisson::{
    DensityFieldKernel, ScreenedPoissonConfig, ScreenedPoissonField2d, ScreenedPoissonOutput,
    ScreenedPoissonStats,
};

use egraph_core::{CoreError, CoreResult, Scalar};

#[inline]
pub(crate) fn deinterleave_flat_2d<S: Scalar>(
    coords: &[S],
    num_nodes: usize,
    sx: &mut Vec<S>,
    sy: &mut Vec<S>,
    context: &'static str,
) -> CoreResult<()> {
    let expected = num_nodes.saturating_mul(2);
    if coords.len() != expected {
        return Err(CoreError::dimension_mismatch(
            expected,
            coords.len(),
            context,
        ));
    }
    sx.clear();
    sy.clear();
    sx.reserve(num_nodes);
    sy.reserve(num_nodes);
    let (pairs, _) = coords.as_chunks::<2>();
    for &[x, y] in pairs {
        sx.push(x);
        sy.push(y);
    }
    Ok(())
}
