use egraph_core::{
    CoreError, CoreResult, EuclideanLayout, EuclideanStorageMut, Layout2d, Layout3d, LayoutNd,
    Scalar,
};

mod sealed {
    pub trait Sealed {}
    impl<S: egraph_core::Scalar> Sealed for egraph_core::Layout2d<S> {}
    impl<S: egraph_core::Scalar> Sealed for egraph_core::Layout3d<S> {}
    impl<S: egraph_core::Scalar> Sealed for egraph_core::LayoutNd<S> {}
}

/// Internal allocation bridge for sealed [`EuclideanLayout`] targets (`Layout2d`, `Layout3d`, `LayoutNd`).
#[doc(hidden)]
pub trait EuclideanLayoutAlloc: EuclideanLayout + sealed::Sealed + Sized {
    fn fixed_dims() -> Option<usize>;
    fn alloc_zeros(num_nodes: usize, dims: usize) -> CoreResult<Self>;
}

impl<S: Scalar> EuclideanLayoutAlloc for Layout2d<S> {
    #[inline]
    fn fixed_dims() -> Option<usize> {
        Some(2)
    }

    #[inline]
    fn alloc_zeros(num_nodes: usize, _dims: usize) -> CoreResult<Self> {
        Ok(Layout2d::zeros(num_nodes))
    }
}

impl<S: Scalar> EuclideanLayoutAlloc for Layout3d<S> {
    #[inline]
    fn fixed_dims() -> Option<usize> {
        Some(3)
    }

    #[inline]
    fn alloc_zeros(num_nodes: usize, _dims: usize) -> CoreResult<Self> {
        Ok(Layout3d::zeros(num_nodes))
    }
}

impl<S: Scalar> EuclideanLayoutAlloc for LayoutNd<S> {
    #[inline]
    fn fixed_dims() -> Option<usize> {
        None
    }

    #[inline]
    fn alloc_zeros(num_nodes: usize, dims: usize) -> CoreResult<Self> {
        LayoutNd::zeros(num_nodes, dims)
    }
}

#[inline]
pub(super) fn validate_distance_slice(distances: &[f64], ctx: &'static str) -> CoreResult<()> {
    for &d in distances {
        if d.is_nan() || d == f64::NEG_INFINITY {
            return Err(CoreError::non_finite(ctx));
        }
        if d < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "{ctx}: distances must be >= 0"
            )));
        }
    }
    Ok(())
}

#[inline]
pub(super) fn resolve_target_dims<L: EuclideanLayoutAlloc>(
    configured: Option<usize>,
    ctx: &'static str,
) -> CoreResult<usize> {
    if let Some(d) = configured {
        if d == 0 {
            return Err(CoreError::invalid_input(format!(
                "{ctx}: dims must be >= 1"
            )));
        }
        if let Some(fixed) = L::fixed_dims()
            && d != fixed
        {
            return Err(CoreError::dimension_mismatch(fixed, d, ctx));
        }
        return Ok(d);
    }
    Ok(L::fixed_dims().unwrap_or(2))
}

#[inline]
pub(super) fn set_storage_row<S: Scalar>(
    storage: &mut EuclideanStorageMut<'_, S>,
    node: usize,
    vals: &[f64],
) {
    match storage {
        EuclideanStorageMut::Soa2d { x, y } => {
            if let Some(&v0) = vals.first() {
                x[node] = S::from_f64(v0);
            }
            if let Some(&v1) = vals.get(1) {
                y[node] = S::from_f64(v1);
            }
        }
        EuclideanStorageMut::Soa3d { x, y, z } => {
            if let Some(&v0) = vals.first() {
                x[node] = S::from_f64(v0);
            }
            if let Some(&v1) = vals.get(1) {
                y[node] = S::from_f64(v1);
            }
            if let Some(&v2) = vals.get(2) {
                z[node] = S::from_f64(v2);
            }
        }
        EuclideanStorageMut::FlatNd { coords, dims, .. } => {
            let d = *dims;
            let used = vals.len().min(d);
            let dst_row = &mut coords[node * d..node * d + used];
            for (dst, &src) in dst_row.iter_mut().zip(vals.iter()) {
                *dst = S::from_f64(src);
            }
        }
    }
}

pub(super) fn write_row_major_to_layout<L: EuclideanLayout>(
    layout: &mut L,
    src: &[f64],
    num_nodes: usize,
    src_dims: usize,
    overflow_msg: &'static str,
) -> CoreResult<()> {
    let mut storage = layout.as_euclidean_storage_mut();
    for i in 0..num_nodes {
        let row = &src[i * src_dims..(i + 1) * src_dims];
        set_storage_row(&mut storage, i, row);
    }
    if layout.validate_finite().is_err() {
        return Err(CoreError::numerical_breakdown(overflow_msg));
    }
    Ok(())
}

macro_rules! impl_euclidean_layout_generator {
    ($ty:ident) => {
        impl<'a, L: super::storage::EuclideanLayoutAlloc, W: egraph_core::Scalar>
            egraph_core::LayoutGenerator for $ty<'a, L, W>
        {
            type Output = L;

            fn generate(&mut self) -> egraph_core::CoreResult<Self::Output> {
                let dims = self.resolve_dims()?;
                let mut layout = L::alloc_zeros(self.num_nodes(), dims)?;
                self.generate_into_layout(&mut layout, dims)?;
                Ok(layout)
            }
        }
    };
}

pub(super) use impl_euclidean_layout_generator;
