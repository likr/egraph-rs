//! Slice-based `CsrGraph` constructors for Python NumPy and WASM `TypedArray` slices.

use egraph_core::CsrGraph;

use crate::error::FfiResult;

/// Builds a `CsrGraph<f64>` from host CSR slices (`indptr`, `indices`, optional `weights`).
pub fn build_csr_f64(
    indptr: &[u32],
    indices: &[u32],
    weights: Option<&[f64]>,
    directed: bool,
    sanitize: bool,
) -> FfiResult<CsrGraph<f64>> {
    CsrGraph::from_csr(indptr, indices, weights, directed, sanitize).map_err(Into::into)
}

/// Builds a `CsrGraph<f32>` from host CSR slices (`indptr`, `indices`, optional `weights`).
pub fn build_csr_f32(
    indptr: &[u32],
    indices: &[u32],
    weights: Option<&[f32]>,
    directed: bool,
    sanitize: bool,
) -> FfiResult<CsrGraph<f32>> {
    CsrGraph::from_csr(indptr, indices, weights, directed, sanitize).map_err(Into::into)
}

/// Builds a `CsrGraph<f64>` from flat interleaved edge endpoints `[u0, v0, u1, v1, ...]`.
pub fn build_edges_f64(
    num_nodes: usize,
    edges: &[u32],
    weights: Option<&[f64]>,
    directed: bool,
    sanitize: bool,
) -> FfiResult<CsrGraph<f64>> {
    CsrGraph::from_flat_edges(num_nodes, edges, weights, directed, sanitize).map_err(Into::into)
}

/// Builds a `CsrGraph<f32>` from flat interleaved edge endpoints `[u0, v0, u1, v1, ...]`.
pub fn build_edges_f32(
    num_nodes: usize,
    edges: &[u32],
    weights: Option<&[f32]>,
    directed: bool,
    sanitize: bool,
) -> FfiResult<CsrGraph<f32>> {
    CsrGraph::from_flat_edges(num_nodes, edges, weights, directed, sanitize).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorKind;

    #[test]
    fn test_build_csr_and_edges_f64_f32() {
        let g64 =
            build_csr_f64(&[0, 1, 2], &[1, 0], Some(&[2.5_f64, 2.5_f64]), false, false).unwrap();
        assert_eq!(g64.num_nodes(), 2);
        assert_eq!(g64.num_edges(), 1);
        assert_eq!(g64.neighbor_weights(0), Some(&[2.5_f64][..]));

        let g32 = build_csr_f32(&[0, 1, 1], &[1], Some(&[1.25_f32]), true, false).unwrap();
        assert!(g32.is_directed());
        assert_eq!(g32.neighbor_weights(0), Some(&[1.25_f32][..]));

        let e64 = build_edges_f64(3, &[0, 1, 1, 2, 0, 0], None, false, true).unwrap();
        assert_eq!(e64.num_edges(), 2);
        assert_eq!(e64.neighbors(1), &[0, 2]);

        let e32 =
            build_edges_f32(3, &[0, 1, 1, 2], Some(&[0.5_f32, 1.5_f32]), false, false).unwrap();
        assert_eq!(e32.num_edges(), 2);
        assert_eq!(e32.neighbor_weights(1), Some(&[0.5_f32, 1.5_f32][..]));
    }

    #[test]
    fn test_build_csr_and_edges_error_mapping() {
        let err_odd = build_edges_f64(3, &[0, 1, 2], None, false, true).unwrap_err();
        assert_eq!(err_odd.kind, ErrorKind::DimensionMismatch);

        let err_oob = build_edges_f32(2, &[0, 5], None, false, true).unwrap_err();
        assert_eq!(err_oob.kind, ErrorKind::IndexOutOfBounds);

        let err_nan =
            build_csr_f64(&[0, 1, 2], &[1, 0], Some(&[f64::NAN, 1.0]), false, true).unwrap_err();
        assert_eq!(err_nan.kind, ErrorKind::NonFiniteValue);
    }
}
