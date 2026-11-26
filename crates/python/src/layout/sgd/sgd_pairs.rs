//! Validated 1D NumPy pair extraction and builder helpers for legacy `Sgd`.

use crate::FloatType;
use crate::layout::knn::{extract_f64_array_1d, extract_index_array_1d_with_isize};
use petgraph_layout_sgd::Sgd;
use pyo3::prelude::*;

pub(super) type SgdPairTuple = (usize, usize, FloatType, FloatType, FloatType, FloatType);

pub(super) fn extract_validated_pairs_1d(
    sources: &Bound<'_, PyAny>,
    targets: &Bound<'_, PyAny>,
    distances: &Bound<'_, PyAny>,
    weights: &Bound<'_, PyAny>,
    n_nodes: Option<usize>,
    prefix: &str,
) -> PyResult<Vec<SgdPairTuple>> {
    let src_name = if prefix.is_empty() {
        "sources".to_string()
    } else {
        format!("{prefix}_sources")
    };
    let tgt_name = if prefix.is_empty() {
        "targets".to_string()
    } else {
        format!("{prefix}_targets")
    };
    let dist_name = if prefix.is_empty() {
        "distances".to_string()
    } else {
        format!("{prefix}_distances")
    };
    let wt_name = if prefix.is_empty() {
        "weights".to_string()
    } else {
        format!("{prefix}_weights")
    };

    let src = extract_index_array_1d_with_isize(sources, &src_name)?;
    let tgt = extract_index_array_1d_with_isize(targets, &tgt_name)?;
    let d_arr = extract_f64_array_1d(distances, &dist_name)?;
    let w_arr = extract_f64_array_1d(weights, &wt_name)?;
    let d_view = d_arr.as_array();
    let w_view = w_arr.as_array();

    let len = src.len();
    if tgt.len() != len || d_view.len() != len || w_view.len() != len {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "array lengths must match: {src_name}={}, {tgt_name}={}, {dist_name}={}, {wt_name}={}",
            len,
            tgt.len(),
            d_view.len(),
            w_view.len()
        )));
    }

    let mut pairs = Vec::with_capacity(len);
    for (idx, (((&u, &v), &d), &w)) in src
        .iter()
        .zip(tgt.iter())
        .zip(d_view.iter())
        .zip(w_view.iter())
        .enumerate()
    {
        if let Some(n) = n_nodes
            && (u >= n || v >= n)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "pair index ({u}, {v}) at position {idx} is out of bounds for n_nodes={n}"
            )));
        }
        if u == v {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "self-loop pair ({u}, {u}) at position {idx} is not allowed"
            )));
        }
        if !d.is_finite() || d <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{dist_name}[{idx}] must be finite and > 0, got {d}"
            )));
        }
        if !w.is_finite() || w <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{wt_name}[{idx}] must be finite and > 0, got {w}"
            )));
        }
        pairs.push((u, v, d, d, w, w));
    }
    Ok(pairs)
}

pub(super) fn build_sgd_from_numpy_pairs(
    sources: &Bound<'_, PyAny>,
    targets: &Bound<'_, PyAny>,
    distances: &Bound<'_, PyAny>,
    weights: &Bound<'_, PyAny>,
    n_nodes: Option<usize>,
) -> PyResult<Sgd<FloatType>> {
    let pairs = extract_validated_pairs_1d(sources, targets, distances, weights, n_nodes, "")?;
    Ok(Sgd::from_node_pairs(pairs))
}

pub(super) fn build_sgd_from_numpy_hinge_pairs(
    sources: &Bound<'_, PyAny>,
    targets: &Bound<'_, PyAny>,
    distances: &Bound<'_, PyAny>,
    weights: &Bound<'_, PyAny>,
    n_nodes: Option<usize>,
) -> PyResult<Sgd<FloatType>> {
    let hinges = extract_validated_pairs_1d(sources, targets, distances, weights, n_nodes, "")?;
    Ok(Sgd::new_with_hinge_pairs(Vec::new(), hinges))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_sgd_from_numpy_pairs_with_hinges(
    sources: &Bound<'_, PyAny>,
    targets: &Bound<'_, PyAny>,
    distances: &Bound<'_, PyAny>,
    weights: &Bound<'_, PyAny>,
    hinge_sources: &Bound<'_, PyAny>,
    hinge_targets: &Bound<'_, PyAny>,
    hinge_distances: &Bound<'_, PyAny>,
    hinge_weights: &Bound<'_, PyAny>,
    n_nodes: Option<usize>,
) -> PyResult<Sgd<FloatType>> {
    let springs = extract_validated_pairs_1d(sources, targets, distances, weights, n_nodes, "")?;
    let hinges = extract_validated_pairs_1d(
        hinge_sources,
        hinge_targets,
        hinge_distances,
        hinge_weights,
        n_nodes,
        "hinge",
    )?;
    Ok(Sgd::new_with_hinge_pairs(springs, hinges))
}

pub(super) fn append_hinge_pairs_from_numpy(
    sgd: &mut Sgd<FloatType>,
    sources: &Bound<'_, PyAny>,
    targets: &Bound<'_, PyAny>,
    distances: &Bound<'_, PyAny>,
    weights: &Bound<'_, PyAny>,
    n_nodes: Option<usize>,
) -> PyResult<()> {
    let hinges = extract_validated_pairs_1d(sources, targets, distances, weights, n_nodes, "")?;
    sgd.add_hinge_pairs(hinges);
    Ok(())
}
