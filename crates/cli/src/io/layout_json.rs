use std::path::Path;

use egraph::prelude::{CoreError, Layout2d, Layout3d, LayoutNd, ManifoldKind, ManifoldLayout2d};
use serde::Deserialize;

use super::{CliError, CliResult};

/// Parsed layout loaded from a JSON coordinates file.
#[derive(Debug, Clone)]
pub enum LoadedLayout {
    /// Standard Euclidean 2D layout.
    Euclidean2d(Layout2d<f64>),
    /// Standard Euclidean 3D layout.
    Euclidean3d(Layout3d<f64>),
    /// Arbitrary-dimension Euclidean N-D layout.
    Nd(LayoutNd<f64>),
    /// Non-Euclidean 2D manifold layout with physical metric parameters.
    Manifold(ManifoldLayout2d<f64>),
}

#[derive(Debug, Deserialize)]
struct LayoutCoordsObject {
    coords: Vec<[f64; 2]>,
}

#[derive(Debug, Deserialize)]
struct LayoutSplitObject {
    x: Vec<f64>,
    y: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct LayoutNodesObject {
    nodes: Vec<LayoutNodePoint>,
}

#[derive(Debug, Deserialize)]
struct LayoutNodePoint {
    x: f64,
    y: f64,
}

fn coords_pairs_to_layout(pairs: &[[f64; 2]]) -> CliResult<Layout2d<f64>> {
    let mut xs = Vec::with_capacity(pairs.len());
    let mut ys = Vec::with_capacity(pairs.len());
    for &[x, y] in pairs {
        xs.push(x);
        ys.push(y);
    }
    Ok(Layout2d::from_xy(xs, ys)?)
}

fn parse_manifold_kind_str(raw: &str) -> CliResult<Option<ManifoldKind>> {
    let key = raw.trim().to_ascii_lowercase().replace('_', "-");
    match key.as_str() {
        "none" => Ok(None),
        "hyperbolic" | "hyperbolic-poincare" | "poincare" => {
            Ok(Some(ManifoldKind::HyperbolicPoincare))
        }
        "hyperbolic-lorentz" | "lorentz" | "hyperboloid" => {
            Ok(Some(ManifoldKind::HyperbolicLorentz))
        }
        "spherical" | "sphere" => Ok(Some(ManifoldKind::Spherical)),
        "torus" | "torus-2d" => Ok(Some(ManifoldKind::Torus2d)),
        _ => Err(CliError::InvalidInput(format!(
            "unrecognized manifold kind '{raw}'; expected none, hyperbolic, hyperbolic-lorentz, spherical, or torus"
        ))),
    }
}

pub(super) fn parse_2d_layout_value(
    val: &serde_json::Value,
    path: &Path,
    expected_nodes: usize,
) -> CliResult<Layout2d<f64>> {
    let layout = if val.get("coords").is_some() {
        let obj: LayoutCoordsObject = serde_json::from_value(val.clone())?;
        coords_pairs_to_layout(&obj.coords)?
    } else if val.get("x").is_some() && val.get("y").is_some() {
        let obj: LayoutSplitObject = serde_json::from_value(val.clone())?;
        Layout2d::from_xy(obj.x, obj.y)?
    } else if val.get("nodes").is_some() {
        let obj: LayoutNodesObject = serde_json::from_value(val.clone())?;
        let mut xs = Vec::with_capacity(obj.nodes.len());
        let mut ys = Vec::with_capacity(obj.nodes.len());
        for pt in obj.nodes {
            xs.push(pt.x);
            ys.push(pt.y);
        }
        Layout2d::from_xy(xs, ys)?
    } else if val.is_array() {
        let pairs: Vec<[f64; 2]> = serde_json::from_value(val.clone())?;
        coords_pairs_to_layout(&pairs)?
    } else {
        return Err(CliError::InvalidInput(format!(
            "unrecognized layout JSON schema in '{}'",
            path.display()
        )));
    };

    if layout.len() != expected_nodes {
        return Err(CliError::Core(CoreError::dimension_mismatch(
            expected_nodes,
            layout.len(),
            "load_layout_from_path",
        )));
    }
    layout.validate_finite()?;
    Ok(layout)
}

fn parse_nd_or_3d_layout_value(
    val: &serde_json::Value,
    path: &Path,
    expected_nodes: usize,
) -> CliResult<LoadedLayout> {
    let rows_val = val
        .get("coords_nd")
        .or_else(|| val.get("coords"))
        .ok_or_else(|| {
            CliError::InvalidInput(format!(
                "N-D layout JSON in '{}' must provide 'coords_nd' (or 'coords')",
                path.display()
            ))
        })?;
    let rows: Vec<Vec<f64>> = serde_json::from_value(rows_val.clone())?;
    if rows.len() != expected_nodes {
        return Err(CliError::Core(CoreError::dimension_mismatch(
            expected_nodes,
            rows.len(),
            "load_nd_layout_from_path",
        )));
    }

    let explicit_dims = match val.get("dims") {
        Some(d_val) => {
            let d_u64 = d_val.as_u64().ok_or_else(|| {
                CliError::InvalidInput(format!(
                    "'dims' field in '{}' must be a positive integer",
                    path.display()
                ))
            })?;
            let d = usize::try_from(d_u64).map_err(|_| {
                CliError::InvalidInput(format!("'dims' overflows usize in '{}'", path.display()))
            })?;
            if d == 0 {
                return Err(CliError::Core(CoreError::invalid_input(
                    "dims must be >= 1",
                )));
            }
            Some(d)
        }
        None => None,
    };

    let dims = match explicit_dims {
        Some(d) => d,
        None => rows
            .first()
            .map(Vec::len)
            .filter(|&d| d >= 1)
            .ok_or_else(|| {
                CliError::InvalidInput(format!(
                    "cannot infer N-D 'dims' for empty layout in '{}'; specify 'dims' explicitly",
                    path.display()
                ))
            })?,
    };

    if dims == 3 {
        let mut xs = Vec::with_capacity(expected_nodes);
        let mut ys = Vec::with_capacity(expected_nodes);
        let mut zs = Vec::with_capacity(expected_nodes);
        for (idx, row) in rows.iter().enumerate() {
            if row.len() != 3 {
                return Err(CliError::InvalidInput(format!(
                    "3D coordinate row {idx} in '{}' has length {}, expected 3",
                    path.display(),
                    row.len()
                )));
            }
            xs.push(row[0]);
            ys.push(row[1]);
            zs.push(row[2]);
        }
        let layout3d = Layout3d::from_xyz(xs, ys, zs)?;
        layout3d.validate_finite()?;
        return Ok(LoadedLayout::Euclidean3d(layout3d));
    }

    let mut flat = Vec::with_capacity(expected_nodes * dims);
    for (idx, row) in rows.iter().enumerate() {
        if row.len() != dims {
            return Err(CliError::InvalidInput(format!(
                "N-D coordinate row {idx} in '{}' has length {}, expected {dims}",
                path.display(),
                row.len()
            )));
        }
        flat.extend_from_slice(row);
    }

    let nd = LayoutNd::from_flat(expected_nodes, dims, &flat)?;
    nd.validate_finite()?;
    Ok(LoadedLayout::Nd(nd))
}

pub(super) fn parse_any_layout_value(
    val: &serde_json::Value,
    path: &Path,
    expected_nodes: usize,
) -> CliResult<LoadedLayout> {
    if let Some(m_val) = val.get("manifold") {
        let m_str = m_val.as_str().ok_or_else(|| {
            CliError::InvalidInput(format!(
                "'manifold' field in '{}' must be a string",
                path.display()
            ))
        })?;
        if let Some(kind) = parse_manifold_kind_str(m_str)? {
            let layout2d = parse_2d_layout_value(val, path, expected_nodes)?;
            let mut manifold = ManifoldLayout2d::from_layout(kind, layout2d)?;
            let scale = match val.get("manifold_scale") {
                Some(v) => Some(v.as_f64().ok_or_else(|| {
                    CliError::InvalidInput(format!(
                        "'manifold_scale' in '{}' must be a finite number",
                        path.display()
                    ))
                })?),
                None => None,
            };
            let aspect = match val.get("torus_aspect") {
                Some(v) => Some(v.as_f64().ok_or_else(|| {
                    CliError::InvalidInput(format!(
                        "'torus_aspect' in '{}' must be a finite number",
                        path.display()
                    ))
                })?),
                None => None,
            };
            if kind == ManifoldKind::Torus2d {
                let s = scale.unwrap_or(1.0);
                let a = aspect.unwrap_or(1.0);
                manifold.set_torus_extents(s * a, s)?;
            } else {
                if let Some(s) = scale {
                    manifold.set_scale(s)?;
                }
                if let Some(a) = aspect {
                    manifold.set_aspect(a)?;
                }
            }
            return Ok(LoadedLayout::Manifold(manifold));
        }
    }

    let is_nd = val.get("coords_nd").is_some()
        || val
            .get("dims")
            .and_then(|d| d.as_u64())
            .is_some_and(|d| d != 2);
    if is_nd {
        return parse_nd_or_3d_layout_value(val, path, expected_nodes);
    }

    Ok(LoadedLayout::Euclidean2d(parse_2d_layout_value(
        val,
        path,
        expected_nodes,
    )?))
}
