//! $O(N \log N + K)$ axis-aligned 2D rectangle overlap detection via x-sweep + interval tree.

use egraph_core::{
    CoreError, CoreResult, EuclideanLayout, EuclideanStorageRef, Layout2d, NodeExtents2d, Scalar,
};

mod sweep;
#[cfg(test)]
mod tests;

use sweep::{Rect64, sweep_rects, sweep_rects64};

/// Axis-aligned closed-boundary rectangle `[x_min, x_max] x [y_min, y_max]`.
///
/// Two rectangles overlap iff their interiors intersect strictly (`a.x_min < b.x_max &&
/// b.x_min < a.x_max && a.y_min < b.y_max && b.y_min < a.y_max`), so shared boundaries do not
/// count as overlaps.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct Rect2d<S: Scalar = f64> {
    pub x_min: S,
    pub x_max: S,
    pub y_min: S,
    pub y_max: S,
}

impl<S: Scalar> Rect2d<S> {
    /// Constructs a validated [`Rect2d`] checking finiteness and `x_min <= x_max`, `y_min <= y_max`.
    pub fn new(x_min: S, x_max: S, y_min: S, y_max: S) -> CoreResult<Self> {
        if !x_min.is_finite() || !x_max.is_finite() || !y_min.is_finite() || !y_max.is_finite() {
            return Err(CoreError::non_finite("Rect2d::new"));
        }
        if x_min > x_max {
            return Err(CoreError::invalid_input("Rect2d requires x_min <= x_max"));
        }
        if y_min > y_max {
            return Err(CoreError::invalid_input("Rect2d requires y_min <= y_max"));
        }
        Ok(Self {
            x_min,
            x_max,
            y_min,
            y_max,
        })
    }

    /// Constructs a [`Rect2d`] from center `(cx, cy)` and non-negative full `(width, height)`.
    pub fn from_center_size(cx: S, cy: S, width: S, height: S) -> CoreResult<Self> {
        if !cx.is_finite() || !cy.is_finite() || !width.is_finite() || !height.is_finite() {
            return Err(CoreError::non_finite("Rect2d::from_center_size"));
        }
        if width < S::ZERO || height < S::ZERO {
            return Err(CoreError::invalid_input(
                "Rect2d width and height must be >= 0",
            ));
        }
        let hw = width / S::TWO;
        let hh = height / S::TWO;
        Self::new(cx - hw, cx + hw, cy - hh, cy + hh)
    }

    /// Constructs a [`Rect2d`] from center `(cx, cy)` and non-negative `(half_width, half_height)`.
    pub fn from_center_extents(cx: S, cy: S, half_width: S, half_height: S) -> CoreResult<Self> {
        if !cx.is_finite() || !cy.is_finite() || !half_width.is_finite() || !half_height.is_finite()
        {
            return Err(CoreError::non_finite("Rect2d::from_center_extents"));
        }
        if half_width < S::ZERO || half_height < S::ZERO {
            return Err(CoreError::invalid_input(
                "Rect2d half_width and half_height must be >= 0",
            ));
        }
        Self::new(
            cx - half_width,
            cx + half_width,
            cy - half_height,
            cy + half_height,
        )
    }

    /// Returns a validated copy with `x_min` replaced.
    pub fn with_x_min(self, x_min: S) -> CoreResult<Self> {
        Self::new(x_min, self.x_max, self.y_min, self.y_max)
    }

    /// Returns a validated copy with `x_max` replaced.
    pub fn with_x_max(self, x_max: S) -> CoreResult<Self> {
        Self::new(self.x_min, x_max, self.y_min, self.y_max)
    }

    /// Returns a validated copy with `y_min` replaced.
    pub fn with_y_min(self, y_min: S) -> CoreResult<Self> {
        Self::new(self.x_min, self.x_max, y_min, self.y_max)
    }

    /// Returns a validated copy with `y_max` replaced.
    pub fn with_y_max(self, y_max: S) -> CoreResult<Self> {
        Self::new(self.x_min, self.x_max, self.y_min, y_max)
    }

    /// Returns `true` if `self` and `other` have strictly overlapping interiors.
    #[inline]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.x_min < other.x_max
            && other.x_min < self.x_max
            && self.y_min < other.y_max
            && other.y_min < self.y_max
    }
}

/// $O(N \log N + K)$ axis-aligned 2D rectangle overlap sweep detector.
pub struct RectangleOverlapSweep;

impl RectangleOverlapSweep {
    /// Finds all overlapping rectangle pairs `(i, j)` with `i < j`, sorted lexicographically.
    pub fn find_overlaps<S: Scalar>(rects: &[Rect2d<S>]) -> CoreResult<Vec<(usize, usize)>> {
        sweep_rects(rects, false)
    }

    /// Returns `true` if any two rectangles in `rects` overlap, stopping on the first overlap.
    pub fn has_overlaps<S: Scalar>(rects: &[Rect2d<S>]) -> CoreResult<bool> {
        Ok(!sweep_rects(rects, true)?.is_empty())
    }

    /// Convenience overlap finder from an SoA [`Layout2d<S>`] and per-node half-extents.
    pub fn find_layout_overlaps<S: Scalar>(
        layout: &Layout2d<S>,
        half_widths: &[S],
        half_heights: &[S],
    ) -> CoreResult<Vec<(usize, usize)>> {
        let rects = layout_rects(layout, half_widths, half_heights)?;
        Self::find_overlaps(&rects)
    }

    /// Convenience overlap check from an SoA [`Layout2d<S>`] and per-node half-extents.
    pub fn has_layout_overlaps<S: Scalar>(
        layout: &Layout2d<S>,
        half_widths: &[S],
        half_heights: &[S],
    ) -> CoreResult<bool> {
        let rects = layout_rects(layout, half_widths, half_heights)?;
        Self::has_overlaps(&rects)
    }

    /// Finds all overlapping node bounding-box pairs `(i, j)` (`i < j`) for any 2D
    /// [`EuclideanLayout`] (`Layout2d` or 2D `LayoutNd`) and [`NodeExtents2d`] view.
    ///
    /// Returns [`CoreError::DimensionMismatch`] if `layout.dims() != 2` or
    /// `extents.len() != layout.len()`.
    pub fn find_layout_extents_overlaps<L: EuclideanLayout + ?Sized>(
        layout: &L,
        extents: &NodeExtents2d<'_, L::Scalar>,
    ) -> CoreResult<Vec<(usize, usize)>> {
        let rects64 = layout_extents_rects64(
            layout,
            extents,
            "RectangleOverlapSweep::find_layout_extents_overlaps",
        )?;
        Ok(sweep_rects64(&rects64, false))
    }

    /// Returns `true` if any two node bounding boxes in `layout` with `extents` overlap,
    /// stopping on the first overlap.
    ///
    /// Returns [`CoreError::DimensionMismatch`] if `layout.dims() != 2` or
    /// `extents.len() != layout.len()`.
    pub fn has_layout_extents_overlaps<L: EuclideanLayout + ?Sized>(
        layout: &L,
        extents: &NodeExtents2d<'_, L::Scalar>,
    ) -> CoreResult<bool> {
        let rects64 = layout_extents_rects64(
            layout,
            extents,
            "RectangleOverlapSweep::has_layout_extents_overlaps",
        )?;
        Ok(!sweep_rects64(&rects64, true).is_empty())
    }
}

fn layout_rects<S: Scalar>(
    layout: &Layout2d<S>,
    half_widths: &[S],
    half_heights: &[S],
) -> CoreResult<Vec<Rect2d<S>>> {
    let n = layout.len();
    if half_widths.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            half_widths.len(),
            "RectangleOverlapSweep::half_widths",
        ));
    }
    if half_heights.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            half_heights.len(),
            "RectangleOverlapSweep::half_heights",
        ));
    }
    layout.validate_finite()?;
    let xs = layout.x();
    let ys = layout.y();
    let mut rects = Vec::with_capacity(n);
    for i in 0..n {
        let hw = half_widths[i];
        let hh = half_heights[i];
        if !hw.is_finite() || !hh.is_finite() {
            return Err(CoreError::non_finite("RectangleOverlapSweep::extents"));
        }
        if hw < S::ZERO || hh < S::ZERO {
            return Err(CoreError::invalid_input(
                "half_widths and half_heights must be >= 0",
            ));
        }
        rects.push(Rect2d::new(xs[i] - hw, xs[i] + hw, ys[i] - hh, ys[i] + hh)?);
    }
    Ok(rects)
}

fn layout_extents_rects64<L: EuclideanLayout + ?Sized>(
    layout: &L,
    extents: &NodeExtents2d<'_, L::Scalar>,
    context: &'static str,
) -> CoreResult<Vec<Rect64>> {
    if layout.dims() != 2 {
        return Err(CoreError::dimension_mismatch(2, layout.dims(), context));
    }
    let n = layout.len();
    if extents.len() != n {
        return Err(CoreError::dimension_mismatch(n, extents.len(), context));
    }
    layout.validate_finite()?;
    extents.validate()?;

    let mut rects64 = Vec::with_capacity(n);
    let push_box = |cx: f64, cy: f64, hw: f64, hh: f64, out: &mut Vec<Rect64>| -> CoreResult<()> {
        let x_min = cx - hw;
        let x_max = cx + hw;
        let y_min = cy - hh;
        let y_max = cy + hh;
        if !x_min.is_finite() || !x_max.is_finite() || !y_min.is_finite() || !y_max.is_finite() {
            return Err(CoreError::non_finite(context));
        }
        out.push(Rect64 {
            x_min,
            x_max,
            y_min,
            y_max,
        });
        Ok(())
    };

    match layout.as_euclidean_storage() {
        EuclideanStorageRef::Soa2d { x, y } => {
            if let Some((hw_slice, hh_slice)) = extents.as_slices() {
                for i in 0..n {
                    push_box(
                        x[i].to_f64(),
                        y[i].to_f64(),
                        hw_slice[i].to_f64(),
                        hh_slice[i].to_f64(),
                        &mut rects64,
                    )?;
                }
            } else {
                for i in 0..n {
                    let (hw, hh) = extents.extents(i)?;
                    push_box(
                        x[i].to_f64(),
                        y[i].to_f64(),
                        hw.to_f64(),
                        hh.to_f64(),
                        &mut rects64,
                    )?;
                }
            }
        }
        EuclideanStorageRef::FlatNd {
            coords, dims: 2, ..
        } => {
            if let Some((hw_slice, hh_slice)) = extents.as_slices() {
                for i in 0..n {
                    let base = i * 2;
                    push_box(
                        coords[base].to_f64(),
                        coords[base + 1].to_f64(),
                        hw_slice[i].to_f64(),
                        hh_slice[i].to_f64(),
                        &mut rects64,
                    )?;
                }
            } else {
                for i in 0..n {
                    let base = i * 2;
                    let (hw, hh) = extents.extents(i)?;
                    push_box(
                        coords[base].to_f64(),
                        coords[base + 1].to_f64(),
                        hw.to_f64(),
                        hh.to_f64(),
                        &mut rects64,
                    )?;
                }
            }
        }
        _ => {
            return Err(CoreError::dimension_mismatch(2, layout.dims(), context));
        }
    }

    Ok(rects64)
}
