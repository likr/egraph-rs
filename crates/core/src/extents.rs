//! Shared 2D node bounding shapes and zero-copy node extent views.

use crate::error::{CoreError, CoreResult};
use crate::scalar::Scalar;

#[inline]
fn validate_non_negative_extent<S: Scalar>(val: S, ctx: &'static str) -> CoreResult<()> {
    if !val.is_finite() {
        return Err(CoreError::non_finite(ctx));
    }
    if val < S::ZERO {
        return Err(CoreError::invalid_input(format!(
            "{ctx}: extent must be non-negative"
        )));
    }
    Ok(())
}

/// 2D node bounding shape shared by overlap-removal, clustering, and layout metrics.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NodeShape2d<S: Scalar = f64> {
    /// Axis-aligned rectangle centered at the vertex with half-extents `(half_width, half_height)`.
    Rectangle { half_width: S, half_height: S },
    /// Disk centered at the vertex with radius `radius`.
    Circle { radius: S },
}

impl<S: Scalar> NodeShape2d<S> {
    /// Constructs a validated axis-aligned rectangular node shape.
    pub fn rectangle(half_width: S, half_height: S) -> CoreResult<Self> {
        validate_non_negative_extent(half_width, "NodeShape2d::rectangle::half_width")?;
        validate_non_negative_extent(half_height, "NodeShape2d::rectangle::half_height")?;
        Ok(Self::Rectangle {
            half_width,
            half_height,
        })
    }

    /// Constructs a validated circular node shape.
    pub fn circle(radius: S) -> CoreResult<Self> {
        validate_non_negative_extent(radius, "NodeShape2d::circle::radius")?;
        Ok(Self::Circle { radius })
    }

    /// Returns the horizontal half-extent of the bounding box containing this shape.
    #[inline]
    pub fn half_width(&self) -> S {
        match *self {
            Self::Rectangle { half_width, .. } => half_width,
            Self::Circle { radius } => radius,
        }
    }

    /// Returns the vertical half-extent of the bounding box containing this shape.
    #[inline]
    pub fn half_height(&self) -> S {
        match *self {
            Self::Rectangle { half_height, .. } => half_height,
            Self::Circle { radius } => radius,
        }
    }

    /// Validates that all stored extents are finite and non-negative.
    pub fn validate(&self) -> CoreResult<()> {
        match *self {
            Self::Rectangle {
                half_width,
                half_height,
            } => {
                validate_non_negative_extent(half_width, "NodeShape2d::half_width")?;
                validate_non_negative_extent(half_height, "NodeShape2d::half_height")
            }
            Self::Circle { radius } => validate_non_negative_extent(radius, "NodeShape2d::radius"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ExtentsBacking<'a, S: Scalar> {
    Split {
        half_widths: &'a [S],
        half_heights: &'a [S],
    },
    Radii(&'a [S]),
    Uniform {
        num_nodes: usize,
        half_width: S,
        half_height: S,
    },
    Shapes(&'a [NodeShape2d<S>]),
}

/// Zero-copy borrowed view over per-node 2D half-extents (`half_width`, `half_height`).
///
/// Supports split `(half_widths, half_heights)` slices, isotropic `radii` slices, uniform
/// constant extents without heap allocation, and heterogeneous [`NodeShape2d`] slices.
#[derive(Debug, Clone, Copy)]
pub struct NodeExtents2d<'a, S: Scalar = f64> {
    backing: ExtentsBacking<'a, S>,
}

impl<'a, S: Scalar> NodeExtents2d<'a, S> {
    /// Constructs a validated view from split `half_widths` and `half_heights` slices.
    pub fn new(half_widths: &'a [S], half_heights: &'a [S]) -> CoreResult<Self> {
        if half_widths.len() != half_heights.len() {
            return Err(CoreError::dimension_mismatch(
                half_widths.len(),
                half_heights.len(),
                "NodeExtents2d::new",
            ));
        }
        let view = Self {
            backing: ExtentsBacking::Split {
                half_widths,
                half_heights,
            },
        };
        view.validate()?;
        Ok(view)
    }

    /// Constructs a validated view from isotropic node `radii` (`half_width = half_height = r`).
    pub fn from_radii(radii: &'a [S]) -> CoreResult<Self> {
        let view = Self {
            backing: ExtentsBacking::Radii(radii),
        };
        view.validate()?;
        Ok(view)
    }

    /// Constructs a validated zero-allocation uniform extent view for `num_nodes` vertices.
    pub fn uniform(num_nodes: usize, half_width: S, half_height: S) -> CoreResult<Self> {
        let view = Self {
            backing: ExtentsBacking::Uniform {
                num_nodes,
                half_width,
                half_height,
            },
        };
        view.validate()?;
        Ok(view)
    }

    /// Constructs a validated view over a slice of [`NodeShape2d`] descriptors.
    pub fn from_shapes(shapes: &'a [NodeShape2d<S>]) -> CoreResult<Self> {
        let view = Self {
            backing: ExtentsBacking::Shapes(shapes),
        };
        view.validate()?;
        Ok(view)
    }

    /// Returns the number of vertices covered by this extent view.
    #[inline]
    pub fn len(&self) -> usize {
        match self.backing {
            ExtentsBacking::Split { half_widths, .. } => half_widths.len(),
            ExtentsBacking::Radii(radii) => radii.len(),
            ExtentsBacking::Uniform { num_nodes, .. } => num_nodes,
            ExtentsBacking::Shapes(shapes) => shapes.len(),
        }
    }

    /// Returns `true` if `self.len() == 0`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the horizontal half-extent for `node`.
    #[inline]
    pub fn half_width(&self, node: usize) -> CoreResult<S> {
        let n = self.len();
        if node >= n {
            return Err(CoreError::index_out_of_bounds(
                node,
                n,
                "NodeExtents2d::half_width",
            ));
        }
        Ok(match self.backing {
            ExtentsBacking::Split { half_widths, .. } => half_widths[node],
            ExtentsBacking::Radii(radii) => radii[node],
            ExtentsBacking::Uniform { half_width, .. } => half_width,
            ExtentsBacking::Shapes(shapes) => shapes[node].half_width(),
        })
    }

    /// Returns the vertical half-extent for `node`.
    #[inline]
    pub fn half_height(&self, node: usize) -> CoreResult<S> {
        let n = self.len();
        if node >= n {
            return Err(CoreError::index_out_of_bounds(
                node,
                n,
                "NodeExtents2d::half_height",
            ));
        }
        Ok(match self.backing {
            ExtentsBacking::Split { half_heights, .. } => half_heights[node],
            ExtentsBacking::Radii(radii) => radii[node],
            ExtentsBacking::Uniform { half_height, .. } => half_height,
            ExtentsBacking::Shapes(shapes) => shapes[node].half_height(),
        })
    }

    /// Returns `(half_width, half_height)` for `node`.
    #[inline]
    pub fn extents(&self, node: usize) -> CoreResult<(S, S)> {
        let n = self.len();
        if node >= n {
            return Err(CoreError::index_out_of_bounds(
                node,
                n,
                "NodeExtents2d::extents",
            ));
        }
        Ok(match self.backing {
            ExtentsBacking::Split {
                half_widths,
                half_heights,
            } => (half_widths[node], half_heights[node]),
            ExtentsBacking::Radii(radii) => (radii[node], radii[node]),
            ExtentsBacking::Uniform {
                half_width,
                half_height,
                ..
            } => (half_width, half_height),
            ExtentsBacking::Shapes(shapes) => {
                (shapes[node].half_width(), shapes[node].half_height())
            }
        })
    }

    /// Returns the [`NodeShape2d`] descriptor for `node`.
    #[inline]
    pub fn shape(&self, node: usize) -> CoreResult<NodeShape2d<S>> {
        let n = self.len();
        if node >= n {
            return Err(CoreError::index_out_of_bounds(
                node,
                n,
                "NodeExtents2d::shape",
            ));
        }
        Ok(match self.backing {
            ExtentsBacking::Split {
                half_widths,
                half_heights,
            } => NodeShape2d::Rectangle {
                half_width: half_widths[node],
                half_height: half_heights[node],
            },
            ExtentsBacking::Radii(radii) => NodeShape2d::Circle {
                radius: radii[node],
            },
            ExtentsBacking::Uniform {
                half_width,
                half_height,
                ..
            } => NodeShape2d::Rectangle {
                half_width,
                half_height,
            },
            ExtentsBacking::Shapes(shapes) => shapes[node].clone(),
        })
    }

    /// Returns contiguous `(half_widths, half_heights)` slices when directly slice-backed.
    #[inline]
    pub fn as_slices(&self) -> Option<(&'a [S], &'a [S])> {
        match self.backing {
            ExtentsBacking::Split {
                half_widths,
                half_heights,
            } => Some((half_widths, half_heights)),
            ExtentsBacking::Radii(radii) => Some((radii, radii)),
            ExtentsBacking::Uniform { .. } | ExtentsBacking::Shapes(_) => None,
        }
    }

    /// Materializes `(half_widths, half_heights)` into owned vectors.
    pub fn to_vecs(&self) -> (Vec<S>, Vec<S>) {
        match self.backing {
            ExtentsBacking::Split {
                half_widths,
                half_heights,
            } => (half_widths.to_vec(), half_heights.to_vec()),
            ExtentsBacking::Radii(radii) => (radii.to_vec(), radii.to_vec()),
            ExtentsBacking::Uniform {
                num_nodes,
                half_width,
                half_height,
            } => (vec![half_width; num_nodes], vec![half_height; num_nodes]),
            ExtentsBacking::Shapes(shapes) => {
                let mut hw = Vec::with_capacity(shapes.len());
                let mut hh = Vec::with_capacity(shapes.len());
                for s in shapes {
                    hw.push(s.half_width());
                    hh.push(s.half_height());
                }
                (hw, hh)
            }
        }
    }

    /// Validates that all stored extents are finite and non-negative.
    pub fn validate(&self) -> CoreResult<()> {
        match self.backing {
            ExtentsBacking::Split {
                half_widths,
                half_heights,
            } => {
                for &w in half_widths {
                    validate_non_negative_extent(w, "NodeExtents2d::half_widths")?;
                }
                for &h in half_heights {
                    validate_non_negative_extent(h, "NodeExtents2d::half_heights")?;
                }
            }
            ExtentsBacking::Radii(radii) => {
                for &r in radii {
                    validate_non_negative_extent(r, "NodeExtents2d::radii")?;
                }
            }
            ExtentsBacking::Uniform {
                half_width,
                half_height,
                ..
            } => {
                validate_non_negative_extent(half_width, "NodeExtents2d::uniform::half_width")?;
                validate_non_negative_extent(half_height, "NodeExtents2d::uniform::half_height")?;
            }
            ExtentsBacking::Shapes(shapes) => {
                for shape in shapes {
                    shape.validate()?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_shape_and_extents_variants() {
        let rect = NodeShape2d::rectangle(2.0_f64, 1.5).unwrap();
        assert_eq!(rect.half_width(), 2.0);
        assert_eq!(rect.half_height(), 1.5);
        let circ = NodeShape2d::circle(3.0_f64).unwrap();
        assert_eq!(circ.half_width(), 3.0);
        assert_eq!(circ.half_height(), 3.0);

        let hw = [1.0_f64, 2.5];
        let hh = [0.5_f64, 1.5];
        let split = NodeExtents2d::new(&hw, &hh).unwrap();
        assert_eq!(split.len(), 2);
        assert!(!split.is_empty());
        assert_eq!(split.half_width(1).unwrap(), 2.5);
        assert_eq!(split.half_height(0).unwrap(), 0.5);
        assert_eq!(split.extents(1).unwrap(), (2.5, 1.5));
        assert_eq!(split.as_slices(), Some((&hw[..], &hh[..])));
        assert_eq!(split.to_vecs(), (vec![1.0, 2.5], vec![0.5, 1.5]));

        let radii = [1.25_f64, 2.0];
        let rad_view = NodeExtents2d::from_radii(&radii).unwrap();
        assert_eq!(rad_view.as_slices(), Some((&radii[..], &radii[..])));
        assert!(matches!(
            rad_view.shape(0).unwrap(),
            NodeShape2d::Circle { radius } if radius == 1.25
        ));

        let uni = NodeExtents2d::uniform(3, 0.75_f64, 0.5).unwrap();
        assert_eq!(uni.len(), 3);
        assert_eq!(uni.as_slices(), None);
        assert_eq!(uni.extents(2).unwrap(), (0.75, 0.5));
        assert_eq!(uni.to_vecs(), (vec![0.75; 3], vec![0.5; 3]));

        let shapes = [rect, circ];
        let shape_view = NodeExtents2d::from_shapes(&shapes).unwrap();
        assert_eq!(shape_view.as_slices(), None);
        assert_eq!(shape_view.to_vecs(), (vec![2.0, 3.0], vec![1.5, 3.0]));
    }

    #[test]
    fn test_node_extents_validation_errors() {
        assert!(matches!(
            NodeShape2d::rectangle(-1.0_f64, 1.0),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            NodeShape2d::circle(f64::NAN),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            NodeExtents2d::new(&[1.0_f64], &[1.0, 2.0]),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            NodeExtents2d::from_radii(&[-0.1_f64]),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            NodeExtents2d::uniform(2, f64::INFINITY, 1.0),
            Err(CoreError::NonFiniteValue { .. })
        ));
        let view = NodeExtents2d::uniform(1, 1.0_f64, 1.0).unwrap();
        assert!(matches!(
            view.extents(1),
            Err(CoreError::IndexOutOfBounds { .. })
        ));
        assert!(matches!(
            view.shape(1),
            Err(CoreError::IndexOutOfBounds { .. })
        ));
    }
}
