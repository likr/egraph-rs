use petgraph_drawing::DrawingValue;

/// Calculates the angle between two vectors in 2D space.
///
/// This function computes the angle (in radians) between two edges represented as
/// vectors from the origin to points (x1, y1) and (x2, y2). The angle is calculated
/// using the dot product formula: cos(θ) = (v1·v2)/(|v1|·|v2|).
///
/// # Parameters
///
/// * `x1`: The x-coordinate of the first vector
/// * `y1`: The y-coordinate of the first vector
/// * `x2`: The x-coordinate of the second vector
/// * `y2`: The y-coordinate of the second vector
///
/// # Returns
///
/// * `Some(angle)`: The angle between the vectors in radians, if it can be computed
/// * `None`: If the angle cannot be defined (e.g., when either vector has zero length)
///
pub fn edge_angle<S: DrawingValue>(x1: S, y1: S, x2: S, y2: S) -> Option<S> {
    let cos = (x1 * x2 + y1 * y2) / (x1.hypot(y1) * x2.hypot(y2));
    let angle = cos.acos();
    if angle.is_finite() {
        Some(angle)
    } else {
        None
    }
}
