/// returns the angle between the edges represented as vectors (x1, y1) and (x2, y2).
/// returns None if the angle could not be defined
pub fn edge_angle(x1: f32, y1: f32, x2: f32, y2: f32) -> Option<f32> {
    let cos = (x1 * x2 + y1 * y2) / (x1.hypot(y1) * x2.hypot(y2));
    let angle = cos.acos();
    if angle.is_finite() {
        Some(angle)
    } else {
        None
    }
}
