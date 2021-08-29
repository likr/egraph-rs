use crate::Point;

pub trait Force {
    fn apply(&self, points: &mut [Point], alpha: f32);
}

pub trait ForceToNode {
    fn apply_to_node(&self, u: usize, points: &mut [Point], alpha: f32);
}
