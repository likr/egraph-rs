use crate::graph::Graph;

#[repr(C)]
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct Point {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
}

impl Point {
    pub fn new(x: f32, y: f32) -> Point {
        Point {
            x: x,
            y: y,
            vx: 0.,
            vy: 0.,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Group {
    pub x: f32,
    pub y: f32,
}

impl Group {
    pub fn new(x: f32, y: f32) -> Group {
        Group { x: x, y: y }
    }
}

pub trait ForceContext {
    fn apply(&self, points: &mut Vec<Point>, alpha: f32);
}

pub trait Force<D, G: Graph<D>> {
    fn build(&self, graph: &G) -> Box<dyn ForceContext>;
}
