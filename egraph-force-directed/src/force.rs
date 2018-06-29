#[derive(Copy, Clone, Debug)]
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

#[derive(Copy, Clone, Debug)]
pub struct Link {
    pub source: usize,
    pub target: usize,
    pub length: f32,
    pub strength: f32,
}

impl Link {
    pub fn new(source: usize, target: usize) -> Link {
        Link {
            source: source,
            target: target,
            length: 30.,
            strength: 1.,
        }
    }
}

pub trait Force {
    fn apply(&self, points: &mut Vec<Point>, alpha: f32);
}
