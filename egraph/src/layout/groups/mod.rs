pub mod radial;
pub mod treemap;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Group {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl Group {
    pub fn new(x: f64, y: f64, width: f64, height: f64) -> Group {
        Group {
            x,
            y,
            width,
            height,
        }
    }
}

pub trait Grouping {
    fn call(&self, width: f64, height: f64, values: &Vec<f64>) -> Vec<Group>;
}

pub use self::radial::RadialGrouping;
pub use self::treemap::TreemapGrouping;
