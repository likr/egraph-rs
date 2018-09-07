pub mod biclustering;

mod connected_components;
mod edge_concentration;

pub use self::connected_components::connected_components;
pub use self::edge_concentration::{
    edge_concentration,
    inter_group_edge_concentration,
};
