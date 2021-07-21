#[macro_use]
extern crate force_derive;

pub mod collide_force;
pub mod link_force;
pub mod many_body_force;
pub mod position_force;
pub mod radial_force;

pub use self::collide_force::CollideForce;
pub use self::link_force::LinkForce;
pub use self::many_body_force::ManyBodyForce;
pub use self::position_force::PositionForce;
pub use self::radial_force::RadialForce;

use petgraph::graph::{Graph, IndexType};
use petgraph::EdgeType;
use petgraph_layout_force_simulation::{Force, ForceToNode, Point, MIN_DISTANCE};

pub fn force_connected<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
) -> Vec<Box<dyn Force>> {
    vec![
        Box::new(ManyBodyForce::new(&graph)),
        Box::new(LinkForce::new(&graph)),
    ]
}

pub fn force_nonconnected<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
) -> Vec<Box<dyn Force>> {
    vec![
        Box::new(ManyBodyForce::new(&graph)),
        Box::new(LinkForce::new(&graph)),
        Box::new(PositionForce::new(&graph, |_, _| {
            position_force::NodeArgument {
                strength: None,
                x: Some(0.),
                y: Some(0.),
            }
        })),
    ]
}
