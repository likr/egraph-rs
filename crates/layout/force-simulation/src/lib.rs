pub mod force;
pub mod simulation;

pub use self::simulation::{Force, Point, Simulation};
use petgraph::graph::{Graph, IndexType, NodeIndex};
use petgraph::EdgeType;
use std::collections::HashMap;
use std::f32::consts::PI;

pub const MIN_DISTANCE: f32 = 1e-6;

pub fn initial_placement<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
) -> HashMap<NodeIndex<Ix>, (f32, f32)> {
    let mut result = HashMap::new();
    for (i, u) in graph.node_indices().enumerate() {
        let r = 10. * (i as usize as f32).sqrt();
        let theta = PI * (3. - (5. as f32).sqrt()) * (i as usize as f32);
        let x = r * theta.cos();
        let y = r * theta.sin();
        result.insert(u, (x, y));
    }
    result
}

pub fn force_connected<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
) -> Vec<Box<dyn Force>> {
    vec![
        Box::new(force::ManyBodyForce::new(&graph)),
        Box::new(force::LinkForce::new(&graph)),
        Box::new(force::CenterForce::new()),
    ]
}

pub fn force_nonconnected<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
) -> Vec<Box<dyn Force>> {
    vec![
        Box::new(force::ManyBodyForce::new(&graph)),
        Box::new(force::LinkForce::new(&graph)),
        Box::new(force::PositionForce::new(
            &graph,
            |_, _| 0.1,
            |_, _| Some(0.),
            |_, _| Some(0.),
        )),
    ]
}
