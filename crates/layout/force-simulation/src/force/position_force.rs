use crate::{Force, Point};
use petgraph::graph::{Graph, IndexType, NodeIndex};
use petgraph::EdgeType;

#[derive(Clone, Copy, Default)]
pub struct NodeArgument {
    pub strength: Option<f32>,
    pub x: Option<f32>,
    pub y: Option<f32>,
}

pub struct PositionForce {
    strength: Vec<f32>,
    x: Vec<Option<f32>>,
    y: Vec<Option<f32>>,
}

impl PositionForce {
    pub fn new<
        N,
        E,
        Ty: EdgeType,
        Ix: IndexType,
        F: FnMut(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> NodeArgument,
    >(
        graph: &Graph<N, E, Ty, Ix>,
        mut accessor: F,
    ) -> PositionForce {
        let n = graph.node_count();
        let mut strength = Vec::with_capacity(n);
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for u in graph.node_indices() {
            let argument = accessor(graph, u);
            strength.push(if let Some(v) = argument.strength {
                v
            } else {
                default_strength_accessor(graph, u)
            });
            x.push(argument.x);
            y.push(argument.y);
        }
        PositionForce { strength, x, y }
    }
}

impl Force for PositionForce {
    fn apply(&self, points: &mut Vec<Point>, alpha: f32) {
        for i in 0..points.len() {
            let strength = self.strength[i];
            let point = points.get_mut(i).unwrap();
            if let Some(xi) = self.x[i] {
                point.vx += (xi - point.x) * alpha * strength;
            }
            if let Some(yi) = self.y[i] {
                point.vy += (yi - point.y) * alpha * strength;
            }
        }
    }
}

pub fn default_strength_accessor<N, E, Ty: EdgeType, Ix: IndexType>(
    _graph: &Graph<N, E, Ty, Ix>,
    _u: NodeIndex<Ix>,
) -> f32 {
    0.1
}
