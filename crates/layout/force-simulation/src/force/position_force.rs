use crate::{Force, Point};
use petgraph::graph::{Graph, IndexType, NodeIndex};
use petgraph::EdgeType;

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
        F1: FnMut(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> f32,
        F2: FnMut(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> Option<f32>,
        F3: FnMut(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> Option<f32>,
    >(
        graph: &Graph<N, E, Ty, Ix>,
        mut strength_accessor: F1,
        mut x_accessor: F2,
        mut y_accessor: F3,
    ) -> PositionForce {
        let strength = graph
            .node_indices()
            .map(|u| strength_accessor(graph, u))
            .collect();
        let x = graph.node_indices().map(|u| x_accessor(graph, u)).collect();
        let y = graph.node_indices().map(|u| y_accessor(graph, u)).collect();
        PositionForce { strength, x, y }
    }

    pub fn new_x<
        N,
        E,
        Ty: EdgeType,
        Ix: IndexType,
        F1: FnMut(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> f32,
        F2: FnMut(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> Option<f32>,
    >(
        graph: &Graph<N, E, Ty, Ix>,
        strength_accessor: F1,
        x_accessor: F2,
    ) -> PositionForce {
        PositionForce::new(graph, strength_accessor, x_accessor, |_, _| None)
    }

    pub fn new_y<
        N,
        E,
        Ty: EdgeType,
        Ix: IndexType,
        F1: FnMut(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> f32,
        F2: FnMut(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> Option<f32>,
    >(
        graph: &Graph<N, E, Ty, Ix>,
        strength_accessor: F1,
        y_accessor: F2,
    ) -> PositionForce {
        PositionForce::new(graph, strength_accessor, |_, _| None, y_accessor)
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
