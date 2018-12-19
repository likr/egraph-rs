use super::force::{Force, ForceContext, Point};
use petgraph::graph::IndexType;
use petgraph::prelude::*;
use petgraph::EdgeType;

pub struct PositionForceContext {
    strength: Vec<f32>,
    x: Vec<Option<f32>>,
    y: Vec<Option<f32>>,
}

impl PositionForceContext {
    pub fn new(
        strength: Vec<f32>,
        x: Vec<Option<f32>>,
        y: Vec<Option<f32>>,
    ) -> PositionForceContext {
        PositionForceContext { strength, x, y }
    }
}

impl ForceContext for PositionForceContext {
    fn apply(&self, points: &mut Vec<Point>, alpha: f32) {
        for i in 0..points.len() {
            let strength = self.strength[i];
            let mut point = points[i];
            if let Some(xi) = self.x[i] {
                point.vx -= (point.x + point.vx - xi) * alpha * strength;
            }
            if let Some(yi) = self.y[i] {
                point.vy -= (point.y + point.vy - yi) * alpha * strength;
            }
        }
    }
}

pub struct PositionForce<N, E, Ty: EdgeType, Ix: IndexType> {
    pub strength: Box<Fn(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> f32>,
    pub x: Box<Fn(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> Option<f32>>,
    pub y: Box<Fn(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> Option<f32>>,
}

impl<N, E, Ty: EdgeType, Ix: IndexType> PositionForce<N, E, Ty, Ix> {
    pub fn new() -> PositionForce<N, E, Ty, Ix> {
        PositionForce {
            strength: Box::new(|_, _| 0.1),
            x: Box::new(|_, _| None),
            y: Box::new(|_, _| None),
        }
    }
}

impl<N, E, Ty: EdgeType, Ix: IndexType> Force<N, E, Ty, Ix> for PositionForce<N, E, Ty, Ix> {
    fn build(&self, graph: &Graph<N, E, Ty, Ix>) -> Box<ForceContext> {
        let strength_accessor = &self.strength;
        let strength = graph
            .node_indices()
            .map(|index| strength_accessor(graph, index))
            .collect();

        let x_accessor = &self.x;
        let x = graph
            .node_indices()
            .map(|index| x_accessor(graph, index))
            .collect();

        let y_accessor = &self.y;
        let y = graph
            .node_indices()
            .map(|index| y_accessor(graph, index))
            .collect();

        Box::new(PositionForceContext::new(strength, x, y))
    }
}
