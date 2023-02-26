use crate::graph::{IndexType, PyGraph};
use petgraph::graph::{node_index, NodeIndex};
use petgraph_layout_force_simulation::{Coordinates, Point};
use pyo3::prelude::*;

#[pyclass]
#[pyo3(name = "Coordinates")]
pub struct PyCoordinates {
    coordinates: Coordinates<IndexType>,
}

impl PyCoordinates {
    pub fn new(coordinates: Coordinates<IndexType>) -> PyCoordinates {
        PyCoordinates { coordinates }
    }
    pub fn indices(&self) -> &[NodeIndex<IndexType>] {
        &self.coordinates.indices
    }

    pub fn indices_mut(&mut self) -> &mut [NodeIndex<IndexType>] {
        &mut self.coordinates.indices
    }

    pub fn points(&self) -> &[Point] {
        &self.coordinates.points
    }

    pub fn points_mut(&mut self) -> &mut [Point] {
        &mut self.coordinates.points
    }

    pub fn coordinates(&self) -> &Coordinates<IndexType> {
        &self.coordinates
    }

    pub fn coordinates_mut(&mut self) -> &mut Coordinates<IndexType> {
        &mut self.coordinates
    }

    pub fn position(&self, u: usize) -> Option<(f32, f32)> {
        let u = node_index(u);
        self.coordinates.position(u)
    }

    pub fn set_position(&mut self, u: usize, p: (f32, f32)) {
        let u = node_index(u);
        self.coordinates.set_position(u, p);
    }
}

#[pymethods]
impl PyCoordinates {
    pub fn x(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.coordinates.x(u)
    }

    pub fn y(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.coordinates.y(u)
    }

    pub fn set_x(&mut self, u: usize, x: f32) {
        let u = node_index(u);
        self.coordinates.set_x(u, x);
    }

    pub fn set_y(&mut self, u: usize, y: f32) {
        let u = node_index(u);
        self.coordinates.set_y(u, y);
    }

    pub fn vx(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.coordinates.vx(u)
    }

    pub fn vy(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.coordinates.vy(u)
    }

    pub fn len(&self) -> usize {
        self.coordinates.len()
    }

    pub fn centralize(&mut self) {
        self.coordinates.centralize();
    }

    pub fn update_position(&mut self, velocity_decay: f32) {
        self.coordinates.update_position(velocity_decay);
    }

    pub fn clamp_region(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) {
        self.coordinates.clamp_region(x0, y0, x1, y1);
    }

    #[staticmethod]
    pub fn initial_placement(graph: &PyGraph) -> PyCoordinates {
        PyCoordinates::new(Coordinates::initial_placement(graph.graph()))
    }

    #[staticmethod]
    pub fn initial_placement_with_bfs_order(graph: &PyGraph, s: usize) -> PyCoordinates {
        PyCoordinates::new(Coordinates::initial_placement_with_bfs_order(
            graph.graph(),
            node_index(s),
        ))
    }
}

pub fn register(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyCoordinates>()?;
    Ok(())
}
