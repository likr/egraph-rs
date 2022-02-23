use petgraph::graph::{Graph, IndexType};
use petgraph::EdgeType;
use petgraph_layout_force_simulation::Coordinates;
use spade::{DelaunayTriangulation, HasPosition, Point2, Triangulation};
use std::collections::HashSet;

struct Point {
    x: f32,
    y: f32,
    index: usize,
}

impl Point {
    fn new(index: usize, x: f32, y: f32) -> Point {
        Point { x, y, index }
    }
}

impl HasPosition for Point {
    type Scalar = f32;
    fn position(&self) -> Point2<f32> {
        Point2::new(self.x, self.y)
    }
}

pub fn shape<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
    coordinates: &Coordinates<Ix>,
) -> f32 {
    let mut graph_edges = HashSet::new();
    for e in graph.edge_indices() {
        let (u, v) = graph.edge_endpoints(e).unwrap();
        graph_edges.insert((u.index(), v.index()));
        graph_edges.insert((v.index(), u.index()));
    }

    let mut triangulation: DelaunayTriangulation<_> = DelaunayTriangulation::new();
    for u in graph.node_indices() {
        let (x, y) = coordinates.position(u).unwrap();
        triangulation.insert(Point::new(u.index(), x, y)).unwrap();
    }

    let mut cap = 0;
    let mut cup = graph_edges.len();
    for edge in triangulation.directed_edges() {
        let u = edge.from().data().index;
        let v = edge.to().data().index;
        if graph_edges.contains(&(u, v)) {
            cap += 2;
        } else {
            cup += 2;
        }
    }

    cap as f32 / cup as f32
}
