use petgraph::graph::{Graph, IndexType};
use petgraph::EdgeType;
use petgraph_layout_force_simulation::Coordinates;

pub fn edge_crossings<N, E, Ty: EdgeType, Ix: IndexType>(
  graph: &Graph<N, E, Ty, Ix>,
  coordinates: &Coordinates<Ix>,
) -> usize {
  let edges = graph.edge_indices().collect::<Vec<_>>();
  let n = edges.len();
  let mut count = 0;

  for i in 1..n {
    let (source1, target1) = graph.edge_endpoints(edges[i]).unwrap();
    let (x11, y11) = coordinates.position(source1).unwrap();
    let (x12, y12) = coordinates.position(target1).unwrap();
    for j in 0..i {
      let (source2, target2) = graph.edge_endpoints(edges[j]).unwrap();
      if source1 == source2 || target1 == target2 {
        continue;
      }
      let (x21, y21) = coordinates.position(source2).unwrap();
      let (x22, y22) = coordinates.position(target2).unwrap();
      let s = (x12 - x11) * (y21 - y11) - (y11 - y12) * (x21 - x11);
      let t = (x12 - x11) * (y22 - y11) - (y11 - y12) * (x22 - x11);
      if s * t > 0. {
        continue;
      }
      let s = (x21 - x22) * (y11 - y21) - (y21 - y22) * (x11 - x21);
      let t = (x21 - x22) * (y12 - y21) - (y21 - y22) * (x12 - x21);
      if s * t > 0. {
        continue;
      }
      count += 1;
    }
  }
  count
}
