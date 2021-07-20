pub mod map;
pub mod simulation;

use petgraph::visit::{GetAdjacencyMatrix, IntoEdgeReferences, IntoNodeIdentifiers, NodeCount};
use std::collections::HashMap;
use std::hash::Hash;

fn to_tangent_space(x: (f32, f32), y: (f32, f32)) -> (f32, f32) {
  let dx = y.0 - x.0;
  let dy = y.1 - x.1;
  let dr = 1. - x.0 * y.0 - x.1 * y.1;
  let di = x.1 * y.0 - x.0 * y.1;
  let d = dr * dr + di * di;
  let z = ((dr * dx + di * dy) / d, (dr * dy - di * dx) / d);
  let z_norm = (z.0 * z.0 + z.1 * z.1).sqrt();
  if z_norm < 1e-4 {
    return (0., 0.);
  }
  let e = ((1. + z_norm) / (1. - z_norm)).ln();
  if e.is_finite() {
    (z.0 / z_norm * e, z.1 / z_norm * e)
  } else {
    (z.0 / z_norm, z.1 / z_norm)
  }
}

fn from_tangent_space(x: (f32, f32), z: (f32, f32)) -> (f32, f32) {
  let z_norm = (z.0 * z.0 + z.1 * z.1).sqrt();
  let y = if z_norm < 1e-4 {
    (0., 0.)
  } else if z_norm.exp().is_infinite() {
    (z.0 / z_norm, z.1 / z_norm)
  } else {
    let e = ((1. - z_norm.exp()) / (1. + z_norm.exp())).abs();
    (z.0 / z_norm * e, z.1 / z_norm * e)
  };
  let dx = -y.0 - x.0;
  let dy = -y.1 - x.1;
  let dr = -1. - x.0 * y.0 - x.1 * y.1;
  let di = x.1 * y.0 - x.0 * y.1;
  let d = dr * dr + di * di;
  ((dr * dx + di * dy) / d, (dr * dy - di * dx) / d)
}

fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
  let min_distance = 1e-3;
  let dx = a.0 - b.0;
  let dy = a.1 - b.1;
  let d = (dx * dx + dy * dy).sqrt();
  if d < min_distance {
    min_distance
  } else {
    d
  }
}

pub fn non_euclidean_fruchterman_reingold<G>(
  graph: G,
  coordinates: &mut HashMap<G::NodeId, (f32, f32)>,
  repeat: usize,
  k: f32,
) where
  G: GetAdjacencyMatrix + IntoEdgeReferences + IntoNodeIdentifiers + NodeCount,
  G::NodeId: Eq + Hash,
{
  let indices = graph.node_identifiers().collect::<Vec<_>>();
  let mut pos = indices.iter().map(|u| coordinates[&u]).collect::<Vec<_>>();
  let n = graph.node_count();
  let matrix = graph.adjacency_matrix();

  let mut v = vec![(0., 0.); n];
  let mut z = vec![(0., 0.); n];
  for i in 0..repeat {
    for u in 0..n {
      for v in 0..n {
        z[v] = to_tangent_space(pos[u], pos[v]);
      }
      let mut vx = 0.;
      let mut vy = 0.;
      for v in 0..n {
        if u == v {
          continue;
        }
        let d = dist(z[u], z[v]);
        let t = (z[v].1 - z[u].1).atan2(z[v].0 - z[u].0);
        if graph.is_adjacent(&matrix, indices[u], indices[v]) {
          vx += d * d / k * t.cos();
          vy += d * d / k * t.sin();
        }
        vx -= k * k / d * t.cos();
        vy -= k * k / d * t.sin();
      }
      v[u] = (vx * (0.1 / (i + 1) as f32), vy * (0.1 / (i + 1) as f32));
    }
    for u in 0..n {
      pos[u] = from_tangent_space(pos[u], v[u]);
      let d = (pos[u].0 * pos[u].0 + pos[u].1 * pos[u].1).sqrt();
      if d >= 1. {
        pos[u].0 *= 0.99 / d;
        pos[u].1 *= 0.99 / d;
      }
    }
  }

  for (&u, (x, y)) in indices.iter().zip(pos) {
    coordinates.insert(u, (x, y));
  }
}

#[test]
fn test_non_euclidean_fruchterman_reingold() {
  use petgraph::Graph;
  use std::f32::consts::PI;

  let n = 10;
  let mut graph = Graph::new_undirected();
  let nodes = (0..n).map(|_| graph.add_node(())).collect::<Vec<_>>();
  for i in 0..n {
    for j in 0..i {
      graph.add_edge(nodes[j], nodes[i], ());
    }
  }
  let r = 0.5;

  let mut coordinates = HashMap::new();
  for (i, &u) in nodes.iter().enumerate() {
    let t = 2. * PI * i as f32 / n as f32;
    coordinates.insert(u, (r * t.cos(), r * t.sin()));
  }

  for &u in &nodes {
    println!("{:?}", coordinates[&u]);
  }

  non_euclidean_fruchterman_reingold(&graph, &mut coordinates, 300, 1.);

  for &u in &nodes {
    println!("{:?}", coordinates[&u]);
  }
}
