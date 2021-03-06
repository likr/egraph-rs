use petgraph::visit::{IntoEdgeReferences, IntoNodeIdentifiers, NodeCount};
use petgraph_algorithm_shortest_path::warshall_floyd;
use std::collections::HashMap;
use std::hash::Hash;

pub fn kamada_kawai<G, F>(
  graph: G,
  coordinates: &mut HashMap<G::NodeId, (f32, f32)>,
  length: &mut F,
  eps: f32,
  width: f32,
  height: f32,
) where
  G: IntoEdgeReferences + IntoNodeIdentifiers + NodeCount,
  G::NodeId: Eq + Hash,
  F: FnMut(G::EdgeRef) -> f32,
{
  let mut pos = graph
    .node_identifiers()
    .map(|u| coordinates[&u])
    .collect::<Vec<_>>();
  let n = pos.len();
  let d = warshall_floyd(graph, length);

  let mut d_max = 0.;
  for i in 0..n {
    for j in 0..n {
      if d[i][j] > d_max {
        d_max = d[i][j];
      }
    }
  }

  let size = if width < height { width } else { height };
  let mut k = vec![vec![0.; n]; n];
  let mut l = vec![vec![0.; n]; n];
  for i in 0..n {
    for j in 0..n {
      k[i][j] = 100. / (d[i][j] * d[i][j]);
      l[i][j] = 0.5 * size * d[i][j] / d_max
    }
  }

  loop {
    let mut delta2_max = 0.;
    let mut m_target = 0;
    for m in 0..n {
      let (xm, ym) = pos[m];
      let mut dedx = 0.;
      let mut dedy = 0.;
      for i in 0..n {
        if i != m {
          let (xi, yi) = pos[i];
          let dx = xm - xi;
          let dy = ym - yi;
          let d = (dx * dx + dy * dy).sqrt();
          dedx += k[m][i] * (1. - l[m][i] / d) * dx;
          dedy += k[m][i] * (1. - l[m][i] / d) * dy;
        }
      }
      let delta2 = dedx * dedx + dedy * dedy;
      if delta2 > delta2_max {
        delta2_max = delta2;
        m_target = m;
      }
    }

    if delta2_max < eps * eps {
      break;
    }

    let m = m_target;
    loop {
      let (xm, ym) = pos[m];
      let mut hxx = 0.;
      let mut hyy = 0.;
      let mut hxy = 0.;
      let mut dedx = 0.;
      let mut dedy = 0.;
      for i in 0..n {
        if i != m {
          let (xi, yi) = pos[i];
          let dx = xm - xi;
          let dy = ym - yi;
          let d = (dx * dx + dy * dy).sqrt();
          hxx += k[m][i] * (1. - l[m][i] * dy * dy / (d * d * d));
          hyy += k[m][i] * (1. - l[m][i] * dx * dx / (d * d * d));
          hxy += k[m][i] * l[m][i] * dx * dy / (d * d * d);
          dedx += k[m][i] * (1. - l[m][i] / d) * dx;
          dedy += k[m][i] * (1. - l[m][i] / d) * dy;
        }
      }
      if dedx * dedx + dedy * dedy < eps * eps {
        break;
      }
      let d = hxx * hyy - hxy * hxy;
      let delta_x = (hyy * dedx - hxy * dedy) / d;
      let delta_y = (hxx * dedy - hxy * dedx) / d;
      pos[m].0 -= delta_x;
      pos[m].1 -= delta_y;
      break;
    }
  }

  for (u, (x, y)) in graph.node_identifiers().zip(pos) {
    coordinates.insert(u, (x, y));
  }
}

pub struct KamadaKawai<G>
where
  G: IntoEdgeReferences + IntoNodeIdentifiers + NodeCount,
  G::NodeId: Eq + Hash,
{
  pub length: Box<dyn FnMut(G::EdgeRef) -> f32>,
  pub eps: f32,
  pub width: f32,
  pub height: f32,
}

impl<G> KamadaKawai<G>
where
  G: IntoEdgeReferences + IntoNodeIdentifiers + NodeCount,
  G::NodeId: Eq + Hash,
{
  pub fn new() -> KamadaKawai<G> {
    KamadaKawai {
      length: Box::new(|_| 1.),
      eps: 1e-1,
      width: 1000.,
      height: 1000.,
    }
  }

  pub fn apply(&mut self, graph: G, coordinates: &mut HashMap<G::NodeId, (f32, f32)>)
  where
    G: IntoEdgeReferences + IntoNodeIdentifiers + NodeCount,
    G::NodeId: Eq + Hash,
  {
    kamada_kawai(
      graph,
      coordinates,
      &mut self.length,
      self.eps,
      self.width,
      self.height,
    );
  }
}

#[test]
fn test_kamada_kawai() {
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
  let r = 500.;
  let width = 2. * r;
  let height = 2. * r;

  let mut coordinates = HashMap::new();
  for (i, &u) in nodes.iter().enumerate() {
    let t = 2. * PI * i as f32 / n as f32;
    coordinates.insert(u, (r * t.cos(), r * t.sin()));
  }

  for &u in &nodes {
    println!("{:?}", coordinates[&u]);
  }

  let mut drawing = KamadaKawai::new();
  drawing.width = width;
  drawing.height = height;
  drawing.apply(&graph, &mut coordinates);

  for &u in &nodes {
    println!("{:?}", coordinates[&u]);
  }
}
