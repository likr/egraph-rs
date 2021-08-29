use petgraph::visit::{IntoEdgeReferences, IntoNodeIdentifiers, NodeCount};
use petgraph_algorithm_shortest_path::warshall_floyd;
use petgraph_layout_force_simulation::{Coordinates, Point};
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

pub struct KamadaKawai {
  k: Vec<Vec<f32>>,
  l: Vec<Vec<f32>>,
  pub eps: f32,
  pub width: f32,
  pub height: f32,
}

impl KamadaKawai {
  pub fn new<G, F>(graph: G, length: &mut F) -> KamadaKawai
  where
    G: IntoEdgeReferences + IntoNodeIdentifiers + NodeCount,
    G::NodeId: Eq + Hash,
    F: FnMut(G::EdgeRef) -> f32,
  {
    let eps = 1e-1;
    let width = 1000.;
    let height = 1000.;
    let n = graph.node_count();
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
    KamadaKawai {
      k,
      l,
      eps,
      width,
      height,
    }
  }

  pub fn select_node(&self, coordinates: &Coordinates<u32>) -> Option<usize> {
    let n = coordinates.len();
    let KamadaKawai { k, l, eps, .. } = self;
    let mut delta2_max = 0.;
    let mut m_target = 0;
    for m in 0..n {
      let Point { x: xm, y: ym, .. } = coordinates.points[m];
      let mut dedx = 0.;
      let mut dedy = 0.;
      for i in 0..n {
        if i != m {
          let Point { x: xi, y: yi, .. } = coordinates.points[i];
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
      None
    } else {
      Some(m_target)
    }
  }

  pub fn apply_to_node(&self, m: usize, coordinates: &mut Coordinates<u32>) {
    let n = coordinates.len();
    let KamadaKawai { k, l, eps, .. } = self;
    loop {
      let Point { x: xm, y: ym, .. } = coordinates.points[m];
      let mut hxx = 0.;
      let mut hyy = 0.;
      let mut hxy = 0.;
      let mut dedx = 0.;
      let mut dedy = 0.;
      for i in 0..n {
        if i != m {
          let Point { x: xi, y: yi, .. } = coordinates.points[i];
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
      coordinates.points[m].x -= delta_x;
      coordinates.points[m].y -= delta_y;
      break;
    }
  }

  pub fn run(&self, coordinates: &mut Coordinates<u32>) {
    while let Some(m) = self.select_node(coordinates) {
      self.apply_to_node(m, coordinates);
    }
  }
}

#[test]
fn test_kamada_kawai() {
  use petgraph::Graph;

  let n = 10;
  let mut graph = Graph::new_undirected();
  let nodes = (0..n).map(|_| graph.add_node(())).collect::<Vec<_>>();
  for i in 0..n {
    for j in 0..i {
      graph.add_edge(nodes[j], nodes[i], ());
    }
  }

  let mut coordinates = Coordinates::initial_placement(&graph);

  for &u in &nodes {
    println!("{:?}", coordinates.position(u));
  }

  let kamada_kawai = KamadaKawai::new(&graph, &mut |_| 1.);
  kamada_kawai.run(&mut coordinates);

  for &u in &nodes {
    println!("{:?}", coordinates.position(u));
  }
}
