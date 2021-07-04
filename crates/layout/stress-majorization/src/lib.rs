use petgraph::visit::{IntoEdgeReferences, IntoNodeIdentifiers, NodeCount};
use petgraph_algorithm_shortest_path::warshall_floyd;
use std::collections::HashMap;
use std::hash::Hash;

fn line_search(a: &[f32], dx: &[f32], d: &[f32]) -> f32 {
  let n = dx.len();
  let mut alpha = -dot(d, &dx);
  let mut s = 0.;
  for i in 0..n {
    for j in 0..n {
      s += d[i] * d[j] * a[i * n + j];
    }
  }
  alpha /= s;
  alpha
}

fn delta_f(a: &[f32], b: &[f32], x: &[f32], dx: &mut [f32]) {
  let n = b.len();
  for i in 0..n {
    dx[i] = 0.;
    for j in 0..n {
      dx[i] += a[i * n + j] * x[j];
    }
    dx[i] -= b[i];
  }
}

fn dot(u: &[f32], v: &[f32]) -> f32 {
  let n = u.len();
  let mut s = 0.;
  for i in 0..n {
    s += u[i] * v[i];
  }
  s
}

pub fn conjugate_gradient(a: &[f32], b: &[f32], x: &mut [f32], epsilon: f32) {
  let n = b.len();
  let mut dx = vec![0.; n];
  let mut d = vec![0.; n];
  delta_f(a, b, &x, &mut dx);
  for i in 0..n {
    d[i] = -dx[i];
  }
  let mut dx_norm0 = dot(&dx, &dx);
  for _ in 0..n {
    let alpha = line_search(a, &dx, &d);
    for i in 0..n {
      x[i] += alpha * d[i];
    }
    delta_f(a, b, &x, &mut dx);
    let dx_norm = dot(&dx, &dx);
    if dx_norm < epsilon {
      break;
    }
    let beta = dx_norm / dx_norm0;
    dx_norm0 = dx_norm;
    for i in 0..n {
      d[i] = beta * d[i] - dx[i];
    }
  }
}

fn stress(x: &[f32], y: &[f32], w: &[f32], d: &Vec<Vec<f32>>) -> f32 {
  let n = x.len() + 1;
  let mut s = 0.;
  for j in 1..n - 1 {
    for i in 0..j {
      let dx = x[i] - x[j];
      let dy = y[i] - y[j];
      let norm = (dx * dx + dy * dy).sqrt();
      let dij = d[i][j];
      let wij = w[i * n + j];
      let e = norm - dij;
      s += wij * e * e;
    }
  }
  for i in 0..n - 1 {
    let j = n - 1;
    let dx = x[i];
    let dy = y[i];
    let norm = (dx * dx + dy * dy).sqrt();
    let dij = d[i][j];
    let wij = w[i * n + j];
    let e = norm - dij;
    s += wij * e * e;
  }
  s
}

pub fn stress_majorization<G, F>(
  graph: G,
  coordinates: &mut HashMap<G::NodeId, (f32, f32)>,
  length: &mut F,
) where
  G: IntoEdgeReferences + IntoNodeIdentifiers + NodeCount,
  G::NodeId: Eq + Hash,
  F: FnMut(G::EdgeRef) -> f32,
{
  let pos = graph
    .node_identifiers()
    .map(|u| coordinates[&u])
    .collect::<Vec<_>>();
  let n = pos.len();
  let d = warshall_floyd(graph, length);

  let mut w = vec![0.; n * n];
  for j in 1..n {
    for i in 0..j {
      let dij = d[i][j];
      let wij = 1. / (dij * dij);
      w[i * n + j] = wij;
      w[j * n + i] = wij;
    }
  }

  let mut l_w = vec![0.; (n - 1) * (n - 1)];
  for j in 1..n - 1 {
    for i in 0..j {
      let wij = w[i * n + j];
      l_w[i * (n - 1) + j] = -wij;
      l_w[j * (n - 1) + i] = -wij;
      l_w[i * (n - 1) + i] += wij;
      l_w[j * (n - 1) + j] += wij;
    }
  }
  for i in 0..n - 1 {
    let j = n - 1;
    l_w[i * (n - 1) + i] += w[i * n + j];
  }

  let mut x_x = vec![0.; n - 1];
  let mut z_x = vec![0.; n - 1];
  let mut x_y = vec![0.; n - 1];
  let mut z_y = vec![0.; n - 1];
  for i in 0..n - 1 {
    let (xi, yi) = pos[i];
    x_x[i] = xi;
    z_x[i] = xi;
    x_y[i] = yi;
    z_y[i] = yi;
  }

  let epsilon = 1e-4;
  let mut l_z = vec![0.; (n - 1) * (n - 1)];
  let mut b = vec![0.; n - 1];
  let mut stress0 = stress(&z_x, &z_y, &w, &d);
  loop {
    for i in 1..n - 1 {
      for j in 0..i {
        let dx = z_x[i] - z_x[j];
        let dy = z_y[i] - z_y[j];
        let norm = (dx * dx + dy * dy).sqrt();
        let lij = if norm < 1e-4 {
          0.
        } else {
          -w[i * n + j] * d[i][j] / norm
        };
        l_z[i * (n - 1) + j] = lij;
        l_z[j * (n - 1) + i] = lij;
      }
    }
    for i in 0..n - 1 {
      let mut s = 0.;
      for j in 0..n - 1 {
        if i != j {
          s -= l_z[i * (n - 1) + j];
        }
      }
      let j = n - 1;
      let dx = z_x[i];
      let dy = z_y[i];
      let norm = (dx * dx + dy * dy).sqrt();
      s -= if norm < 0.1 {
        0.
      } else {
        -w[i * n + j] * d[i][j] / norm
      };
      l_z[i * (n - 1) + i] = s;
    }

    for i in 0..n - 1 {
      let mut s = 0.;
      for j in 0..n - 1 {
        s += l_z[i * (n - 1) + j] * z_x[j];
      }
      b[i] = s;
    }

    conjugate_gradient(&l_w, &b, &mut x_x, epsilon);
    for i in 0..n - 1 {
      let mut s = 0.;
      for j in 0..n - 1 {
        s += l_z[i * (n - 1) + j] * z_y[j];
      }
      b[i] = s;
    }
    conjugate_gradient(&l_w, &b, &mut x_y, epsilon);

    let stress = stress(&x_x, &x_y, &w, &d);
    if (stress0 - stress) / stress0 < epsilon {
      break;
    }
    stress0 = stress;
    for i in 0..n - 1 {
      z_x[i] = x_x[i];
      z_y[i] = x_y[i];
    }
  }
  for (i, u) in graph.node_identifiers().enumerate() {
    if i == n - 1 {
      coordinates.insert(u, (0., 0.));
    } else {
      coordinates.insert(u, (z_x[i], z_y[i]));
    }
  }
}

#[test]
fn test_conjugate_gradient() {
  let a = vec![3., 1., 1., 2.];
  let b = vec![6., 7.];
  let mut x = vec![2., 1.];
  let epsilon = 1e-4;
  conjugate_gradient(&a, &b, &mut x, epsilon);
  let x_exact = vec![1., 3.];
  let mut d = 0.;
  for i in 0..x.len() {
    let dx = x[i] - x_exact[i];
    d += dx * dx;
  }
  assert!(d < epsilon);
}

#[test]
fn test_stress_majorization() {
  use petgraph::Graph;
  use std::f32::consts::PI;

  let n = 10;
  let mut graph = Graph::new_undirected();
  let nodes = (0..n).map(|_| graph.add_node(())).collect::<Vec<_>>();
  for j in 1..n {
    for i in 0..j {
      graph.add_edge(nodes[i], nodes[j], ());
    }
  }
  let r = 500.;
  let mut coordinates = HashMap::new();
  for (i, &u) in nodes.iter().enumerate() {
    let t = 2. * PI * i as f32 / n as f32;
    coordinates.insert(u, (r * t.cos(), r * t.sin()));
  }

  for &u in &nodes {
    println!("{:?}", coordinates[&u]);
  }

  stress_majorization(&graph, &mut coordinates, &mut |_| 1.);
  for &u in &nodes {
    println!("{:?}", coordinates[&u]);
  }
}
