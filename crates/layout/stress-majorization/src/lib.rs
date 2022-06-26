use ndarray::prelude::*;
use petgraph::visit::{IntoEdgeReferences, IntoNodeIdentifiers, NodeCount};
use petgraph_algorithm_shortest_path::warshall_floyd;
use petgraph_layout_force_simulation::Coordinates;
use std::hash::Hash;

fn line_search(a: &Array2<f32>, dx: &Array1<f32>, d: &Array1<f32>) -> f32 {
    let n = dx.len();
    let mut alpha = -d.dot(dx);
    let mut s = 0.;
    for i in 0..n {
        for j in 0..n {
            s += d[i] * d[j] * a[[i, j]];
        }
    }
    alpha /= s;
    alpha
}

fn delta_f(a: &Array2<f32>, b: &Array1<f32>, x: &Array1<f32>, dx: &mut Array1<f32>) {
    let n = b.len();
    for i in 0..n {
        dx[i] = 0.;
        for j in 0..n {
            dx[i] += a[[i, j]] * x[j];
        }
        dx[i] -= b[i];
    }
}

pub fn conjugate_gradient(a: &Array2<f32>, b: &Array1<f32>, x: &mut Array1<f32>, epsilon: f32) {
    let n = b.len();
    let mut dx = Array1::zeros(n);
    let mut d = Array1::zeros(n);
    delta_f(a, b, &x, &mut dx);
    for i in 0..n {
        d[i] = -dx[i];
    }
    let mut dx_norm0 = dx.dot(&dx);
    for _ in 0..n {
        let alpha = line_search(a, &dx, &d);
        for i in 0..n {
            x[i] += alpha * d[i];
        }
        delta_f(a, b, &x, &mut dx);
        let dx_norm = dx.dot(&dx);
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

fn stress(x: &Array1<f32>, y: &Array1<f32>, w: &Array2<f32>, d: &Array2<f32>) -> f32 {
    let n = x.len() + 1;
    let mut s = 0.;
    for j in 1..n - 1 {
        for i in 0..j {
            let dx = x[i] - x[j];
            let dy = y[i] - y[j];
            let norm = (dx * dx + dy * dy).sqrt();
            let dij = d[[i, j]];
            let wij = w[[i, j]];
            let e = norm - dij;
            s += wij * e * e;
        }
    }
    for i in 0..n - 1 {
        let j = n - 1;
        let dx = x[i];
        let dy = y[i];
        let norm = (dx * dx + dy * dy).sqrt();
        let dij = d[[i, j]];
        let wij = w[[i, j]];
        let e = norm - dij;
        s += wij * e * e;
    }
    s
}

pub struct StressMajorization {
    d: Array2<f32>,
    w: Array2<f32>,
    l_w: Array2<f32>,
    l_z: Array2<f32>,
    b: Array1<f32>,
    stress: f32,
    x_x: Array1<f32>,
    x_y: Array1<f32>,
    epsilon: f32,
}

impl StressMajorization {
    pub fn new<G, F>(graph: G, coordinates: &Coordinates<u32>, length: &mut F) -> StressMajorization
    where
        G: IntoEdgeReferences + IntoNodeIdentifiers + NodeCount,
        G::NodeId: Eq + Hash,
        F: FnMut(G::EdgeRef) -> f32,
    {
        let d = warshall_floyd(graph, length);
        StressMajorization::new_with_distance_matrix(coordinates, &d)
    }

    pub fn new_with_distance_matrix(
        coordinates: &Coordinates<u32>,
        d: &Array2<f32>,
    ) -> StressMajorization {
        let n = coordinates.len();

        let mut w = Array2::zeros((n, n));
        for j in 1..n {
            for i in 0..j {
                let dij = d[[i, j]];
                let wij = 1. / (dij * dij);
                w[[i, j]] = wij;
                w[[j, i]] = wij;
            }
        }

        let mut l_w = Array2::zeros((n - 1, n - 1));
        for j in 1..n - 1 {
            for i in 0..j {
                let wij = w[[i, j]];
                l_w[[i, j]] = -wij;
                l_w[[j, i]] = -wij;
                l_w[[i, i]] += wij;
                l_w[[j, j]] += wij;
            }
        }
        for i in 0..n - 1 {
            let j = n - 1;
            l_w[[i, i]] += w[[i, j]];
        }

        let mut x_x = Array1::zeros(n - 1);
        let mut x_y = Array1::zeros(n - 1);
        for i in 0..n - 1 {
            x_x[i] = coordinates.points[i].x;
            x_y[i] = coordinates.points[i].y;
        }

        let epsilon = 1e-4;
        let l_z = Array2::zeros((n - 1, n - 1));
        let b = Array1::zeros(n - 1);
        let stress = stress(&x_x, &x_y, &w, &d);
        StressMajorization {
            b,
            d: d.clone(),
            l_w,
            l_z,
            w,
            x_x,
            x_y,
            stress,
            epsilon,
        }
    }

    pub fn apply(&mut self, coordinates: &mut Coordinates<u32>) -> f32 {
        let n = coordinates.len();
        let StressMajorization {
            b, d, l_w, l_z, w, ..
        } = self;
        for i in 1..n - 1 {
            for j in 0..i {
                let dx = coordinates.points[i].x - coordinates.points[j].x;
                let dy = coordinates.points[i].y - coordinates.points[j].y;
                let norm = (dx * dx + dy * dy).sqrt();
                let lij = if norm < 1e-4 {
                    0.
                } else {
                    -w[[i, j]] * d[[i, j]] / norm
                };
                l_z[[i, j]] = lij;
                l_z[[j, i]] = lij;
            }
        }
        for i in 0..n - 1 {
            let mut s = 0.;
            for j in 0..n - 1 {
                if i != j {
                    s -= l_z[[i, j]];
                }
            }
            let j = n - 1;
            let dx = coordinates.points[i].x;
            let dy = coordinates.points[i].y;
            let norm = (dx * dx + dy * dy).sqrt();
            s -= if norm < 1e-4 {
                0.
            } else {
                -w[[i, j]] * d[[i, j]] / norm
            };
            l_z[[i, i]] = s;
        }

        for i in 0..n - 1 {
            let mut s = 0.;
            for j in 0..n - 1 {
                s += l_z[[i, j]] * coordinates.points[j].x;
            }
            b[i] = s;
        }
        conjugate_gradient(&l_w, &b, &mut self.x_x, self.epsilon);

        for i in 0..n - 1 {
            let mut s = 0.;
            for j in 0..n - 1 {
                s += l_z[[i, j]] * coordinates.points[j].y;
            }
            b[i] = s;
        }
        conjugate_gradient(&l_w, &b, &mut self.x_y, self.epsilon);

        let stress = stress(&self.x_x, &self.x_y, &w, &d);
        let diff = (self.stress - stress) / self.stress;
        self.stress = stress;
        for i in 0..n - 1 {
            coordinates.points[i].x = self.x_x[i];
            coordinates.points[i].y = self.x_y[i];
        }
        diff
    }

    pub fn run(&mut self, coordinates: &mut Coordinates<u32>) {
        loop {
            if self.apply(coordinates) < self.epsilon {
                break;
            }
        }
    }
}

#[test]
fn test_conjugate_gradient() {
    let a = arr2(&[[3., 1.], [1., 2.]]);
    let b = arr1(&[6., 7.]);
    let mut x = arr1(&[2., 1.]);
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

    let n = 10;
    let mut graph = Graph::new_undirected();
    let nodes = (0..n).map(|_| graph.add_node(())).collect::<Vec<_>>();
    for j in 1..n {
        for i in 0..j {
            graph.add_edge(nodes[i], nodes[j], ());
        }
    }
    let mut coordinates = Coordinates::initial_placement(&graph);

    for &u in &nodes {
        println!("{:?}", coordinates.position(u));
    }

    let mut stress_majorization = StressMajorization::new(&graph, &coordinates, &mut |_| 1.);
    stress_majorization.run(&mut coordinates);

    for &u in &nodes {
        println!("{:?}", coordinates.position(u));
    }
}
