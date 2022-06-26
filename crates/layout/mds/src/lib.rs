use ndarray::prelude::*;
use petgraph::prelude::*;
use petgraph::{
    graph::{EdgeReference, IndexType},
    EdgeType,
};
use petgraph_algorithm_shortest_path::{multi_source_dijkstra, warshall_floyd};
use petgraph_layout_force_simulation::Coordinates;

fn cos(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let ab = a.dot(b);
    let aa = a.dot(a);
    let bb = b.dot(b);
    ab / (aa * bb).sqrt()
}

fn double_centering(delta: &Array2<f32>) -> Array2<f32> {
    let n = delta.shape()[0];
    let k = delta.shape()[1];
    let sum_col = delta.mean_axis(Axis(1)).unwrap();
    let sum_row = delta.mean_axis(Axis(0)).unwrap();
    let sum_all = sum_col.mean().unwrap();
    let mut c = Array::zeros((n, k));
    for i in 0..n {
        for j in 0..k {
            c[[i, j]] = (sum_col[i] + sum_row[j] - delta[[i, j]] - sum_all) / 2.;
        }
    }
    c
}

fn power_iteration(a: &Array2<f32>, eps: f32) -> (f32, Array1<f32>) {
    let n = a.shape()[0];
    let mut x = Array1::from_elem(n, 1. / n as f32);
    let mut x_next = a.dot(&x);
    for _ in 0..10 {
        if 1. - cos(&x_next, &x) < eps {
            break;
        }
        x_next /= x_next.dot(&x_next).sqrt();
        x = x_next;
        x_next = a.dot(&x);
    }
    let e = x_next.dot(&x_next) / x_next.dot(&x);
    x_next /= x_next.dot(&x_next).sqrt();
    (e, x_next)
}

fn eigendecomposition(a: &Array2<f32>, k: usize, eps: f32) -> (Array1<f32>, Array2<f32>) {
    let n = a.shape()[0];
    let mut b = a.clone();
    let mut e = Array1::zeros(k);
    let mut v = Array2::zeros((n, k));
    let (ei, vi) = power_iteration(&b, eps);
    e[0] = ei;
    v.slice_mut(s![.., 0]).assign(&vi);
    for i in 1..k {
        for r in 0..n {
            for c in 0..n {
                b[[r, c]] -= e[i - 1] * v[[r, i - 1]] * v[[c, i - 1]];
            }
        }
        let (ei, vi) = power_iteration(&b, eps);
        e[i] = ei;
        v.slice_mut(s![.., i]).assign(&vi);
    }
    (e, v)
}

fn classical_mds<N, E, Ty: EdgeType, Ix: IndexType, F: FnMut(EdgeReference<'_, E, Ix>) -> f32>(
    graph: &Graph<N, E, Ty, Ix>,
    length: &mut F,
    eps: f32,
) -> Coordinates<Ix> {
    let mut delta = warshall_floyd(&graph, length);
    delta = delta.mapv_into(|v| v.powi(2));
    let b = double_centering(&delta);
    let (e, v) = eigendecomposition(&b, 2, eps);
    let xy = v.dot(&Array2::from_diag(&e.mapv(|v| v.sqrt())));
    let mut coordinates = Coordinates::new(graph);
    for (i, u) in graph.node_indices().enumerate() {
        coordinates.set_position(u, (xy[[i, 0]], xy[[i, 1]]));
    }
    coordinates
}

fn pivot_mds<N, E, Ty: EdgeType, Ix: IndexType, F: FnMut(EdgeReference<'_, E, Ix>) -> f32>(
    graph: &Graph<N, E, Ty, Ix>,
    length: &mut F,
    sources: &[NodeIndex<Ix>],
    eps: f32,
) -> Coordinates<Ix> {
    let mut delta = multi_source_dijkstra(&graph, length, &sources);
    delta = delta.mapv_into(|v| v.powi(2));
    let c = double_centering(&delta);
    let ct_c = c.t().dot(&c);
    let (e, v) = eigendecomposition(&ct_c, 2, eps);
    let xy = v.dot(&Array2::from_diag(&e.mapv(|v| v.sqrt())));
    let xy = c.dot(&xy);
    let mut coordinates = Coordinates::new(graph);
    for (i, u) in graph.node_indices().enumerate() {
        coordinates.set_position(u, (xy[[i, 0]], xy[[i, 1]]));
    }
    coordinates
}

pub struct ClassicalMds {
    pub eps: f32,
}

impl ClassicalMds {
    pub fn new() -> ClassicalMds {
        ClassicalMds { eps: 1e-3 }
    }

    pub fn run<N, E, Ty: EdgeType, Ix: IndexType, F: FnMut(EdgeReference<'_, E, Ix>) -> f32>(
        &self,
        graph: &Graph<N, E, Ty, Ix>,
        length: &mut F,
    ) -> Coordinates<Ix> {
        classical_mds(graph, length, self.eps)
    }
}

pub struct PivotMds {
    pub eps: f32,
}

impl PivotMds {
    pub fn new() -> PivotMds {
        PivotMds { eps: 1e-3 }
    }

    pub fn run<N, E, Ty: EdgeType, Ix: IndexType, F: FnMut(EdgeReference<'_, E, Ix>) -> f32>(
        &self,
        graph: &Graph<N, E, Ty, Ix>,
        length: &mut F,
        sources: &[NodeIndex<Ix>],
    ) -> Coordinates<Ix> {
        pivot_mds(graph, length, sources, self.eps)
    }
}
