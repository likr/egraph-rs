use ndarray::prelude::*;
use petgraph::visit::{IntoEdges, IntoNodeIdentifiers, NodeCount};
use petgraph_algorithm_shortest_path::warshall_floyd;
use petgraph_drawing::{Drawing, DrawingIndex};

fn norm(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt().max(1.)
}

pub struct KamadaKawai {
    k: Array2<f32>,
    l: Array2<f32>,
    pub eps: f32,
}

impl KamadaKawai {
    pub fn new<G, F>(graph: G, length: F) -> KamadaKawai
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeCount,
        G::NodeId: DrawingIndex,
        F: FnMut(G::EdgeRef) -> f32,
    {
        let l = warshall_floyd(graph, length);
        KamadaKawai::new_with_distance_matrix(&l)
    }

    pub fn new_with_distance_matrix(l: &Array2<f32>) -> KamadaKawai {
        let eps = 1e-1;
        let n = l.nrows();

        let mut k = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                k[[i, j]] = 1. / (l[[i, j]] * l[[i, j]]);
            }
        }
        KamadaKawai {
            k,
            l: l.clone(),
            eps,
        }
    }

    pub fn select_node<N>(&self, drawing: &Drawing<N, (f32, f32)>) -> Option<usize>
    where
        N: DrawingIndex,
    {
        let n = drawing.len();
        let KamadaKawai { k, l, eps, .. } = self;
        let mut delta2_max = 0.;
        let mut m_target = 0;
        for m in 0..n {
            let xm = drawing.coordinates[m].0;
            let ym = drawing.coordinates[m].1;
            let mut dedx = 0.;
            let mut dedy = 0.;
            for i in 0..n {
                if i != m {
                    let xi = drawing.coordinates[i].0;
                    let yi = drawing.coordinates[i].1;
                    let dx = xm - xi;
                    let dy = ym - yi;
                    let d = norm(dx, dy);
                    dedx += k[[m, i]] * (1. - l[[m, i]] / d) * dx;
                    dedy += k[[m, i]] * (1. - l[[m, i]] / d) * dy;
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

    pub fn apply_to_node<N>(&self, m: usize, drawing: &mut Drawing<N, (f32, f32)>)
    where
        N: DrawingIndex,
    {
        let n = drawing.len();
        let KamadaKawai { k, l, .. } = self;
        let xm = drawing.coordinates[m].0;
        let ym = drawing.coordinates[m].1;
        let mut hxx = 0.;
        let mut hyy = 0.;
        let mut hxy = 0.;
        let mut dedx = 0.;
        let mut dedy = 0.;
        for i in 0..n {
            if i != m {
                let xi = drawing.coordinates[i].0;
                let yi = drawing.coordinates[i].1;
                let dx = xm - xi;
                let dy = ym - yi;
                let d = norm(dx, dy);
                let d3 = d * d * d;
                hxx += k[[m, i]] * (1. - l[[m, i]] * dy * dy / d3);
                hyy += k[[m, i]] * (1. - l[[m, i]] * dx * dx / d3);
                hxy += k[[m, i]] * l[[m, i]] * dx * dy / d3;
                dedx += k[[m, i]] * (1. - l[[m, i]] / d) * dx;
                dedy += k[[m, i]] * (1. - l[[m, i]] / d) * dy;
            }
        }
        let det = hxx * hyy - hxy * hxy;
        let delta_x = (hyy * dedx - hxy * dedy) / det;
        let delta_y = (hxx * dedy - hxy * dedx) / det;
        drawing.coordinates[m].0 -= delta_x;
        drawing.coordinates[m].1 -= delta_y;
    }

    pub fn run<N>(&self, drawing: &mut Drawing<N, (f32, f32)>)
    where
        N: DrawingIndex,
    {
        while let Some(m) = self.select_node(drawing) {
            self.apply_to_node(m, drawing);
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

    let mut coordinates = Drawing::initial_placement(&graph);

    for &u in &nodes {
        println!("{:?}", coordinates.position(u));
    }

    let kamada_kawai = KamadaKawai::new(&graph, &mut |_| 1.);
    kamada_kawai.run(&mut coordinates);

    for &u in &nodes {
        println!("{:?}", coordinates.position(u));
    }
}
