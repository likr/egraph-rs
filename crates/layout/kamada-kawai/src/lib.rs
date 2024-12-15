use ndarray::prelude::*;
use petgraph::visit::{IntoEdges, IntoNodeIdentifiers, NodeCount};
use petgraph_algorithm_shortest_path::{all_sources_dijkstra, DistanceMatrix, FullDistanceMatrix};
use petgraph_drawing::{Drawing, DrawingEuclidean2d, DrawingIndex, DrawingValue};

fn norm<S>(x: S, y: S) -> S
where
    S: DrawingValue,
{
    x.hypot(y).max(S::one())
}

pub struct KamadaKawai<S> {
    k: Array2<S>,
    l: Array2<S>,
    pub eps: S,
}

impl<S> KamadaKawai<S> {
    pub fn new<G, F>(graph: G, length: F) -> Self
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeCount,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        S: DrawingValue,
    {
        let l = all_sources_dijkstra(graph, length);
        KamadaKawai::new_with_distance_matrix(&l)
    }

    pub fn new_with_distance_matrix<N>(d: &FullDistanceMatrix<N, S>) -> Self
    where
        N: DrawingIndex,
        S: DrawingValue,
    {
        let eps = S::from_f32(1e-1).unwrap();
        let n = d.shape().0;

        let mut l = Array2::zeros((n, n));
        let mut k = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                l[[i, j]] = d.get_by_index(i, j);
                k[[i, j]] = S::one() / (l[[i, j]] * l[[i, j]]);
            }
        }
        KamadaKawai { k, l, eps }
    }

    pub fn select_node<N>(&self, drawing: &DrawingEuclidean2d<N, S>) -> Option<usize>
    where
        N: DrawingIndex,
        S: DrawingValue,
    {
        let n = drawing.len();
        let KamadaKawai { k, l, eps, .. } = self;
        let mut delta2_max = S::zero();
        let mut m_target = 0;
        for m in 0..n {
            let xm = drawing.raw_entry(m).0;
            let ym = drawing.raw_entry(m).1;
            let mut dedx = S::zero();
            let mut dedy = S::zero();
            for i in 0..n {
                if i != m {
                    let xi = drawing.raw_entry(i).0;
                    let yi = drawing.raw_entry(i).1;
                    let dx = xm - xi;
                    let dy = ym - yi;
                    let d = norm(dx, dy);
                    dedx += k[[m, i]] * (S::one() - l[[m, i]] / d) * dx;
                    dedy += k[[m, i]] * (S::one() - l[[m, i]] / d) * dy;
                }
            }
            let delta2 = dedx * dedx + dedy * dedy;
            if delta2 > delta2_max {
                delta2_max = delta2;
                m_target = m;
            }
        }

        if delta2_max < *eps * *eps {
            None
        } else {
            Some(m_target)
        }
    }

    pub fn apply_to_node<N>(&self, m: usize, drawing: &mut DrawingEuclidean2d<N, S>)
    where
        N: DrawingIndex,
        S: DrawingValue,
    {
        let n = drawing.len();
        let KamadaKawai { k, l, .. } = self;
        let xm = drawing.raw_entry(m).0;
        let ym = drawing.raw_entry(m).1;
        let mut hxx = S::zero();
        let mut hyy = S::zero();
        let mut hxy = S::zero();
        let mut dedx = S::zero();
        let mut dedy = S::zero();
        for i in 0..n {
            if i != m {
                let xi = drawing.raw_entry(i).0;
                let yi = drawing.raw_entry(i).1;
                let dx = xm - xi;
                let dy = ym - yi;
                let d = norm(dx, dy);
                let d3 = d * d * d;
                hxx += k[[m, i]] * (S::one() - l[[m, i]] * dy * dy / d3);
                hyy += k[[m, i]] * (S::one() - l[[m, i]] * dx * dx / d3);
                hxy += k[[m, i]] * l[[m, i]] * dx * dy / d3;
                dedx += k[[m, i]] * (S::one() - l[[m, i]] / d) * dx;
                dedy += k[[m, i]] * (S::one() - l[[m, i]] / d) * dy;
            }
        }
        let det = hxx * hyy - hxy * hxy;
        let delta_x = (hyy * dedx - hxy * dedy) / det;
        let delta_y = (hxx * dedy - hxy * dedx) / det;
        drawing.raw_entry_mut(m).0 -= delta_x;
        drawing.raw_entry_mut(m).1 -= delta_y;
    }

    pub fn run<N>(&self, drawing: &mut DrawingEuclidean2d<N, S>)
    where
        N: DrawingIndex,
        S: DrawingValue,
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

    let mut coordinates = DrawingEuclidean2d::initial_placement(&graph);

    for &u in &nodes {
        println!("{:?}", coordinates.position(u));
    }

    let kamada_kawai = KamadaKawai::new(&graph, &mut |_| 1.);
    kamada_kawai.run(&mut coordinates);

    for &u in &nodes {
        println!("{:?}", coordinates.position(u));
    }
}
