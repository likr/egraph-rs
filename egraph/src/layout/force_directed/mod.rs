pub mod edge_bundling;
pub mod force;
pub mod simulation;

pub use self::force::{Link, Point};
use petgraph::graph::IndexType;
use petgraph::{EdgeType, Graph};
use std::collections::HashMap;
use std::f32;

pub fn initial_placement(n: usize) -> Vec<Point> {
    (0..n)
        .map(|i| {
            let r = 10. * (i as usize as f32).sqrt();
            let theta = f32::consts::PI * (3. - (5. as f32).sqrt()) * (i as usize as f32);
            let x = r * theta.cos();
            let y = r * theta.sin();
            Point::new(x, y)
        })
        .collect::<Vec<_>>()
}

pub fn initial_links<N, E, Ty: EdgeType, Ix: IndexType>(graph: &Graph<N, E, Ty, Ix>) -> Vec<Link> {
    let mut links = graph
        .edge_indices()
        .map(|edge| {
            let (source, target) = graph.edge_endpoints(edge).unwrap();
            Link::new(source.index(), target.index())
        })
        .collect::<Vec<_>>();
    let mut count: HashMap<usize, usize> = HashMap::new();
    for link in &links {
        if !count.contains_key(&link.source) {
            count.insert(link.source, 0);
        }
        if !count.contains_key(&link.target) {
            count.insert(link.target, 0);
        }
        {
            let v = count.get_mut(&link.source).unwrap();
            *v += 1;
        }
        {
            let v = count.get_mut(&link.target).unwrap();
            *v += 1;
        }
    }
    for mut link in &mut links {
        let source_count = *count.get(&link.source).unwrap();
        let target_count = *count.get(&link.target).unwrap();
        link.strength = 1. / source_count.min(target_count) as f32;
        link.bias = source_count as f32 / (source_count + target_count) as f32
    }
    links
}
