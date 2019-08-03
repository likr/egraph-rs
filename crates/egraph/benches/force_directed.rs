#![feature(test)]
extern crate test;

use egraph::layout::force_directed::force::{LinkForce, ManyBodyForce};
use egraph::layout::force_directed::SimulationBuilder;
use egraph_petgraph_adapter::PetgraphWrapper;
use petgraph::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;
use test::Bencher;

#[bench]
fn bench_many_body_force(bench: &mut Bencher) {
    let n = 100;
    let mut graph: Graph<(), ()> = Graph::new();
    for _ in 0..n {
        graph.add_node(());
    }
    let graph = PetgraphWrapper::new(graph);

    let many_body_force = Rc::new(RefCell::new(ManyBodyForce::new()));
    let mut builder = SimulationBuilder::new();
    builder.add(many_body_force);
    builder.iterations = 10;

    bench.iter(|| {
        builder.start(&graph);
    })
}

#[bench]
fn bench_link_force(bench: &mut Bencher) {
    let n = 100;
    let mut graph: Graph<(), ()> = Graph::new();
    for _ in 0..n {
        graph.add_node(());
    }
    for i in 0..n {
        for j in i + 1..n {
            graph.add_edge(NodeIndex::new(i), NodeIndex::new(j), ());
        }
    }

    let graph = PetgraphWrapper::new(graph);

    let many_body_force = Rc::new(RefCell::new(LinkForce::new()));
    let mut builder = SimulationBuilder::new();
    builder.add(many_body_force);
    builder.iterations = 10;

    bench.iter(|| {
        builder.start(&graph);
    })
}
