use super::geometry::{find_x_neighbours, find_y_neighbours, Node, Rectangle};
use super::{
    generate_rectangle_no_overlap_constraints_x, generate_rectangle_no_overlap_constraints_y,
};
use petgraph_drawing::DrawingEuclidean2d;
use std::collections::BTreeSet;

#[test]
fn test_find_x_neighbours() {
    let rectangles = vec![
        Rectangle::new(0.0, 10.0, 0.0, 10.0),
        Rectangle::new(8.0, 18.0, 2.0, 12.0),
        Rectangle::new(20.0, 30.0, 2.0, 12.0),
        Rectangle::new(5.0, 15.0, 3.0, 12.0),
        Rectangle::new(-5.0, 5.0, 8.0, 18.0),
    ];
    let mut nodes = rectangles
        .into_iter()
        .enumerate()
        .map(|(i, r)| Node::new_x(i, r))
        .collect::<Vec<_>>();

    let mut scanline = BTreeSet::new();

    scanline.insert(nodes[0].index());
    find_x_neighbours(nodes[0].v, &scanline, &mut nodes);
    assert_eq!(nodes[0].prev.len(), 0);
    assert_eq!(nodes[0].next.len(), 0);

    scanline.insert(nodes[1].index());
    find_x_neighbours(nodes[1].v, &scanline, &mut nodes);
    assert_eq!(nodes[0].prev.len(), 0);
    assert_eq!(nodes[0].next.len(), 1);
    assert_eq!(nodes[1].prev.len(), 1);
    assert_eq!(nodes[1].next.len(), 0);

    scanline.insert(nodes[2].index());
    find_x_neighbours(nodes[2].v, &scanline, &mut nodes);
    assert_eq!(nodes[0].prev.len(), 0);
    assert_eq!(nodes[0].next.len(), 1);
    assert_eq!(nodes[1].prev.len(), 1);
    assert_eq!(nodes[1].next.len(), 1);
    assert_eq!(nodes[2].prev.len(), 1);
    assert_eq!(nodes[2].next.len(), 0);

    scanline.insert(nodes[3].index());
    find_x_neighbours(nodes[3].v, &scanline, &mut nodes);
    assert_eq!(nodes[0].prev.len(), 0);
    assert_eq!(nodes[0].next.len(), 2);
    assert_eq!(nodes[1].prev.len(), 2);
    assert_eq!(nodes[1].next.len(), 1);
    assert_eq!(nodes[2].prev.len(), 2);
    assert_eq!(nodes[2].next.len(), 0);
    assert_eq!(nodes[3].prev.len(), 1);
    assert_eq!(nodes[3].next.len(), 2);

    scanline.insert(nodes[4].index());
    find_x_neighbours(nodes[4].v, &scanline, &mut nodes);
    assert_eq!(nodes[0].prev.len(), 0);
    assert_eq!(nodes[0].next.len(), 2);
    assert_eq!(nodes[1].prev.len(), 2);
    assert_eq!(nodes[1].next.len(), 1);
    assert_eq!(nodes[2].prev.len(), 2);
    assert_eq!(nodes[2].next.len(), 0);
    assert_eq!(nodes[3].prev.len(), 2);
    assert_eq!(nodes[3].next.len(), 2);
    assert_eq!(nodes[4].prev.len(), 0);
    assert_eq!(nodes[4].next.len(), 1);
}

#[test]
fn test_find_y_neighbours() {
    let rectangles = vec![
        Rectangle::new(-5.0, 5.0, 8.0, 18.0),
        Rectangle::new(0.0, 10.0, 0.0, 10.0),
    ];
    let mut nodes = rectangles
        .into_iter()
        .enumerate()
        .map(|(i, r)| Node::new_y(i, r))
        .collect::<Vec<_>>();

    let mut scanline = BTreeSet::new();

    scanline.insert(nodes[0].index());
    find_y_neighbours(nodes[0].v, &scanline, &mut nodes);
    assert_eq!(nodes[0].prev.len(), 0);
    assert_eq!(nodes[0].next.len(), 0);

    scanline.insert(nodes[1].index());
    find_y_neighbours(nodes[1].v, &scanline, &mut nodes);
    assert_eq!(nodes[0].prev.len(), 1);
    assert_eq!(nodes[0].next.len(), 0);
    assert_eq!(nodes[1].prev.len(), 0);
    assert_eq!(nodes[1].next.len(), 1);
}

#[test]
fn test_find_x_neighbours_dup() {
    let rectangles = vec![
        Rectangle::new(0.0, 10.0, 0.0, 20.0),
        Rectangle::new(0.0, 10.0, 0.0, 20.0),
        Rectangle::new(0.0, 10.0, 0.0, 20.0),
    ];
    let mut nodes = rectangles
        .into_iter()
        .enumerate()
        .map(|(i, r)| Node::new_x(i, r))
        .collect::<Vec<_>>();

    let mut scanline = BTreeSet::new();

    scanline.insert(nodes[0].index());
    find_x_neighbours(nodes[0].v, &scanline, &mut nodes);
    assert_eq!(nodes[0].prev.len(), 0);
    assert_eq!(nodes[0].next.len(), 0);

    scanline.insert(nodes[1].index());
    find_x_neighbours(nodes[1].v, &scanline, &mut nodes);
    assert_eq!(nodes[0].prev.len(), 0);
    assert_eq!(nodes[0].next.len(), 1);
    assert_eq!(nodes[1].prev.len(), 1);
    assert_eq!(nodes[1].next.len(), 0);

    scanline.insert(nodes[2].index());
    find_x_neighbours(nodes[2].v, &scanline, &mut nodes);
    assert_eq!(nodes[0].prev.len(), 0);
    assert_eq!(nodes[0].next.len(), 2);
    assert_eq!(nodes[1].prev.len(), 1);
    assert_eq!(nodes[1].next.len(), 1);
    assert_eq!(nodes[2].prev.len(), 2);
    assert_eq!(nodes[2].next.len(), 0);
}

#[test]
fn test_generate_rectangle_no_overlap_constraints_x() {
    let mut graph = petgraph::Graph::<(), ()>::new();
    let n1 = graph.add_node(());
    let n2 = graph.add_node(());
    let n3 = graph.add_node(());
    let n4 = graph.add_node(());
    let n5 = graph.add_node(());
    let size = [
        vec![10.0, 10.0],
        vec![10.0, 10.0],
        vec![10.0, 10.0],
        vec![10.0, 10.0],
        vec![10.0, 10.0],
    ];
    let mut drawing = DrawingEuclidean2d::new(&graph);
    drawing.set_x(n1, 5.);
    drawing.set_y(n1, 5.);
    drawing.set_x(n2, 13.);
    drawing.set_y(n2, 7.);
    drawing.set_x(n3, 25.);
    drawing.set_y(n3, 7.);
    drawing.set_x(n4, 10.);
    drawing.set_y(n4, 8.);
    drawing.set_x(n5, 0.);
    drawing.set_y(n5, 13.);
    let constraints =
        generate_rectangle_no_overlap_constraints_x(&drawing, |u, d| size[u.index()][d]);
    assert_eq!(constraints.len(), 9);
}

#[test]
fn test_generate_rectangle_no_overlap_constraints_y() {
    let mut graph = petgraph::Graph::<(), ()>::new();
    let n1 = graph.add_node(());
    let n2 = graph.add_node(());
    let n3 = graph.add_node(());
    let size = [vec![10.0, 10.0], vec![10.0, 10.0], vec![10.0, 10.0]];
    let mut drawing = DrawingEuclidean2d::new(&graph);
    drawing.set_x(n1, 0.);
    drawing.set_y(n1, 0.);
    drawing.set_x(n2, 0.);
    drawing.set_y(n2, 4.);
    drawing.set_x(n3, 0.);
    drawing.set_y(n3, 8.);
    let constraints =
        generate_rectangle_no_overlap_constraints_y(&drawing, |u, d| size[u.index()][d]);
    assert_eq!(constraints.len(), 3);
}
