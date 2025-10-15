use petgraph::prelude::*;
use petgraph_drawing::DrawingEuclidean2d;
use petgraph_layout_separation_constraints::*;

#[test]
fn test_project_rectangle_no_overlap_constraints_2d() {
    // Create a graph with 2 overlapping nodes
    let mut graph = Graph::<(), ()>::new();
    let n1 = graph.add_node(());
    let n2 = graph.add_node(());
    let n3 = graph.add_node(());
    let n4 = graph.add_node(());
    let n5 = graph.add_node(());
    let nodes = [n1, n2, n3, n4, n5];

    // Create a drawing with the nodes positioned with overlap
    let mut drawing = DrawingEuclidean2d::<_, f32>::new(&graph);
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

    // Set node size so they overlap (each node is 10.0 wide)
    let size = [
        vec![10.0, 10.0],
        vec![10.0, 10.0],
        vec![10.0, 10.0],
        vec![10.0, 10.0],
        vec![10.0, 10.0],
    ];

    // Apply constraints to remove overlaps
    project_rectangle_no_overlap_constraints_2d(&mut drawing, |u, d| size[u.index()][d]);

    // Check that the nodes are no longer overlapping
    for j in 0..5 {
        for i in 0..j {
            let u = nodes[i];
            let v = nodes[j];
            let dx = (drawing.x(u).unwrap() - drawing.x(v).unwrap()).abs();
            let dy = (drawing.y(u).unwrap() - drawing.y(v).unwrap()).abs();
            let gap_x = (size[i][0] + size[j][0]) / 2.0;
            let gap_y = (size[i][1] + size[j][1]) / 2.0;
            assert!(
                dx >= gap_x || dy >= gap_y,
                "|x({i}) - x({j})| = {dx} >= {gap_x} or |y({i}) - y({j})| = {dy} >= {gap_y}"
            );
        }
    }
}
