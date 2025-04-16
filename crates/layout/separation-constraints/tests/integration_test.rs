use petgraph::graph::{Graph, NodeIndex, UnGraph};
use petgraph_drawing::{Drawing, DrawingEuclidean, MetricCartesian}; // Already imported, ensure it's used
use petgraph_layout_separation_constraints::{Constraint, ConstraintGraph}; // Ensure correct crate name

#[test]
fn test_constraint_new() {
    let c = Constraint::new(0, 1, 10.0);
    assert_eq!(c.left, 0);
    assert_eq!(c.right, 1);
    assert_eq!(c.gap, 10.0);
}

#[test]
fn test_constraint_graph_new() {
    let n = 3;
    // Create a simple graph with 3 nodes
    let mut graph: UnGraph<(), ()> = Graph::new_undirected();
    let nodes: Vec<NodeIndex<u32>> = (0..n).map(|_| graph.add_node(())).collect();

    let mut drawing = DrawingEuclidean::new(&graph, 2); // Pass graph by reference
                                                        // Re-apply 3-argument set calls explicitly
                                                        // Node 0: (0.0, 0.0)
    drawing.set(nodes[0], 0, 0.0); // x0
    drawing.set(nodes[0], 1, 0.0); // y0
                                   // Node 1: (5.0, 5.0)
    drawing.set(nodes[1], 0, 5.0); // x1
    drawing.set(nodes[1], 1, 5.0); // y1
                                   // Node 2: (10.0, 10.0)
    drawing.set(nodes[2], 0, 10.0); // x2
    drawing.set(nodes[2], 1, 10.0); // y2

    let constraints = vec![
        Constraint::new(nodes[0].index(), nodes[1].index(), 2.0), // Use node indices
        Constraint::new(nodes[1].index(), nodes[2].index(), 3.0),
    ];

    // Test for dimension 0 (x-coordinates) - Pass by reference now
    let _cg_x = ConstraintGraph::new(&drawing, 0, &constraints);

    // Minimal checks as internals are private
    // This primarily ensures `new` runs without panicking

    // Test for dimension 1 (y-coordinates) - Pass by reference now
    let _cg_y = ConstraintGraph::new(&drawing, 1, &constraints);

    // Add more checks here if internal state access becomes possible for tests.
}

#[test]
fn test_project() {
    let n = 3;
    let mut graph: UnGraph<(), ()> = Graph::new_undirected();
    let nodes: Vec<NodeIndex<u32>> = (0..n).map(|_| graph.add_node(())).collect();

    let mut drawing = DrawingEuclidean::new(&graph, 1); // Use 1D for simplicity
                                                        // Initial positions: 0, 1, 3 (violates constraint 0)
    drawing.set(nodes[0], 0, 0.0);
    drawing.set(nodes[1], 0, 1.0);
    drawing.set(nodes[2], 0, 3.0);

    let constraints = vec![
        Constraint::new(nodes[0].index(), nodes[1].index(), 2.0), // x0 + 2 <= x1 (violated: 0 + 2 > 1)
        Constraint::new(nodes[1].index(), nodes[2].index(), 1.0), // x1 + 1 <= x2 (satisfied: 1 + 1 < 3)
    ];

    let mut cg = ConstraintGraph::new(&drawing, 0, &constraints);

    // Extract x-coordinates into a mutable slice
    let mut x: Vec<f32> = (0..n).map(|i| *drawing.raw_entry(i).nth(0)).collect();

    // Project the coordinates
    cg.project(&mut x);

    // Check if constraints are now satisfied
    // Constraint 0: x[1] >= x[0] + 2.0
    assert!(
        x[1] >= x[0] + constraints[0].gap - 1e-6,
        "Constraint 0 failed: {} < {} + {}",
        x[1],
        x[0],
        constraints[0].gap
    );
    // Constraint 1: x[2] >= x[1] + 1.0
    assert!(
        x[2] >= x[1] + constraints[1].gap - 1e-6,
        "Constraint 1 failed: {} < {} + {}",
        x[2],
        x[1],
        constraints[1].gap
    );

    // Check specific values based on expected merge operation
    // Initial pos: 0, 1. violate c0 (gap 2). merge(0, 1, c0)
    // d = off0 + gap0 - off1 = 0 + 2.0 - 0 = 2.0
    // nvar_l=1, nvar_r=1. posL=0, posR=1.
    // new_pos0 = (posL * nL + (posR - d) * nR) / (nL + nR)
    //          = (0.0 * 1 + (1.0 - 2.0) * 1) / (1 + 1) = -0.5
    // off0 stays 0. off1 becomes d + old_off1 = 2.0 + 0 = 2.0
    // Final pos: x[0]=-0.5+0=-0.5, x[1]=-0.5+2.0=1.5, x[2]=3.0 (unaffected)
    assert!(
        (x[0] - (-0.5)).abs() < 1e-6,
        "x[0] expected -0.5, got {}",
        x[0]
    );
    assert!((x[1] - 1.5).abs() < 1e-6, "x[1] expected 1.5, got {}", x[1]);
    assert!((x[2] - 3.0).abs() < 1e-6, "x[2] expected 3.0, got {}", x[2]);
}

// Test that split_blocks returns true (no split) when LM >= 0
#[test]
fn test_split_blocks_no_split() {
    let n = 2;
    let mut graph: UnGraph<(), ()> = Graph::new_undirected();
    let nodes: Vec<NodeIndex<u32>> = (0..n).map(|_| graph.add_node(())).collect();

    let mut drawing = DrawingEuclidean::new(&graph, 1);
    // Initial positions violate the constraint
    drawing.set(nodes[0], 0, 0.0);
    drawing.set(nodes[1], 0, 0.5); // x0=0, x1=0.5

    let constraints = vec![
        Constraint::new(nodes[0].index(), nodes[1].index(), 1.0), // c0: x0 + 1 <= x1 (violated: 0+1 > 0.5)
    ];

    let mut cg = ConstraintGraph::new(&drawing, 0, &constraints);

    // --- Setup state by calling project to merge blocks ---
    let mut x_project: Vec<f32> = (0..n).map(|i| *drawing.raw_entry(i).nth(0)).collect();
    cg.project(&mut x_project);
    // After project: Block 0={0,1}, active={c0}, pos=-0.25, off={0:0, 1:1}. x_project = [-0.25, 0.75]

    // --- Call split_blocks with the projected coordinates ---
    // In this state, the LM for c0 should be >= 0, so no split should occur.
    let nosplit_result = cg.split_blocks(&x_project);

    assert!(nosplit_result, "Expected no split (true), but got false"); // Expecting NO split (true)
}

// Test that split_blocks returns false (split happened) when LM < 0
#[test]
fn test_split_blocks_with_split() {
    let n = 2;
    let mut graph: UnGraph<(), ()> = Graph::new_undirected();
    let nodes: Vec<NodeIndex<u32>> = (0..n).map(|_| graph.add_node(())).collect();

    let mut drawing = DrawingEuclidean::new(&graph, 1);
    // Initial positions that violate the constraint
    drawing.set(nodes[0], 0, 0.0);
    drawing.set(nodes[1], 0, 0.5); // x0=0, x1=0.5

    let constraints = vec![
        Constraint::new(nodes[0].index(), nodes[1].index(), 1.0), // c0: x0 + 1 <= x1 (violated)
    ];

    let mut cg = ConstraintGraph::new(&drawing, 0, &constraints);

    // --- Setup state by calling project to merge blocks ---
    let mut x_project: Vec<f32> = (0..n).map(|i| *drawing.raw_entry(i).nth(0)).collect();
    cg.project(&mut x_project);
    // After project: Block 0={0,1}, active={c0}, pos=-0.25, off={0:0, 1:1}. x_project = [-0.25, 0.75]

    // --- Call split_blocks with coordinates that should cause a split ---
    // Choose desired positions 'x' such that the Lagrange Multiplier for c0 becomes negative.
    // Based on previous analysis, if desired x = [10.0, 10.0], lm(c0) should be negative.
    let x_split = [0.0, 10.0];
    let nosplit_result = cg.split_blocks(&x_split);

    assert!(!nosplit_result, "Expected split (false), but got true"); // Expecting SPLIT (false)

    // Optional: Add assertions here to verify the block state after splitting, if possible
    // e.g., assert that variables 0 and 1 are now in different blocks.
    // This requires adding methods to ConstraintGraph to inspect internal state for testing.
}
