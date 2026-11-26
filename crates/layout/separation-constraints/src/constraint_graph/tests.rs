use super::*;
use petgraph::graph::{Graph, NodeIndex, UnGraph};
use petgraph_drawing::DrawingEuclidean;

// Helper to create a ConstraintGraph for testing (1D assumed).
fn create_test_graph(
    n: usize,
    initial_positions: &[f32],
    constraints_data: &[(usize, usize, f32)],
) -> ConstraintGraph<f32> {
    let mut graph: UnGraph<(), ()> = Graph::new_undirected();
    // Add nodes to the petgraph Graph (weights are not relevant here).
    let nodes: Vec<NodeIndex<u32>> = (0..n).map(|_| graph.add_node(())).collect();

    // Create a 1D drawing for simplicity in these tests.
    let mut drawing = DrawingEuclidean::new(&graph, 1);
    for i in 0..n {
        drawing.set(nodes[i], 0, initial_positions[i]);
    }

    let constraints = constraints_data
        .iter()
        .map(|&(l, r, g)| Constraint::new(l, r, g))
        .collect::<Vec<_>>();

    ConstraintGraph::new(&drawing, 0, &constraints)
}

#[test]
fn test_variable_position() {
    let cg = create_test_graph(3, &[0.0, 5.0, 15.0], &[]);
    // Test initial variable positions (offset 0, block pos = initial pos).
    assert_eq!(cg.variable_position(0), 0.0);
    assert_eq!(cg.variable_position(1), 5.0);
    assert_eq!(cg.variable_position(2), 15.0);
}

#[test]
fn test_violation() {
    let constraints_data = [(0, 1, 6.0), (1, 2, 8.0)]; // c0: 0+6<=1, c1: 1+8<=2
    let cg = create_test_graph(3, &[0.0, 5.0, 15.0], &constraints_data);

    // Test violation calc: c0: 0 + 6 - 5 = 1.0 (> 0, violated).
    assert!((cg.violation(0) - 1.0).abs() < 1e-6);
    // Test violation calc: c1: 5 + 8 - 15 = -2.0 (<= 0, not violated).
    assert!((cg.violation(1) - (-2.0)).abs() < 1e-6);
}

#[test]
fn test_optimal_position_single_var() {
    let cg = create_test_graph(1, &[10.0], &[]);
    let desired_x = [12.0];
    // Test optimal position for a block with a single variable (should be desired pos - offset).
    // Initial offset is 0.
    assert!((cg.optimal_position(0, &desired_x) - 12.0).abs() < 1e-6);
}

#[test]
fn test_optimal_position_multi_var() {
    // Setup: Block 0: {var 0 (off=-1), var 1 (off=2)}, Block 2: {var 2 (off=0)}
    let mut cg = create_test_graph(3, &[0.0, 1.0, 10.0], &[]); // Initial positions irrelevant here
    cg.variables[1].block = 0;
    cg.blocks[0].variables.insert(1);
    cg.blocks[1].variables.clear();
    cg.variables[0].offset = -1.0;
    cg.variables[1].offset = 2.0;
    // variables[2].offset remains 0

    let desired_x = [5.0, 8.0, 20.0]; // Desired positions for vars 0, 1, 2

    // Test optimal position for multi-variable block 0.
    // Expected = avg( (desired_x[j] - offset[j]) for j in block )
    // Expected = ( (5.0 - (-1.0)) + (8.0 - 2.0) ) / 2 = (6.0 + 6.0) / 2 = 6.0
    assert!((cg.optimal_position(0, &desired_x) - 6.0).abs() < 1e-6);

    // Test optimal position for single-variable block 2.
    // Expected = (desired_x[2] - offset[2]) / 1 = (20.0 - 0.0) / 1 = 20.0
    assert!((cg.optimal_position(2, &desired_x) - 20.0).abs() < 1e-6);
}

#[test]
fn test_merge_block() {
    let constraints_data = [(0, 1, 5.0)]; // c0: 0 + 5 <= 1
    let mut cg = create_test_graph(3, &[0.0, 10.0, 20.0], &constraints_data); // Initial blocks: {0}, {1}, {2}

    // --- Setup ---
    // Simulate merging block 1 (r=1) into block 0 (l=0) due to constraint c0 (index 0).
    // Initial offsets are 0. Initial positions: pos(0)=0, pos(1)=10.

    // --- Action ---
    cg.merge_block(0, 1, 0);

    // --- Assertions ---
    // Block 1 is now empty.
    assert!(cg.blocks[1].variables.is_empty());
    assert!(cg.blocks[1].active.is_empty());

    // Block 0 contains vars 0 and 1.
    assert_eq!(cg.blocks[0].variables, [0, 1].iter().cloned().collect());
    // Variable 2 remains in block 2.
    assert_eq!(cg.blocks[2].variables, [2].iter().cloned().collect());

    // Variables 0 and 1 now belong to block 0.
    assert_eq!(cg.variables[0].block, 0);
    assert_eq!(cg.variables[1].block, 0);
    assert_eq!(cg.variables[2].block, 2); // Var 2 unaffected.

    // Check offsets relative to block 0's new reference.
    // d = offset_left + gap - offset_right = 0 + 5.0 - 0 = 5.0
    // offset[0] (originally in l) remains 0.
    // offset[1] (originally in r) becomes d + old_offset[1] = 5.0 + 0 = 5.0.
    assert!((cg.variables[0].offset - 0.0).abs() < 1e-6);
    assert!((cg.variables[1].offset - 5.0).abs() < 1e-6);
    assert!((cg.variables[2].offset - 0.0).abs() < 1e-6); // Var 2 unaffected.

    // Check block 0's new position (weighted average).
    // pos_l=0, pos_r=10, nvar_l=1, nvar_r=1, d=5.0
    // new_pos = (0*1 + (10-5)*1) / (1+1) = 5.0 / 2 = 2.5
    assert!((cg.blocks[0].position - 2.5).abs() < 1e-6);

    // Check block 0's active set contains only the merging constraint c0.
    assert_eq!(cg.blocks[0].active, [0].iter().cloned().collect());
}

#[test]
fn test_connected() {
    // Graph: 0 --c0-- 1 --c1-- 2, plus 0 --c2-- 2. Node 3 is isolated.
    let constraints_data = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]; // c0, c1, c2
    let cg = create_test_graph(4, &[0., 1., 2., 3.], &constraints_data);

    // Scenario 1: Active constraints {c0, c2}. Expected connected from 0: {0, 1, 2}.
    let ac1: HashSet<usize> = [0, 2].iter().cloned().collect();
    let connected1 = cg.connected(0, &ac1);
    let expected1: HashSet<usize> = [0, 1, 2].iter().cloned().collect();
    assert_eq!(connected1, expected1);

    // Scenario 2: Active constraint {c1}. Expected connected from 1: {1, 2}.
    let ac2: HashSet<usize> = [1].iter().cloned().collect();
    let connected2 = cg.connected(1, &ac2);
    let expected2: HashSet<usize> = [1, 2].iter().cloned().collect();
    assert_eq!(connected2, expected2);

    // Scenario 3: No active constraints. Expected connected from 0: {0}.
    let ac3: HashSet<usize> = HashSet::new();
    let connected3 = cg.connected(0, &ac3);
    let expected3: HashSet<usize> = [0].iter().cloned().collect();
    assert_eq!(connected3, expected3);

    // Scenario 4: Start from isolated node 3. Expected: {3}.
    let connected4 = cg.connected(3, &ac1); // Use any active set
    let expected4: HashSet<usize> = [3].iter().cloned().collect();
    assert_eq!(connected4, expected4);
}

#[test]
fn test_comp_path() {
    // Graph: 0-c0-1-c1-2-c3-4-c5-5, also 1-c2-3-c4-4
    let constraints_data = [
        (0, 1, 1.0), // c0
        (1, 2, 1.0), // c1
        (1, 3, 1.0), // c2
        (2, 4, 1.0), // c3
        (3, 4, 1.0), // c4
        (4, 5, 1.0), // c5
    ];
    let cg = create_test_graph(6, &[0., 1., 2., 2., 3., 4.], &constraints_data);

    // Test path finding with different active sets.
    // Scenario 1: Path 0->1->3->4->5 is active.
    let ac1: HashSet<usize> = [0, 2, 4, 5].iter().cloned().collect();
    assert_eq!(cg.comp_path(0, 5, &ac1), vec![0, 1, 3, 4, 5]);

    // Scenario 2: Path 0->1->2->4->5 is active.
    let ac2: HashSet<usize> = [0, 1, 3, 5].iter().cloned().collect();
    assert_eq!(cg.comp_path(0, 5, &ac2), vec![0, 1, 2, 4, 5]);

    // Scenario 3: Multiple paths possible, BFS finds one. Find path 3 -> 5.
    // Active: c0, c1, c2, c3, c5. Path 3->1->... or 3->4->...
    // BFS likely finds 3->1->2->4->5 or 3->4->5 depending on neighbor order.
    let ac3: HashSet<usize> = [0, 1, 2, 3, 5].iter().cloned().collect();
    let path3 = cg.comp_path(3, 5, &ac3);
    assert!(
        path3 == vec![3, 1, 2, 4, 5] || path3 == vec![3, 4, 5],
        "Path 3->5 should be one of the valid BFS paths"
    );

    // Scenario 4: Target unreachable with active set {c0, c1}.
    let ac4: HashSet<usize> = [0, 1].iter().cloned().collect();
    assert!(
        cg.comp_path(0, 5, &ac4).is_empty(),
        "Path 0->5 should be empty"
    );
}

#[test]
fn test_comp_dfdv() {
    // Setup: Block 0 = {0, 1, 2}. Active: c0 (0->1, gap 1), c1 (1->2, gap 1).
    // Offsets based on active: off(0)=0, off(1)=1, off(2)=2. Block pos = 0.
    let constraints_data = [(0, 1, 1.0), (1, 2, 1.0)]; // c0, c1
    let mut cg = create_test_graph(3, &[0., 1., 2.], &constraints_data); // Initial pos match active state

    // Form block 0
    cg.variables[1].block = 0;
    cg.variables[2].block = 0;
    cg.blocks[0].variables.insert(1);
    cg.blocks[0].variables.insert(2);
    cg.blocks[1].variables.clear();
    cg.blocks[2].variables.clear();
    // Set offsets consistent with active c0, c1
    cg.variables[0].offset = 0.0;
    cg.variables[1].offset = 1.0;
    cg.variables[2].offset = 2.0;
    cg.blocks[0].position = 0.0;
    cg.blocks[0].active.insert(0); // c0 active
    cg.blocks[0].active.insert(1); // c1 active

    // Desired positions causing some tension.
    let desired_x = [0.5, 1.2, 2.8];
    // current pos: 0.0, 1.0, 2.0

    let mut lm = HashMap::new();
    // Call comp_dfdv starting from root var 0.
    let _root_dfdv = cg.comp_dfdv(0, &cg.blocks[0].active, None, &mut lm, &desired_x);

    // Assert that multipliers were computed for both active constraints.
    assert!(
        lm.contains_key(&0),
        "Lagrange multiplier for c0 should exist"
    );
    assert!(
        lm.contains_key(&1),
        "Lagrange multiplier for c1 should exist"
    );
    assert_eq!(
        lm.len(),
        2,
        "Only multipliers for active constraints c0 and c1 should exist"
    );

    // Assert specific expected lm values based on recursive definition.
    // dfdv(2) = pos(2)-x(2) = 2.0 - 2.8 = -0.8
    // lm(c1) = dfdv(2) = -0.8 (Inserted when v=1=left(c1), value is result of comp_dfdv(2,...))
    // dfdv(1) = (pos(1)-x(1)) + lm(c1) = (1.0 - 1.2) + (-0.8) = -1.0
    // lm(c0) = dfdv(1) = -1.0 (Inserted when v=0=left(c0), value is result of comp_dfdv(1,...))
    assert!((lm[&0] - (-1.0)).abs() < 1e-6, "LM for c0 mismatch");
    assert!((lm[&1] - (-0.8)).abs() < 1e-6, "LM for c1 mismatch"); // Corrected expected value to -0.8
}

#[test]
fn test_expand_block() {
    // Setup: Block 0 = {0, 1, 2}. Active: c0 (0->1, gap 2), c1 (1->2, gap 2).
    // Offsets: off(0)=0, off(1)=2, off(2)=4. Block pos = 0. Actual pos: 0, 2, 4.
    // Violated constraint: c2 (0->2, gap 5). Violation = pos(0)+5-pos(2) = 0+5-4 = 1.0.
    let constraints_data = [(0, 1, 2.0), (1, 2, 2.0), (0, 2, 5.0)]; // c0, c1, c2
    let mut cg = create_test_graph(3, &[0., 2., 4.], &constraints_data); // Initial pos match active

    // Form block 0
    cg.variables[1].block = 0;
    cg.variables[2].block = 0;
    cg.blocks[0].variables.insert(1);
    cg.blocks[0].variables.insert(2);
    cg.blocks[1].variables.clear();
    cg.blocks[2].variables.clear();
    cg.variables[0].offset = 0.0;
    cg.variables[1].offset = 2.0;
    cg.variables[2].offset = 4.0;
    cg.blocks[0].position = 0.0;
    cg.blocks[0].active.insert(0); // c0 active
    cg.blocks[0].active.insert(1); // c1 active

    // Desired positions slightly perturbed to avoid lm tie.
    let desired_x = [0.0, 2.1, 4.0]; // Changed from [0., 2., 4.]
    let violating_c_idx = 2; // Index of constraint c2 (0->2, gap 5)

    // --- Pre-check ---
    assert!(
        (cg.violation(violating_c_idx) - 1.0).abs() < 1e-6,
        "Constraint c2 should be violated initially"
    );

    // --- Action ---
    // Expand block 0 to satisfy the internal violation of c2.
    cg.expand_block(0, violating_c_idx, &desired_x);

    // --- Assertions ---
    // 1. Determine the expected split constraint (sc).
    //    Path 0->2 is 0-c0->1-c1->2. Both c0, c1 are in ps.
    //    Compute lm: current=desired, so dfdv = 0 initially for all vars.
    //    dfdv(2)=0. lm(c1)=-dfdv(2)=0.
    //    dfdv(1)=dfdv(2)=0. lm(c0)=dfdv(1)=0.
    //    Min lm is 0. Tie broken by constraint index? Assume c0 (index 0) is chosen.
    let expected_sc = 0;

    // 2. Check the final active set. Should be {c1, c2} if sc=c0.
    let expected_active_set: HashSet<usize> = [1, 2].iter().cloned().collect();
    assert_eq!(
        cg.blocks[0].active, expected_active_set,
        "Active set should be {{c1, c2}} if c0 was split"
    );

    // 3. Check offset changes.
    //    If sc=c0, ac becomes {c1}. Component connected to right(c2)=2 is {1, 2} (via active c1).
    //    Offsets of {1, 2} should increase by violation(c2) = 1.0.
    //    New offset(1) = 2.0 + 1.0 = 3.0
    //    New offset(2) = 4.0 + 1.0 = 5.0
    assert!(
        (cg.variables[0].offset - 0.0).abs() < 1e-6,
        "Offset 0 should be unchanged"
    );
    assert!(
        (cg.variables[1].offset - 3.0).abs() < 1e-6,
        "Offset 1 should be 3.0"
    );
    assert!(
        (cg.variables[2].offset - 5.0).abs() < 1e-6,
        "Offset 2 should be 5.0"
    );

    // 4. Check block position recalculation.
    //    optimal_pos = ( (dx[0]-off[0]) + (dx[1]-off[1]) + (dx[2]-off[2]) ) / 3
    //              = ( (0.0-0.0) + (2.1-3.0) + (4.0-5.0) ) / 3
    //              = ( 0 - 0.9 - 1.0 ) / 3 = -1.9 / 3
    let expected_new_pos = -1.9 / 3.0; // Updated expected value
    assert!(
        (cg.blocks[0].position - expected_new_pos).abs() < 1e-6,
        "Block position recalculated incorrectly"
    );

    // 5. Check that the violation of c2 is now resolved (close to 0).
    //    New pos(0) = pos + off(0) = -1.9/3 + 0 = -1.9/3
    //    New pos(2) = pos + off(2) = -1.9/3 + 5 = (-1.9 + 15)/3 = 13.1/3
    //    New viol(c2) = pos(0) + gap(c2) - pos(2) = -1.9/3 + 5 - 13.1/3 = (-1.9 + 15 - 13.1)/3 = 0/3 = 0
    assert!(
        cg.violation(violating_c_idx).abs() < 1e-6, // Violation should still resolve to 0
        "Violation of c2 should be resolved"
    );

    // 6. Check that the split constraint sc=c0 now has slack (negative violation).
    //    New pos(1) = pos + off(1) = -1.9/3 + 3 = (-1.9 + 9)/3 = 7.1/3
    //    viol(c0) = pos(0) + gap(c0) - pos(1)
    //             = (-1.9/3) + 2 - (7.1/3)
    //             = (-1.9 + 6 - 7.1) / 3 = -3.0 / 3 = -1.0
    // Negative violation means c0 holds with slack, as expected after splitting it.
    assert!(
        cg.violation(expected_sc) < -1e-6, // Should still be negative
        "Split constraint c0 should now have slack"
    );
}
