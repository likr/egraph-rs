use super::*;
use ndarray::arr2;
use petgraph::graph::{NodeIndex, UnGraph};

#[test]
fn test_full_distance_matrix_from_and_as_array() {
    let mut g = UnGraph::<(), ()>::new_undirected();
    let n0 = g.add_node(());
    let _n1 = g.add_node(());
    let n2 = g.add_node(());

    let arr = arr2(&[[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]]);
    let dm: FullDistanceMatrix<NodeIndex, f64> = FullDistanceMatrix::from_array(&g, arr.clone());
    assert_eq!(dm.shape(), (3, 3));
    assert_eq!(dm.as_array(), &arr);
    assert_eq!(dm.get(n0, n2), Some(2.0));
    assert_eq!(dm.get_by_index(1, 2), 1.5);
}

#[test]
#[should_panic(expected = "FullDistanceMatrix::from_array")]
fn test_full_distance_matrix_from_array_bad_shape() {
    let mut g = UnGraph::<(), ()>::new_undirected();
    g.add_node(());
    g.add_node(());
    let arr = arr2(&[[0.0, 1.0, 2.0], [1.0, 0.0, 1.5]]);
    let _: FullDistanceMatrix<NodeIndex, f64> = FullDistanceMatrix::from_array(&g, arr);
}

#[test]
fn test_pivoted_distance_matrix_from_and_as_array() {
    let mut g = UnGraph::<(), ()>::new_undirected();
    let n0 = g.add_node(());
    let n1 = g.add_node(());
    let n2 = g.add_node(());

    let pivots = vec![n2, n0];
    let arr = arr2(&[[2.0, 1.5, 0.0], [0.0, 1.0, 2.0]]);
    let dm: PivotedDistanceMatrix<NodeIndex, f64> =
        PivotedDistanceMatrix::from_array(&g, pivots.clone(), arr.clone());
    assert_eq!(dm.shape(), (2, 3));
    assert_eq!(dm.pivots(), &pivots);
    assert_eq!(dm.as_array(), &arr);
    assert_eq!(dm.get(n2, n1), Some(1.5));
    assert_eq!(dm.get(n0, n2), Some(2.0));
    assert_eq!(dm.get(n1, n0), None);
    assert_eq!(dm.get_by_index(0, 1), 1.5);
}

#[test]
#[should_panic(expected = "PivotedDistanceMatrix::from_array")]
fn test_pivoted_distance_matrix_from_array_bad_shape() {
    let mut g = UnGraph::<(), ()>::new_undirected();
    let n0 = g.add_node(());
    let n1 = g.add_node(());
    let _n2 = g.add_node(());
    let arr = arr2(&[[0.0, 1.0], [1.0, 0.0], [2.0, 1.5]]);
    let _: PivotedDistanceMatrix<NodeIndex, f64> =
        PivotedDistanceMatrix::from_array(&g, vec![n0, n1], arr);
}
