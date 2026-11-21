use crate::edge_angle::edge_angle;
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use petgraph_drawing::{
    DrawingEuclidean2d, DrawingIndex, DrawingTorus2d, DrawingValue, MetricEuclidean2d,
};

/// Represents a collection of crossing edges in a graph layout.
///
/// Each entry in the vector is a tuple containing the coordinates of two line segments that cross:
/// (x11, y11, x12, y12, x21, y21, x22, y22), where:
/// - (x11, y11) and (x12, y12) are the endpoints of the first line segment
/// - (x21, y21) and (x22, y22) are the endpoints of the second line segment

#[derive(Clone, Copy)]
struct Line<S> {
    x1: S,
    y1: S,
    x2: S,
    y2: S,
}

/// Determines whether two line segments intersect.
///
/// This function uses a fast line segment intersection test based on
/// checking if the endpoints of each line are on opposite sides of the other line.
///
/// # Parameters
///
/// * `line1`: The first line segment
/// * `line2`: The second line segment
///
/// # Returns
///
/// `true` if the line segments intersect, `false` otherwise
fn cross<S: DrawingValue>(line1: &Line<S>, line2: &Line<S>) -> bool {
    if !line1.x1.is_finite()
        || !line1.y1.is_finite()
        || !line1.x2.is_finite()
        || !line1.y2.is_finite()
        || !line2.x1.is_finite()
        || !line2.y1.is_finite()
        || !line2.x2.is_finite()
        || !line2.y2.is_finite()
    {
        return false;
    }

    // Reject disjoint bounding boxes first so non-finite or collinear disjoint segments
    // never report false-positive crossings.
    let min_x1 = line1.x1.min(line1.x2);
    let max_x1 = line1.x1.max(line1.x2);
    let min_y1 = line1.y1.min(line1.y2);
    let max_y1 = line1.y1.max(line1.y2);
    let min_x2 = line2.x1.min(line2.x2);
    let max_x2 = line2.x1.max(line2.x2);
    let min_y2 = line2.y1.min(line2.y2);
    let max_y2 = line2.y1.max(line2.y2);
    if max_x1 < min_x2 || max_x2 < min_x1 || max_y1 < min_y2 || max_y2 < min_y1 {
        return false;
    }

    let dx1 = line1.x2 - line1.x1;
    let dy1 = line1.y2 - line1.y1;
    let dx2 = line2.x2 - line2.x1;
    let dy2 = line2.y2 - line2.y1;

    let s1 = dx1 * (line2.y1 - line1.y1) - dy1 * (line2.x1 - line1.x1);
    let t1 = dx1 * (line2.y2 - line1.y1) - dy1 * (line2.x2 - line1.x1);

    if s1 * t1 > S::zero() {
        return false;
    }

    let s2 = dx2 * (line1.y1 - line2.y1) - dy2 * (line1.x1 - line2.x1);
    let t2 = dx2 * (line1.y2 - line2.y1) - dy2 * (line1.x2 - line2.x1);

    s2 * t2 <= S::zero()
}

pub type CrossingEdges<S> = Vec<(S, S, S, S, S, S, S, S)>;

/// Validating variant of [`crossing_edges`] returning an error when an edge endpoint is missing from `drawing`.
pub fn try_crossing_edges<G, S>(
    graph: G,
    drawing: &DrawingEuclidean2d<G::NodeId, S>,
) -> Result<CrossingEdges<S>, String>
where
    G: IntoEdgeReferences,
    G::NodeId: DrawingIndex,
    S: DrawingValue,
{
    let mut edges = vec![];
    for e in graph.edge_references() {
        let u = e.source();
        let v = e.target();
        let segs = drawing
            .edge_segments(u, v)
            .ok_or_else(|| "graph edge endpoint not found in drawing".to_string())?;
        for &(p, q) in segs.iter() {
            let MetricEuclidean2d(x1, y1) = p;
            let MetricEuclidean2d(x2, y2) = q;
            edges.push((u, v, x1, y1, x2, y2));
        }
    }
    let mut crossing_edges = vec![];
    let m = edges.len();
    for i in 1..m {
        let (source1, target1, x11, y11, x12, y12) = edges[i];
        for &(source2, target2, x21, y21, x22, y22) in edges.iter().take(i) {
            if source1 == source2
                || source1 == target1
                || source1 == target2
                || source2 == target1
                || source2 == target2
                || target1 == target2
            {
                continue;
            }
            let line1 = Line {
                x1: x11,
                y1: y11,
                x2: x12,
                y2: y12,
            };
            let line2 = Line {
                x1: x21,
                y1: y21,
                x2: x22,
                y2: y22,
            };
            if cross(&line1, &line2) {
                crossing_edges.push((x11, y11, x12, y12, x21, y21, x22, y22));
            }
        }
    }
    Ok(crossing_edges)
}

pub fn crossing_edges<G, S>(
    graph: G,
    drawing: &DrawingEuclidean2d<G::NodeId, S>,
) -> CrossingEdges<S>
where
    G: IntoEdgeReferences,
    G::NodeId: DrawingIndex,
    S: DrawingValue,
{
    try_crossing_edges(graph, drawing).unwrap_or_else(|e| panic!("{}", e))
}

/// Validating variant of [`crossing_edges_torus`] returning an error when an edge endpoint is missing from `drawing`.
pub fn try_crossing_edges_torus<G, S>(
    graph: G,
    drawing: &DrawingTorus2d<G::NodeId, S>,
) -> Result<CrossingEdges<S>, String>
where
    G: IntoEdgeReferences,
    G::NodeId: DrawingIndex,
    S: DrawingValue,
{
    let mut edges = vec![];
    for e in graph.edge_references() {
        let u = e.source();
        let v = e.target();
        let segs = drawing
            .edge_segments(u, v)
            .ok_or_else(|| "graph edge endpoint not found in drawing".to_string())?;
        for &(p, q) in segs.iter() {
            edges.push((u, v, p.0 .0, p.1 .0, q.0 .0, q.1 .0));
        }
    }
    let mut crossing_edges = vec![];
    let m = edges.len();
    for i in 1..m {
        let (source1, target1, x11, y11, x12, y12) = edges[i];
        for &(source2, target2, x21, y21, x22, y22) in edges.iter().take(i) {
            if source1 == source2
                || source1 == target1
                || source1 == target2
                || source2 == target1
                || source2 == target2
                || target1 == target2
            {
                continue;
            }
            let line1 = Line {
                x1: x11,
                y1: y11,
                x2: x12,
                y2: y12,
            };
            let line2 = Line {
                x1: x21,
                y1: y21,
                x2: x22,
                y2: y22,
            };
            if cross(&line1, &line2) {
                crossing_edges.push((x11, y11, x12, y12, x21, y21, x22, y22));
            }
        }
    }
    Ok(crossing_edges)
}

/// Identifies all edge crossings in a graph layout in torus 2D space.
///
/// Similar to `crossing_edges`, but for drawings on a torus (a space that wraps around
/// at the boundaries). This is useful for layouts that use periodic boundary conditions.
///
/// # Parameters
///
/// * `graph`: The graph structure to evaluate
/// * `drawing`: The 2D torus layout of the graph
///
/// # Returns
///
/// A `CrossingEdges` collection containing the coordinates of all crossing edge pairs.
///
/// # Type Parameters
///
/// * `G`: A graph type that implements the required traits
pub fn crossing_edges_torus<G, S>(
    graph: G,
    drawing: &DrawingTorus2d<G::NodeId, S>,
) -> CrossingEdges<S>
where
    G: IntoEdgeReferences,
    G::NodeId: DrawingIndex,
    S: DrawingValue,
{
    try_crossing_edges_torus(graph, drawing).unwrap_or_else(|e| panic!("{}", e))
}

/// Validating variant of [`crossing_number`].
pub fn try_crossing_number<G, S>(
    graph: G,
    drawing: &DrawingEuclidean2d<G::NodeId, S>,
) -> Result<S, String>
where
    G: IntoEdgeReferences,
    G::NodeId: DrawingIndex,
    S: DrawingValue,
{
    let crossing_edges = try_crossing_edges(graph, drawing)?;
    Ok(crossing_number_with_crossing_edges(&crossing_edges))
}

/// Validating variant of [`crossing_angle`].
pub fn try_crossing_angle<G, S>(
    graph: G,
    drawing: &DrawingEuclidean2d<G::NodeId, S>,
) -> Result<S, String>
where
    G: IntoEdgeReferences,
    G::NodeId: DrawingIndex,
    S: DrawingValue,
{
    let crossing_edges = try_crossing_edges(graph, drawing)?;
    Ok(crossing_angle_with_crossing_edges(&crossing_edges))
}

/// Calculates the crossing number for a graph layout in Euclidean 2D space.
///
/// The crossing number is simply the count of edge crossings in the layout.
/// A lower crossing number generally indicates a clearer, more readable layout.
///
/// # Parameters
///
/// * `graph`: The graph structure to evaluate
/// * `drawing`: The 2D Euclidean layout of the graph
///
/// # Returns
///
/// An `S` value representing the crossing number (count of edge crossings).
///
/// # Type Parameters
///
/// * `G`: A graph type that implements the required traits
pub fn crossing_number<G, S>(graph: G, drawing: &DrawingEuclidean2d<G::NodeId, S>) -> S
where
    G: IntoEdgeReferences,
    G::NodeId: DrawingIndex,
    S: DrawingValue,
{
    let crossing_edges = crossing_edges(graph, drawing);
    crossing_number_with_crossing_edges(&crossing_edges)
}

/// Calculates the crossing number from a pre-computed collection of crossing edges.
///
/// This function is useful when you have already computed the crossing edges
/// and want to get the crossing number without recalculating the crossings.
///
/// # Parameters
///
/// * `crossing_edges`: A pre-computed collection of crossing edges
///
/// # Returns
///
/// An `S` value representing the crossing number (count of edge crossings).
pub fn crossing_number_with_crossing_edges<S: DrawingValue>(
    crossing_edges: &CrossingEdges<S>,
) -> S {
    S::from_usize(crossing_edges.len()).unwrap()
}

/// Calculates the crossing angle metric for a graph layout in Euclidean 2D space.
///
/// The crossing angle metric evaluates the angles at which edges cross. When edges
/// must cross, it's preferable that they cross at angles close to 90 degrees for
/// better readability. This metric sums up a function of these angles, with higher
/// values indicating crossings at angles closer to 90 degrees.
///
/// # Parameters
///
/// * `graph`: The graph structure to evaluate
/// * `drawing`: The 2D Euclidean layout of the graph
///
/// # Returns
///
/// An `S` value representing the crossing angle metric. Higher values indicate
/// better crossing angles (closer to 90 degrees).
///
/// # Type Parameters
///
/// * `G`: A graph type that implements the required traits
pub fn crossing_angle<G, S>(graph: G, drawing: &DrawingEuclidean2d<G::NodeId, S>) -> S
where
    G: IntoEdgeReferences,
    G::NodeId: DrawingIndex,
    S: DrawingValue,
{
    let crossing_edges = crossing_edges(graph, drawing);
    crossing_angle_with_crossing_edges(&crossing_edges)
}

/// Calculates the crossing angle metric from a pre-computed collection of crossing edges.
///
/// This function is useful when you have already computed the crossing edges
/// and want to calculate the crossing angle metric without recalculating the crossings.
///
/// The metric is calculated as the sum of squared cosines of the angles between
/// crossing edges, adjusted to always use the smaller of the two possible angles.
///
/// # Parameters
///
/// * `crossing_edges`: A pre-computed collection of crossing edges
///
/// # Returns
///
/// An `S` value representing the crossing angle metric. Higher values indicate
/// better crossing angles (closer to 90 degrees).
pub fn crossing_angle_with_crossing_edges<S: DrawingValue>(crossing_edges: &CrossingEdges<S>) -> S {
    let mut s = S::zero();
    for &(x11, y11, x12, y12, x21, y21, x22, y22) in crossing_edges.iter() {
        if let Some(t) = edge_angle(x11 - x12, y11 - y12, x21 - x22, y21 - y22) {
            let t = t.min(S::PI() - t);
            s += t.cos().powi(2);
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph_drawing::Drawing;

    #[test]
    fn test_cross_rejects_nonfinite_and_collinear_disjoint() {
        let mut g = petgraph::graph::UnGraph::<(), ()>::new_undirected();
        let n0 = g.add_node(());
        let n1 = g.add_node(());
        let n2 = g.add_node(());
        let n3 = g.add_node(());
        g.add_edge(n0, n1, ());
        g.add_edge(n2, n3, ());

        let mut d_collinear = DrawingEuclidean2d::initial_placement(&g);
        *d_collinear.position_mut(n0).unwrap() = MetricEuclidean2d(0.0_f64, 0.0);
        *d_collinear.position_mut(n1).unwrap() = MetricEuclidean2d(1.0, 0.0);
        *d_collinear.position_mut(n2).unwrap() = MetricEuclidean2d(2.0, 0.0);
        *d_collinear.position_mut(n3).unwrap() = MetricEuclidean2d(3.0, 0.0);
        assert_eq!(crossing_number(&g, &d_collinear), 0.0);

        let mut d_nan = DrawingEuclidean2d::initial_placement(&g);
        *d_nan.position_mut(n0).unwrap() = MetricEuclidean2d(f64::NAN, 0.0);
        *d_nan.position_mut(n1).unwrap() = MetricEuclidean2d(1.0, 0.0);
        *d_nan.position_mut(n2).unwrap() = MetricEuclidean2d(2.0, 1.0);
        *d_nan.position_mut(n3).unwrap() = MetricEuclidean2d(3.0, 1.0);
        assert_eq!(crossing_number(&g, &d_nan), 0.0);

        let mut g2 = petgraph::graph::UnGraph::<(), ()>::new_undirected();
        g2.add_node(());
        g2.add_node(());
        let d2 = DrawingEuclidean2d::<_, f64>::initial_placement(&g2);
        assert!(try_crossing_number(&g, &d2).is_err());
    }
}
