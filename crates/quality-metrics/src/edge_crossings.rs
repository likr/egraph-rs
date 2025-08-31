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

    if s2 * t2 > S::zero() {
        return false;
    }

    true
}

pub type CrossingEdges<S> = Vec<(S, S, S, S, S, S, S, S)>;

/// Identifies all edge crossings in a graph layout in Euclidean 2D space.
///
/// This function examines all pairs of edges in the graph and determines which
/// ones intersect in the 2D drawing. Edges sharing a common endpoint are not
/// considered to be crossing.
///
/// # Parameters
///
/// * `graph`: The graph structure to evaluate
/// * `drawing`: The 2D Euclidean layout of the graph
///
/// # Returns
///
/// A `CrossingEdges` collection containing the coordinates of all crossing edge pairs.
///
/// # Type Parameters
///
/// * `G`: A graph type that implements the required traits
pub fn crossing_edges<G, S>(
    graph: G,
    drawing: &DrawingEuclidean2d<G::NodeId, S>,
) -> CrossingEdges<S>
where
    G: IntoEdgeReferences,
    G::NodeId: DrawingIndex,
    S: DrawingValue,
{
    let mut edges = vec![];
    for e in graph.edge_references() {
        let u = e.source();
        let v = e.target();
        for &(p, q) in drawing.edge_segments(u, v).unwrap().iter() {
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
    crossing_edges
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
    let mut edges = vec![];
    for e in graph.edge_references() {
        let u = e.source();
        let v = e.target();
        for &(p, q) in drawing.edge_segments(u, v).unwrap().iter() {
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
    crossing_edges
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
