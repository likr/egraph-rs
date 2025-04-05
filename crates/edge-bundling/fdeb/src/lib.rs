//! Force-Directed Edge Bundling (FDEB) implementation for petgraph.
//!
//! This crate provides an implementation of Force-Directed Edge Bundling algorithm
//! for graph visualization. Edge bundling helps reduce visual clutter in dense graphs
//! by grouping edges that follow similar paths.
//!
//! The algorithm is based on Holten, D., & Van Wijk, J. J. (2009). Force‐directed
//! edge bundling for graph visualization. Computer Graphics Forum, 28(3), 983-990.
//!
//! # Example
//!
//! ```
//! use petgraph::Graph;
//! use petgraph_drawing::{Drawing, DrawingEuclidean2d, MetricEuclidean2d};
//! use petgraph_edge_bundling_fdeb::{EdgeBundlingOptions, fdeb};
//!
//! // Create a graph and add nodes and edges
//! let mut graph = Graph::<(), ()>::new();
//! let n1 = graph.add_node(());
//! let n2 = graph.add_node(());
//! let n3 = graph.add_node(());
//! let e1 = graph.add_edge(n1, n2, ());
//! let e2 = graph.add_edge(n2, n3, ());
//!
//! // Create a drawing with node positions
//! let mut drawing = DrawingEuclidean2d::new(&graph);
//!
//! // Set node positions using position_mut
//! if let Some(pos) = drawing.position_mut(n1) {
//!     *pos = MetricEuclidean2d(0.0, 0.0);
//! }
//! if let Some(pos) = drawing.position_mut(n2) {
//!     *pos = MetricEuclidean2d(1.0, 1.0);
//! }
//! if let Some(pos) = drawing.position_mut(n3) {
//!     *pos = MetricEuclidean2d(2.0, 0.0);
//! }
//!
//! // Alternatively, you can use set_x and set_y methods
//! // drawing.set_x(n1, 0.0);
//! // drawing.set_y(n1, 0.0);
//! // drawing.set_x(n2, 1.0);
//! // drawing.set_y(n2, 1.0);
//! // drawing.set_x(n3, 2.0);
//! // drawing.set_y(n3, 0.0);
//!
//! // Apply edge bundling with default options
//! let options = EdgeBundlingOptions::new();
//! let bundled_edges = fdeb(&graph, &drawing, &options);
//!
//! // bundled_edges now contains control points for rendering curved edges
//! ```

use petgraph::visit::{EdgeRef, IntoEdgeReferences, IntoNodeIdentifiers};
use petgraph_drawing::{
    Drawing, DrawingEuclidean2d, DrawingIndex, DrawingValue, MetricEuclidean2d,
};
use std::{collections::HashMap, f32, hash::Hash};

/// A 2D point with position and velocity.
///
/// This structure represents a point in 2D space with its position coordinates
/// and velocity components. It's used to represent the control points of edges
/// during the force-directed edge bundling process.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Point {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
}

impl Point {
    /// Creates a new point at position (x, y) with zero initial velocity.
    ///
    /// # Arguments
    ///
    /// * `x` - The x-coordinate of the point
    /// * `y` - The y-coordinate of the point
    ///
    /// # Returns
    ///
    /// A new `Point` instance with the specified position and zero velocity
    pub fn new(x: f32, y: f32) -> Point {
        Point {
            x,
            y,
            vx: 0.,
            vy: 0.,
        }
    }
}

/// A line segment representing an edge in the graph.
///
/// Each line segment consists of a source and target node, as well as
/// a collection of intermediate control points that are used to bend the edge.
pub struct LineSegment {
    source: usize,
    target: usize,
    point_indices: Vec<usize>,
}

impl LineSegment {
    fn new(source: usize, target: usize) -> LineSegment {
        LineSegment {
            source,
            target,
            point_indices: Vec::new(),
        }
    }
}

struct EdgePair {
    p: usize,
    q: usize,
    compatibility: f32,
    theta: f32,
}

impl EdgePair {
    fn new(p: usize, q: usize, compatibility: f32, theta: f32) -> EdgePair {
        EdgePair {
            p,
            q,
            compatibility,
            theta,
        }
    }
}

fn distance(p1x: f32, p1y: f32, p2x: f32, p2y: f32) -> f32 {
    let dx = p2x - p1x;
    let dy = p2y - p1y;
    (dx * dx + dy * dy).sqrt().max(1e-6)
}

fn angle(p1: Point, p2: Point, q1: Point, q2: Point) -> f32 {
    let p_norm = distance(p1.x, p1.y, p2.x, p2.y);
    let q_norm = distance(q1.x, q1.y, q2.x, q2.y);
    let pq = (p2.x - p1.x) * (q2.x - q1.x) + (p2.y - p1.y) * (q2.y - q1.y);
    (pq / p_norm / q_norm).acos()
}

fn compatibility(p1: Point, p2: Point, q1: Point, q2: Point) -> f32 {
    let p_norm = distance(p1.x, p1.y, p2.x, p2.y);
    let q_norm = distance(q1.x, q1.y, q2.x, q2.y);
    let l_avg = (p_norm + q_norm) / 2.;
    let pmx = (p1.x + p2.x) / 2.;
    let pmy = (p1.y + p2.y) / 2.;
    let qmx = (q1.x + q2.x) / 2.;
    let qmy = (q1.y + q2.y) / 2.;
    let c_a = {
        let pq = (p2.x - p1.x) * (q2.x - q1.x) + (p2.y - p1.y) * (q2.y - q1.y);
        (pq / p_norm / q_norm).abs()
    };
    let c_s = 2. / (l_avg / p_norm.min(q_norm) + p_norm.max(q_norm) / l_avg);
    let c_p = {
        let mpq2 = distance(pmx, pmy, qmx, qmy);
        l_avg / (l_avg + mpq2)
    };
    let c_v = {
        let vp = {
            let i0r =
                ((p1.y - q1.y) * (p1.y - p2.y) - (p1.x - q1.x) * (p2.x - p1.x)) / p_norm / p_norm;
            let i0x = p1.x + i0r * (p2.x - p1.x);
            let i0y = p1.y + i0r * (p2.y - p1.y);
            let i1r =
                ((p1.y - q2.y) * (p1.y - p2.y) - (p1.x - q2.x) * (p2.x - p1.x)) / p_norm / p_norm;
            let i1x = p1.x + i1r * (p2.x - p1.x);
            let i1y = p1.y + i1r * (p2.y - p1.y);
            let imx = (i0x + i1x) / 2.;
            let imy = (i0y + i1y) / 2.;
            (1. - 2. * distance(pmx, pmy, imx, imy) / distance(i0x, i0y, i1x, i1y)).max(0.0)
        };
        let vq = {
            let i0r =
                ((q1.y - p1.y) * (q1.y - q2.y) - (q1.x - p1.x) * (q2.x - q1.x)) / q_norm / q_norm;
            let i0x = q1.x + i0r * (q2.x - q1.x);
            let i0y = q1.y + i0r * (q2.y - q1.y);
            let i1r =
                ((q1.y - p2.y) * (q1.y - q2.y) - (q1.x - p2.x) * (q2.x - q1.x)) / q_norm / q_norm;
            let i1x = q1.x + i1r * (q2.x - q1.x);
            let i1y = q1.y + i1r * (q2.y - q1.y);
            let imx = (i0x + i1x) / 2.;
            let imy = (i0y + i1y) / 2.;
            (1. - 2. * distance(qmx, qmy, imx, imy) / distance(i0x, i0y, i1x, i1y)).max(0.0)
        };
        vp.min(vq)
    };
    c_a * c_s * c_p * c_v
}

fn apply_spring_force(
    mid_points: &mut [Point],
    segments: &[LineSegment],
    points: &[Point],
    num_p: usize,
    k: f32,
) {
    for segment in segments.iter() {
        let d = distance(
            points[segment.source].x,
            points[segment.source].y,
            points[segment.target].x,
            points[segment.target].y,
        );
        let kp = k / (num_p as f32) / d;
        let n = segment.point_indices.len();
        for i in 0..n {
            let p0 = if i == 0 {
                points[segment.source]
            } else {
                mid_points[segment.point_indices[i - 1]]
            };
            let p2 = if i == n - 1 {
                points[segment.target]
            } else {
                mid_points[segment.point_indices[i + 1]]
            };
            let p1 = &mut mid_points[segment.point_indices[i]];
            p1.vx += kp * (p0.x - p1.x + p2.x - p1.x);
            p1.vy += kp * (p0.y - p1.y + p2.y - p1.y);
        }
    }
}

fn apply_electrostatic_force(
    mid_points: &mut [Point],
    segments: &[LineSegment],
    edge_pairs: &Vec<EdgePair>,
    num_p: usize,
) {
    for pair in edge_pairs {
        let EdgePair {
            p,
            q,
            theta,
            compatibility: c_e,
        } = pair;
        let segment_p = &segments[*p];
        let segment_q = &segments[*q];
        for i in 0..num_p {
            let j = if *theta < f32::consts::PI / 2.0 {
                i
            } else {
                num_p - i - 1
            };
            let pi = mid_points[segment_p.point_indices[i]];
            let qi = mid_points[segment_q.point_indices[j]];
            let dx = qi.x - pi.x;
            let dy = qi.y - pi.y;
            if dx.abs() > 1e-6 || dy.abs() > 1e-6 {
                let w = c_e / (dx * dx + dy * dy).sqrt();
                {
                    let qi = &mut mid_points[segment_q.point_indices[j]];
                    qi.vx -= dx * w;
                    qi.vy -= dy * w;
                }
                {
                    let pi = &mut mid_points[segment_p.point_indices[i]];
                    pi.vx += dx * w;
                    pi.vy += dy * w;
                }
            }
        }
    }
}

/// Configuration options for the Force-Directed Edge Bundling algorithm.
///
/// This structure contains parameters that control various aspects of the
/// edge bundling process, such as the number of iterations, step sizes,
/// and compatibility thresholds.
pub struct EdgeBundlingOptions<S> {
    /// Number of subdivision cycles
    cycles: usize,
    /// Initial step size for control point movement
    s0: S,
    /// Initial number of iterations per cycle
    i0: usize,
    /// Step size decrease factor between cycles
    s_step: S,
    /// Iteration count decrease factor between cycles
    i_step: S,
    /// Minimum compatibility threshold for edge interaction
    minimum_edge_compatibility: S,
}

impl<S> Default for EdgeBundlingOptions<S>
where
    S: DrawingValue,
{
    /// Returns a new configuration with default values.
    fn default() -> Self {
        Self::new()
    }
}

impl<S> EdgeBundlingOptions<S> {
    /// Creates a new configuration with default values.
    ///
    /// The default configuration is suitable for most graph visualization scenarios
    /// but may need adjustment for specific cases depending on graph size and density.
    ///
    /// # Returns
    ///
    /// A new `EdgeBundlingOptions` instance with the following default values:
    /// - cycles: 6
    /// - s0: 0.1
    /// - i0: 90
    /// - s_step: 0.5
    /// - i_step: 2/3
    /// - minimum_edge_compatibility: 0.6
    pub fn new() -> Self
    where
        S: DrawingValue,
    {
        EdgeBundlingOptions {
            cycles: 6,
            s0: S::from(0.1).unwrap(),
            i0: 90,
            s_step: S::from(0.5).unwrap(),
            i_step: S::from(2. / 3.).unwrap(),
            minimum_edge_compatibility: S::from(0.6).unwrap(),
        }
    }
}

/// Applies Force-Directed Edge Bundling to a graph.
///
/// This function implements the FDEB algorithm to create smoothly bundled edges
/// for a graph visualization. It works by iteratively applying forces to control points
/// that subdivide each edge, gradually pulling similar edges together.
///
/// # Arguments
///
/// * `graph` - The input graph
/// * `drawing` - The 2D Euclidean drawing containing node positions
/// * `options` - Configuration options for the edge bundling algorithm
///
/// # Returns
///
/// A HashMap mapping each edge ID to a vector of control points (x, y coordinates)
/// that represent the bundled path of the edge. These control points can be used
/// to render the edges as smooth curves.
///
/// # Type Parameters
///
/// * `G` - The graph type, which must support iterating over nodes and edges
/// * `G::NodeId` - The node identifier type, which must be usable as a drawing index
/// * `G::EdgeId` - The edge identifier type, which must be hashable and comparable
pub fn fdeb<G>(
    graph: G,
    drawing: &DrawingEuclidean2d<G::NodeId, f32>,
    options: &EdgeBundlingOptions<f32>,
) -> HashMap<G::EdgeId, Vec<(f32, f32)>>
where
    G: IntoNodeIdentifiers + IntoEdgeReferences,
    G::NodeId: DrawingIndex,
    G::EdgeId: Eq + Hash,
{
    let EdgeBundlingOptions {
        cycles,
        s0,
        i0,
        s_step,
        i_step,
        minimum_edge_compatibility,
    } = options;
    let points = graph
        .node_identifiers()
        .map(|u| {
            let MetricEuclidean2d(x, y) = drawing.position(u).unwrap();
            Point::new(*x, *y)
        })
        .collect::<Vec<Point>>();
    let node_indices = graph
        .node_identifiers()
        .enumerate()
        .map(|(i, u)| (u, i))
        .collect::<HashMap<G::NodeId, usize>>();
    let mut mid_points = Vec::new();
    let mut segments = graph
        .edge_references()
        .map(|e| {
            let u = e.source();
            let v = e.target();
            LineSegment::new(node_indices[&u], node_indices[&v])
        })
        .collect::<Vec<_>>();

    let mut num_iter = *i0;
    let mut alpha = *s0;

    let edge_pairs = {
        let mut edge_pairs = Vec::new();
        let m = segments.len();
        for p in 0..m {
            let segment_p = &segments[p];
            for (q, _) in segments.iter().enumerate().take(m).skip(p + 1) {
                let segment_q = &segments[q];
                let c_e = compatibility(
                    points[segment_p.source],
                    points[segment_p.target],
                    points[segment_q.source],
                    points[segment_q.target],
                );
                if c_e >= *minimum_edge_compatibility {
                    let theta = angle(
                        points[segment_p.source],
                        points[segment_p.target],
                        points[segment_q.source],
                        points[segment_q.target],
                    );
                    edge_pairs.push(EdgePair::new(p, q, c_e, theta));
                }
            }
        }
        edge_pairs
    };

    for cycle in 0..*cycles {
        let dp = 2_i32.pow(cycle as u32);
        for segment in segments.iter_mut() {
            for j in 0..dp {
                let p0 = if j == 0 {
                    points[segment.source]
                } else {
                    mid_points[segment.point_indices[(j * 2 - 1) as usize]]
                };
                let p1 = if j == dp - 1 {
                    points[segment.target]
                } else {
                    mid_points[segment.point_indices[(j * 2) as usize]]
                };
                mid_points.push(Point::new((p0.x + p1.x) / 2., (p0.y + p1.y) / 2.));
                segment
                    .point_indices
                    .insert((j * 2) as usize, mid_points.len() - 1);
            }
        }

        let num_p = (dp * 2 - 1) as usize;
        for _ in 0..num_iter {
            for point in mid_points.iter_mut() {
                point.vx = 0.;
                point.vy = 0.;
            }

            apply_spring_force(&mut mid_points, &segments, &points, num_p, 0.1);
            apply_electrostatic_force(&mut mid_points, &segments, &edge_pairs, num_p);

            for point in mid_points.iter_mut() {
                point.x += alpha * point.vx;
                point.y += alpha * point.vy;
            }
        }

        alpha *= s_step;
        num_iter = (num_iter as f32 * i_step) as usize;
    }

    segments
        .iter()
        .zip(graph.edge_references())
        .map(|(segment, e)| {
            let mut ps = vec![];
            let p0 = points[segment.source];
            ps.push((p0.x, p0.y));
            for &i in &segment.point_indices {
                let p = mid_points[i];
                ps.push((p.x, p.y));
            }
            let p1 = points[segment.target];
            ps.push((p1.x, p1.y));
            (e.id(), ps)
        })
        .collect()
}
