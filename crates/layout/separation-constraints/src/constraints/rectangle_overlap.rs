use petgraph_drawing::{
    Delta, Drawing, DrawingEuclidean2d, DrawingIndex, DrawingValue, MetricCartesian,
};
use std::cmp::Ordering;
use std::collections::BTreeSet;

use crate::{project_1d, Constraint};

/// Checks if two 1-dimensional segments overlap
fn overlap_1d<S>(x00: S, x01: S, x10: S, x11: S) -> bool
where
    S: DrawingValue,
{
    (x00 < x11 && x10 < x01) || (x10 < x01 && x00 < x11)
}

/// Checks if two rectangles overlap by comparing their intervals in all dimensions
fn overlap<S>(a: &Vec<(S, S)>, b: &Vec<(S, S)>) -> bool
where
    S: DrawingValue,
{
    a.iter()
        .zip(b.iter())
        .all(|(&(x00, x01), &(x10, x11))| overlap_1d(x00, x01, x10, x11))
}

/// Represents a rectangle with its dimensions and node index
#[derive(Clone, Debug)]
struct Rectangle {
    /// Index of the node this rectangle represents
    index: usize,
    /// Minimum x-coordinate
    x_min: f32,
    /// Maximum x-coordinate
    x_max: f32,
    /// Minimum y-coordinate
    y_min: f32,
    /// Maximum y-coordinate
    y_max: f32,
}

/// Event type for the sweepline algorithm
#[derive(Debug, Clone)]
enum EventType {
    /// Start of a rectangle
    Open,
    /// End of a rectangle
    Close,
}

/// Event for the sweepline algorithm
#[derive(Debug, Clone)]
struct Event {
    /// Position of the event along the sweep dimension
    position: f32,
    /// Type of event (open or close)
    event_type: EventType,
    /// Index of the rectangle this event belongs to
    rect_index: usize,
}

/// Provides ordering for events
impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        // First order by position
        match self.position.partial_cmp(&other.position) {
            Some(Ordering::Equal) => {}
            Some(ordering) => return ordering,
            None => return Ordering::Equal,
        }

        // If positions are equal, Close events come before Open events
        match (&self.event_type, &other.event_type) {
            (EventType::Close, EventType::Open) => Ordering::Less,
            (EventType::Open, EventType::Close) => Ordering::Greater,
            _ => Ordering::Equal,
        }
    }
}

impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.position == other.position
            && std::mem::discriminant(&self.event_type) == std::mem::discriminant(&other.event_type)
            && self.rect_index == other.rect_index
    }
}

impl Eq for Event {}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::Graph;

    #[test]
    fn test_rectangle_no_overlap_constraints_2d() {
        // Create a graph with 4 nodes in a square formation
        let mut graph = Graph::<(), (), petgraph::Undirected>::new_undirected();
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());
        let n4 = graph.add_node(());

        // Create a drawing with the nodes positioned in a square
        let mut drawing = DrawingEuclidean2d::new(&graph);
        drawing.set_x(n1, 0.0);
        drawing.set_y(n1, 0.0);
        drawing.set_x(n2, 1.0);
        drawing.set_y(n2, 0.0);
        drawing.set_x(n3, 0.0);
        drawing.set_y(n3, 1.0);
        drawing.set_x(n4, 1.0);
        drawing.set_y(n4, 1.0);

        // Set node size
        let node_size = 0.5f32;

        // Generate constraints for x and y dimensions
        let constraints_x = generate_rectangle_no_overlap_constraints_x(&drawing, |_, _| node_size);
        let constraints_y = generate_rectangle_no_overlap_constraints_y(&drawing, |_, _| node_size);

        // We expect constraints between potentially overlapping rectangles
        // For a square of 4 nodes with this specific configuration,
        // the correct number of constraints should be 0 constraints
        // since the nodes are not close enough to overlap with the given size
        assert_eq!(constraints_x.len(), 0);
        assert_eq!(constraints_y.len(), 0);
    }

    #[test]
    fn test_project_rectangle_no_overlap_constraints_2d() {
        // Create a graph with 2 overlapping nodes
        let mut graph = Graph::<(), (), petgraph::Undirected>::new_undirected();
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());

        // Create a drawing with the nodes positioned with overlap
        let mut drawing = DrawingEuclidean2d::new(&graph);
        drawing.set_x(n1, 0.0);
        drawing.set_y(n1, 0.0);
        drawing.set_x(n2, 0.5); // Overlapping in X
        drawing.set_y(n2, 0.5); // Overlapping in Y

        // Set node size so they overlap (each node is 1.0 wide)
        let node_size = 1.0f32;

        // Save original positions
        let orig_x1 = drawing.x(n1).unwrap();
        let orig_y1 = drawing.y(n1).unwrap();
        let orig_x2 = drawing.x(n2).unwrap();
        let orig_y2 = drawing.y(n2).unwrap();

        // Apply constraints to remove overlaps
        project_rectangle_no_overlap_constraints_2d(&mut drawing, |_, _| node_size);

        // Check that positions have changed to resolve overlap
        assert!(
            drawing.x(n1).unwrap() != orig_x1
                || drawing.x(n2).unwrap() != orig_x2
                || drawing.y(n1).unwrap() != orig_y1
                || drawing.y(n2).unwrap() != orig_y2,
            "Rectangle positions should change to resolve overlap"
        );

        // Check that the nodes are no longer overlapping
        // With node size 1.0, centers should be at least 1.0 apart
        let dx = (drawing.x(n1).unwrap() - drawing.x(n2).unwrap()).abs() as f32;
        let dy = (drawing.y(n1).unwrap() - drawing.y(n2).unwrap()).abs() as f32;

        assert!(
            dx >= node_size || dy >= node_size,
            "Rectangles still overlap after projection: dx={}, dy={}",
            dx,
            dy
        );
    }
}

impl Ord for Rectangle {
    fn cmp(&self, other: &Self) -> Ordering {
        // First compare by X coordinate for X-based constraints
        self.x_min
            .partial_cmp(&other.x_min)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Rectangle {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Rectangle {
    fn eq(&self, other: &Self) -> bool {
        self.x_min == other.x_min && self.index == other.index
    }
}

impl Eq for Rectangle {}

/// Represents a rectangle for Y dimension ordering, ordered by y_min
#[derive(Clone, Debug)]
struct RectangleY {
    /// Index of the node this rectangle represents
    index: usize,
    /// Minimum y-coordinate
    y_min: f32,
    /// Maximum y-coordinate
    y_max: f32,
    /// Minimum x-coordinate
    x_min: f32,
    /// Maximum x-coordinate
    x_max: f32,
}

impl Ord for RectangleY {
    fn cmp(&self, other: &Self) -> Ordering {
        // First compare by Y coordinate for Y-based constraints
        self.y_min
            .partial_cmp(&other.y_min)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for RectangleY {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for RectangleY {
    fn eq(&self, other: &Self) -> bool {
        self.y_min == other.y_min && self.index == other.index
    }
}

impl Eq for RectangleY {}

/// Generates separation constraints to prevent rectangle overlaps in the X dimension.
///
/// This function uses a sweep line algorithm to efficiently find potential overlapping
/// rectangle pairs and generate appropriate X-dimension constraints. This implementation
/// is based on the WebCola algorithm but uses Rust's BTreeSet instead of an RBTree.
fn generate_rectangle_no_overlap_constraints_x<N, F>(
    drawing: &DrawingEuclidean2d<N, f32>,
    mut size: F,
) -> Vec<Constraint>
where
    N: DrawingIndex + Copy,
    F: FnMut(N, usize) -> f32,
{
    let n = drawing.len();
    let mut constraints = Vec::new();

    // Create rectangle objects with their dimensions
    let mut rectangles = Vec::with_capacity(n);
    for i in 0..n {
        let x = drawing.raw_entry(i).nth(0);
        let y = drawing.raw_entry(i).nth(1);
        let width = size(*drawing.node_id(i), 0);
        let height = size(*drawing.node_id(i), 1);

        rectangles.push(Rectangle {
            index: i,
            x_min: x - width / 2.0,
            x_max: x + width / 2.0,
            y_min: y - height / 2.0,
            y_max: y + height / 2.0,
        });
    }

    // Create events for the sweep line algorithm
    let mut events = Vec::with_capacity(n * 2);
    for (i, rect) in rectangles.iter().enumerate() {
        // Y coordinates are used as sweep positions for X-constraints
        events.push(Event {
            position: rect.y_min,
            event_type: EventType::Open,
            rect_index: i,
        });

        events.push(Event {
            position: rect.y_max,
            event_type: EventType::Close,
            rect_index: i,
        });
    }

    // Sort events by position
    events.sort();

    // Use BTreeSet to maintain active rectangles
    let mut active_rects: BTreeSet<Rectangle> = BTreeSet::new();

    // Process events in order
    for event in events {
        let rect_index = event.rect_index;
        let rect = &rectangles[rect_index];

        match event.event_type {
            EventType::Open => {
                // When a rectangle becomes active, check it against all other active rectangles
                for active_rect in active_rects.iter() {
                    let other_rect = &rectangles[active_rect.index];

                    // Check if they overlap in the Y dimension - already guaranteed by the sweep
                    // Check if they overlap in the X dimension
                    if overlap_1d(rect.x_min, rect.x_max, other_rect.x_min, other_rect.x_max) {
                        let i = rect.index;
                        let j = other_rect.index;

                        // Calculate the minimum separation needed
                        let gap =
                            (rect.x_max - rect.x_min + other_rect.x_max - other_rect.x_min) / 2.0;

                        // Create constraint ensuring they don't overlap in the X dimension
                        let i_center_x = (rect.x_min + rect.x_max) / 2.0;
                        let j_center_x = (other_rect.x_min + other_rect.x_max) / 2.0;

                        if i_center_x < j_center_x {
                            constraints.push(Constraint::new(i, j, gap));
                        } else {
                            constraints.push(Constraint::new(j, i, gap));
                        }
                    }
                }

                // Add current rectangle to active set
                active_rects.insert(rect.clone());
            }
            EventType::Close => {
                // Remove rectangle from active set
                let rect_to_remove = Rectangle {
                    index: rect_index,
                    ..(*rect).clone()
                };
                active_rects.remove(&rect_to_remove);
            }
        }
    }

    constraints
}

/// Generates separation constraints to prevent rectangle overlaps in the Y dimension.
///
/// This function uses a sweep line algorithm to efficiently find potential overlapping
/// rectangle pairs and generate appropriate Y-dimension constraints. This implementation
/// is based on the WebCola algorithm but uses Rust's BTreeSet instead of an RBTree.
fn generate_rectangle_no_overlap_constraints_y<N, F>(
    drawing: &DrawingEuclidean2d<N, f32>,
    mut size: F,
) -> Vec<Constraint>
where
    N: DrawingIndex + Copy,
    F: FnMut(N, usize) -> f32,
{
    let n = drawing.len();
    let mut constraints = Vec::new();

    // Create rectangle objects with their dimensions
    let mut rectangles = Vec::with_capacity(n);
    for i in 0..n {
        let x = drawing.raw_entry(i).nth(0);
        let y = drawing.raw_entry(i).nth(1);
        let width = size(*drawing.node_id(i), 0);
        let height = size(*drawing.node_id(i), 1);

        rectangles.push(RectangleY {
            index: i,
            x_min: x - width / 2.0,
            x_max: x + width / 2.0,
            y_min: y - height / 2.0,
            y_max: y + height / 2.0,
        });
    }

    // Create events for the sweep line algorithm
    let mut events = Vec::with_capacity(n * 2);
    for (i, rect) in rectangles.iter().enumerate() {
        // X coordinates are used as sweep positions for Y-constraints
        events.push(Event {
            position: rect.x_min,
            event_type: EventType::Open,
            rect_index: i,
        });

        events.push(Event {
            position: rect.x_max,
            event_type: EventType::Close,
            rect_index: i,
        });
    }

    // Sort events by position
    events.sort();

    // Use BTreeSet to maintain active rectangles
    let mut active_rects: BTreeSet<RectangleY> = BTreeSet::new();

    // Process events in order
    for event in events {
        let rect_index = event.rect_index;
        let rect = &rectangles[rect_index];

        match event.event_type {
            EventType::Open => {
                // When a rectangle becomes active, check it against all other active rectangles
                for active_rect in active_rects.iter() {
                    let other_rect = &rectangles[active_rect.index];

                    // Check if they overlap in the X dimension - already guaranteed by the sweep
                    // Check if they overlap in the Y dimension
                    if overlap_1d(rect.y_min, rect.y_max, other_rect.y_min, other_rect.y_max) {
                        let i = rect.index;
                        let j = other_rect.index;

                        // Calculate the minimum separation needed
                        let gap =
                            (rect.y_max - rect.y_min + other_rect.y_max - other_rect.y_min) / 2.0;

                        // Create constraint ensuring they don't overlap in the Y dimension
                        let i_center_y = (rect.y_min + rect.y_max) / 2.0;
                        let j_center_y = (other_rect.y_min + other_rect.y_max) / 2.0;

                        if i_center_y < j_center_y {
                            constraints.push(Constraint::new(i, j, gap));
                        } else {
                            constraints.push(Constraint::new(j, i, gap));
                        }
                    }
                }

                // Add current rectangle to active set
                active_rects.insert(rect.clone());
            }
            EventType::Close => {
                // Remove rectangle from active set
                let rect_to_remove = RectangleY {
                    index: rect_index,
                    ..(*rect).clone()
                };
                active_rects.remove(&rect_to_remove);
            }
        }
    }

    constraints
}

/// Generates separation constraints to prevent rectangle overlaps.
///
/// This is a legacy function maintained for backward compatibility.
/// It directly calls generate_rectangle_no_overlap_constraints_x or
/// generate_rectangle_no_overlap_constraints_y based on the dimension parameter.
///
/// # Arguments
///
/// * `drawing` - A reference to a `DrawingEuclidean2d` that contains the positions of the nodes.
/// * `size` - A function that returns the size of a node in a given dimension.
/// * `k` - The dimension along which to apply the separation constraint (0 for x, 1 for y).
///
/// # Returns
///
/// A vector of `Constraint` objects representing the separation constraints.
pub fn generate_rectangle_no_overlap_constraints<D, Diff, M, F>(
    drawing: &D,
    size: F,
    k: usize,
) -> Vec<Constraint>
where
    D: Drawing<Item = M>,
    D::Index: Clone,
    Diff: Delta<S = f32>,
    M: MetricCartesian<D = Diff>,
    F: FnMut(D::Index, usize) -> f32,
{
    let mut size = size;
    let n = drawing.len();
    let d = drawing.dimension();
    let mut constraints = vec![];
    let sizes = (0..drawing.len())
        .map(|i| {
            let u = drawing.node_id(i);
            let x = drawing.raw_entry(i);
            (0..d)
                .map(|j| {
                    let xj = x.nth(j);
                    let w = size(u.clone(), j) / 2.;
                    (xj - w, xj + w)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for j in 1..n {
        for i in 0..j {
            if overlap(&sizes[i], &sizes[j]) {
                constraints.push(
                    if drawing.raw_entry(i).nth(k) < drawing.raw_entry(j).nth(k) {
                        Constraint::new(
                            i,
                            j,
                            (sizes[i][k].1 - sizes[i][k].0 + sizes[j][k].1 - sizes[j][k].0) / 2.,
                        )
                    } else {
                        Constraint::new(
                            j,
                            i,
                            (sizes[i][k].1 - sizes[i][k].0 + sizes[j][k].1 - sizes[j][k].0) / 2.,
                        )
                    },
                )
            }
        }
    }
    constraints
}

/// Projects rectangle positions to satisfy non-overlap constraints in both dimensions.
///
/// This function is a convenience wrapper that generates constraints for both X and Y
/// dimensions and applies them to ensure rectangles don't overlap. It's equivalent to
/// the `removeOverlaps` function in WebCola.
///
/// # Arguments
///
/// * `drawing` - A mutable reference to a `DrawingEuclidean2d` containing rectangle positions.
/// * `size` - A function that returns the size of a node in a given dimension.
///
/// # Type Parameters
///
/// * `N` - The node ID type.
/// * `F` - The type of the size function.
///
/// # Example
///
/// ```
/// use petgraph::graph::Graph;
/// use petgraph_drawing::DrawingEuclidean2d;
/// use petgraph_layout_separation_constraints::project_rectangle_no_overlap_constraints_2d;
///
/// // Create a graph
/// let mut graph = Graph::<(), (), petgraph::Undirected>::new_undirected();
/// let n1 = graph.add_node(());
/// let n2 = graph.add_node(());
/// let n3 = graph.add_node(());
/// let n4 = graph.add_node(());
///
/// // Create a drawing
/// let mut drawing = DrawingEuclidean2d::new(&graph);
/// drawing.set_x(n1, 0.0);
/// drawing.set_y(n1, 0.0);
/// drawing.set_x(n2, 0.8);
/// drawing.set_y(n2, 0.0);
/// drawing.set_x(n3, 0.0);
/// drawing.set_y(n3, 0.8);
/// drawing.set_x(n4, 0.8);
/// drawing.set_y(n4, 0.8);
///
/// // Apply constraints to remove overlaps
/// let node_size = 1.0; // Size of each node
/// project_rectangle_no_overlap_constraints_2d(&mut drawing, |_, _| node_size);
/// ```
pub fn project_rectangle_no_overlap_constraints_2d<N, F>(
    drawing: &mut DrawingEuclidean2d<N, f32>,
    mut size: F,
) where
    N: DrawingIndex + Copy,
    F: FnMut(N, usize) -> f32,
{
    // Generate and apply constraints for X dimension (0)
    let x_constraints = generate_rectangle_no_overlap_constraints_x(drawing, &mut size);
    project_1d(drawing, 0, &x_constraints);

    // Generate and apply constraints for Y dimension (1)
    let y_constraints = generate_rectangle_no_overlap_constraints_y(drawing, &mut size);
    project_1d(drawing, 1, &y_constraints);
}
