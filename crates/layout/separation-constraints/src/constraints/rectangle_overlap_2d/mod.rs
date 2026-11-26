mod geometry;

use self::geometry::{find_x_neighbours, find_y_neighbours, Event, Node, Rectangle};
use crate::{project_1d, Constraint};
use petgraph_drawing::{Drawing, DrawingEuclidean2d, DrawingIndex, DrawingValue, MetricCartesian};
use std::{cmp::Ordering, collections::BTreeSet};

/// Generates separation constraints to prevent rectangle overlaps in the X dimension.
///
/// This function uses a sweep line algorithm to efficiently find potential overlapping
/// rectangle pairs and generate appropriate X-dimension constraints. This implementation
/// is based on the WebCola algorithm but uses Rust's BTreeSet instead of an RBTree.
fn generate_rectangle_no_overlap_constraints_x<N, F, S>(
    drawing: &DrawingEuclidean2d<N, S>,
    mut size: F,
) -> Vec<Constraint<S>>
where
    N: DrawingIndex + Copy,
    F: FnMut(N, usize) -> S,
    S: DrawingValue,
{
    let n = drawing.len();
    let mut constraints = Vec::new();

    if n == 0 {
        return constraints;
    }

    // Create rectangle objects with their dimensions
    let mut rectangles = Vec::with_capacity(n);
    let mut nodes = Vec::with_capacity(n);

    for i in 0..n {
        let x = *drawing.raw_entry(i).nth(0);
        let y = *drawing.raw_entry(i).nth(1);
        let width = size(*drawing.node_id(i), 0);
        let height = size(*drawing.node_id(i), 1);

        let rect = Rectangle::new(
            x - width / (2.0).into(),
            x + width / (2.0).into(),
            y - height / (2.0).into(),
            y + height / (2.0).into(),
        );

        rectangles.push(rect.clone());
        nodes.push(Node::new_x(i, rect));
    }

    // Create events for the sweep line algorithm
    let mut events = Vec::with_capacity(n * 2);
    for (i, rect) in rectangles.iter().enumerate().take(n) {
        events.push(Event {
            pos: rect.y_min,
            is_open: true,
            v: i,
        });
        events.push(Event {
            pos: rect.y_max,
            is_open: false,
            v: i,
        });
    }

    // Sort events by position
    events.sort_by(|a, b| {
        a.pos
            .partial_cmp(&b.pos)
            .unwrap_or(Ordering::Equal)
            .then_with(|| match (a.is_open, b.is_open) {
                (true, false) => Ordering::Less,
                (false, true) => Ordering::Greater,
                _ => Ordering::Equal,
            })
    });

    // Use BTreeSet to maintain active nodes
    let mut scanline = BTreeSet::new();

    // Process events in order
    for event in events {
        if event.is_open {
            // Open event - find neighbors and insert into scanline
            scanline.insert(nodes[event.v].index());
            find_x_neighbours(event.v, &scanline, &mut nodes);
        } else {
            // Close event - create constraints and remove from scanline

            // Remove node from scanline
            scanline.remove(&nodes[event.v].index());

            // Create constraints with 'prev' neighbors
            let v = nodes[event.v].index();
            let mut u = v;
            while let Some(&w) = nodes[u.index].prev.iter().next_back() {
                u = w;
                let i = u.index;
                let j = v.index;
                let gap = (nodes[i].r.width() + nodes[j].r.width()) / (2. + 1e-6).into();
                constraints.push(Constraint::new(i, j, gap));
                nodes[i].next.remove(&v);
            }

            // Create constraints with 'next' neighbors
            let v = nodes[event.v].index();
            let mut u = v;
            while let Some(&w) = nodes[u.index].next.iter().next() {
                u = w;
                let i = v.index;
                let j = u.index;
                let gap = (nodes[i].r.width() + nodes[j].r.width()) / (2. + 1e-6).into();
                constraints.push(Constraint::new(i, j, gap));
                nodes[j].prev.remove(&v);
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
fn generate_rectangle_no_overlap_constraints_y<N, F, S>(
    drawing: &DrawingEuclidean2d<N, S>,
    mut size: F,
) -> Vec<Constraint<S>>
where
    N: DrawingIndex + Copy,
    F: FnMut(N, usize) -> S,
    S: DrawingValue,
{
    let n = drawing.len();
    let mut constraints = Vec::new();

    if n == 0 {
        return constraints;
    }

    // Create rectangle objects with their dimensions
    let mut rectangles = Vec::with_capacity(n);
    let mut nodes = Vec::with_capacity(n);

    for i in 0..n {
        let x = *drawing.raw_entry(i).nth(0);
        let y = *drawing.raw_entry(i).nth(1);
        let width = size(*drawing.node_id(i), 0);
        let height = size(*drawing.node_id(i), 1);

        let rect = Rectangle::new(
            x - width / (2.0).into(),
            x + width / (2.0).into(),
            y - height / (2.0).into(),
            y + height / (2.0).into(),
        );

        rectangles.push(rect.clone());
        nodes.push(Node::new_y(i, rect));
    }

    // Create events for the sweep line algorithm
    let mut events = Vec::with_capacity(n * 2);
    for (i, rect) in rectangles.iter().enumerate().take(n) {
        events.push(Event {
            pos: rect.x_min,
            is_open: true,
            v: i,
        });
        events.push(Event {
            pos: rect.x_max,
            is_open: false,
            v: i,
        });
    }

    // Sort events by position
    events.sort_by(|a, b| {
        a.pos
            .partial_cmp(&b.pos)
            .unwrap_or(Ordering::Equal)
            .then_with(|| match (a.is_open, b.is_open) {
                (true, false) => Ordering::Less,
                (false, true) => Ordering::Greater,
                _ => Ordering::Equal,
            })
    });

    // Use BTreeSet to maintain active nodes
    let mut scanline = BTreeSet::new();

    // Process events in order
    for event in events {
        if event.is_open {
            // Open event - find neighbors and insert into scanline
            scanline.insert(nodes[event.v].index());
            find_y_neighbours(event.v, &scanline, &mut nodes);
        } else {
            // Close event - create constraints and remove from scanline

            // Remove node from scanline
            scanline.remove(&nodes[event.v].index());

            // Create constraints with 'prev' neighbors
            let v = nodes[event.v].index();
            let mut u = v;
            while let Some(&w) = nodes[u.index].prev.iter().next_back() {
                u = w;
                let i = u.index;
                let j = v.index;
                let gap = (nodes[i].r.height() + nodes[j].r.height()) / (2. + 1e-6).into();
                constraints.push(Constraint::new(i, j, gap));
                nodes[i].next.remove(&v);
            }

            // Create constraints with 'next' neighbors
            let v = nodes[event.v].index();
            let mut u = v;
            while let Some(&w) = nodes[u.index].next.iter().next() {
                u = w;
                let i = v.index;
                let j = u.index;
                let gap = (nodes[i].r.height() + nodes[j].r.height()) / (2. + 1e-6).into();
                constraints.push(Constraint::new(i, j, gap));
                nodes[j].prev.remove(&v);
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
pub fn project_rectangle_no_overlap_constraints_2d<N, F, S>(
    drawing: &mut DrawingEuclidean2d<N, S>,
    mut size: F,
) where
    N: DrawingIndex + Copy,
    F: FnMut(N, usize) -> S,
    S: DrawingValue,
{
    // Generate and apply constraints for X dimension (0)
    let x_constraints = generate_rectangle_no_overlap_constraints_x(drawing, &mut size);
    project_1d(drawing, 0, &x_constraints);

    // Generate and apply constraints for Y dimension (1)
    let y_constraints = generate_rectangle_no_overlap_constraints_y(drawing, &mut size);
    project_1d(drawing, 1, &y_constraints);
}

#[cfg(test)]
mod tests;
