use crate::{project_1d, Constraint};
use petgraph_drawing::{Drawing, DrawingEuclidean2d, DrawingIndex, MetricCartesian};
use std::{cmp::Ordering, collections::BTreeSet};

/// Represents a rectangle with its dimensions and node index
#[derive(Clone, Debug)]
struct Rectangle {
    /// Minimum x-coordinate
    x_min: f32,
    /// Maximum x-coordinate
    x_max: f32,
    /// Minimum y-coordinate
    y_min: f32,
    /// Maximum y-coordinate
    y_max: f32,
}

impl Rectangle {
    fn new(x_min: f32, x_max: f32, y_min: f32, y_max: f32) -> Self {
        Rectangle {
            x_min,
            x_max,
            y_min,
            y_max,
        }
    }
    fn cx(&self) -> f32 {
        (self.x_min + self.x_max) / 2.0
    }

    fn cy(&self) -> f32 {
        (self.y_min + self.y_max) / 2.0
    }

    fn width(&self) -> f32 {
        self.x_max - self.x_min
    }

    fn height(&self) -> f32 {
        self.y_max - self.y_min
    }

    fn overlap_x(&self, r: &Rectangle) -> f32 {
        let ux = self.cx();
        let vx = r.cx();
        if ux <= vx && r.x_min < self.x_max {
            return self.x_max - r.x_min;
        }
        if vx <= ux && self.x_min < r.x_max {
            return r.x_max - self.x_min;
        }
        return 0.0;
    }

    fn overlap_y(&self, r: &Rectangle) -> f32 {
        let uy = self.cy();
        let vy = r.cy();
        if uy <= vy && r.y_min < self.y_max {
            return self.y_max - r.y_min;
        }
        if vy <= uy && self.y_min < r.y_max {
            return r.y_max - self.y_min;
        }
        return 0.0;
    }
}

/// Node structure used for the sweep line algorithm
#[derive(Debug, Clone)]
struct Node {
    /// Variable index
    v: usize,
    /// Rectangle
    r: Rectangle,
    /// Position
    pos: f32,
    /// Previous nodes
    prev: BTreeSet<NodeIndex>,
    /// Next nodes
    next: BTreeSet<NodeIndex>,
}

impl Node {
    fn new_x(v: usize, r: Rectangle) -> Self {
        let pos = r.cx();
        Node {
            v,
            r,
            pos,
            prev: BTreeSet::new(),
            next: BTreeSet::new(),
        }
    }

    fn new_y(v: usize, r: Rectangle) -> Self {
        let pos = r.cy();
        Node {
            v,
            r,
            pos,
            prev: BTreeSet::new(),
            next: BTreeSet::new(),
        }
    }

    fn index(&self) -> NodeIndex {
        NodeIndex {
            pos: self.pos,
            index: self.v,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct NodeIndex {
    pos: f32,
    index: usize,
}

impl Ord for NodeIndex {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl PartialOrd for NodeIndex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.pos.partial_cmp(&other.pos) {
            Some(ord) => Some(match ord {
                Ordering::Equal => self.index.cmp(&other.index),
                _ => ord,
            }),
            None => None,
        }
    }
}

impl PartialEq for NodeIndex {
    fn eq(&self, other: &Self) -> bool {
        (self.pos, self.index).eq(&(other.pos, other.index))
    }
}

impl Eq for NodeIndex {}

/// Event for the sweepline algorithm
#[derive(Debug, Clone)]
struct Event {
    /// Position of the event along the sweep dimension
    pos: f32,
    /// Type of event (open or close)
    is_open: bool,
    /// Associated node index
    v: usize,
}

/// Find X-dimension neighbors
fn find_x_neighbours(v: usize, scanline: &BTreeSet<NodeIndex>, nodes: &mut Vec<Node>) {
    let v_index = nodes[v].index();
    for r in scanline.range(v_index..).skip(1) {
        let u = r.index;
        let u_index = nodes[u].index();
        let u_over_v_x = nodes[u].r.overlap_x(&nodes[v].r);
        if u_over_v_x <= 0.0 || u_over_v_x <= nodes[u].r.overlap_y(&nodes[v].r) {
            nodes[v].next.insert(u_index);
            nodes[u].prev.insert(v_index);
        }
        if u_over_v_x <= 0.0 {
            break;
        }
    }
    let v_index = nodes[v].index();
    for r in scanline.range(..v_index).rev() {
        let u = r.index;
        let u_index = nodes[u].index();
        let u_over_v_x = nodes[u].r.overlap_x(&nodes[v].r);
        if u_over_v_x <= 0.0 || u_over_v_x <= nodes[u].r.overlap_y(&nodes[v].r) {
            nodes[v].prev.insert(u_index);
            nodes[u].next.insert(v_index);
        }
        if u_over_v_x <= 0.0 {
            break;
        }
    }
}

/// Find Y-dimension neighbors according to WebCola's algorithm
fn find_y_neighbours(v: usize, scanline: &BTreeSet<NodeIndex>, nodes: &mut Vec<Node>) {
    let v_index = nodes[v].index();
    for r in scanline.range(v_index..).skip(1) {
        let u = r.index;
        let u_index = nodes[u].index();
        if nodes[u].r.overlap_x(&nodes[v].r) > 0.0 {
            nodes[v].next.insert(u_index);
            nodes[u].prev.insert(v_index);
        }
    }
    let v_index = nodes[v].index();
    for r in scanline.range(..v_index).rev() {
        let u = r.index;
        let u_index = nodes[u].index();
        if nodes[u].r.overlap_x(&nodes[v].r) > 0.0 {
            nodes[v].prev.insert(u_index);
            nodes[u].next.insert(v_index);
        }
    }
}

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

    if n == 0 {
        return constraints;
    }

    // Create rectangle objects with their dimensions
    let mut rectangles = Vec::with_capacity(n);
    let mut nodes = Vec::with_capacity(n);

    for i in 0..n {
        let x = drawing.raw_entry(i).nth(0);
        let y = drawing.raw_entry(i).nth(1);
        let width = size(*drawing.node_id(i), 0);
        let height = size(*drawing.node_id(i), 1);

        let rect = Rectangle::new(
            x - width / 2.0,
            x + width / 2.0,
            y - height / 2.0,
            y + height / 2.0,
        );

        rectangles.push(rect.clone());
        nodes.push(Node::new_x(i, rect));
    }

    // Create events for the sweep line algorithm
    let mut events = Vec::with_capacity(n * 2);
    for i in 0..n {
        let rect = &rectangles[i];
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
        if a.pos > b.pos {
            Ordering::Greater
        } else if a.pos < b.pos {
            return Ordering::Less;
        } else if a.is_open {
            Ordering::Less
        } else if b.is_open {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
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
            while let Some(&w) = nodes[u.index].prev.iter().rev().nth(0) {
                u = w;
                let i = u.index;
                let j = v.index;
                let gap = (nodes[i].r.width() + nodes[j].r.width()) / 2. + 1e-6;
                constraints.push(Constraint::new(i, j, gap));
                nodes[i].next.remove(&v);
            }

            // Create constraints with 'next' neighbors
            let v = nodes[event.v].index();
            let mut u = v;
            while let Some(&w) = nodes[u.index].next.iter().nth(0) {
                u = w;
                let i = v.index;
                let j = u.index;
                let gap = (nodes[i].r.width() + nodes[j].r.width()) / 2. + 1e-6;
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

    if n == 0 {
        return constraints;
    }

    // Create rectangle objects with their dimensions
    let mut rectangles = Vec::with_capacity(n);
    let mut nodes = Vec::with_capacity(n);

    for i in 0..n {
        let x = drawing.raw_entry(i).nth(0);
        let y = drawing.raw_entry(i).nth(1);
        let width = size(*drawing.node_id(i), 0);
        let height = size(*drawing.node_id(i), 1);

        let rect = Rectangle::new(
            x - width / 2.0,
            x + width / 2.0,
            y - height / 2.0,
            y + height / 2.0,
        );

        rectangles.push(rect.clone());
        nodes.push(Node::new_y(i, rect));
    }

    // Create events for the sweep line algorithm
    let mut events = Vec::with_capacity(n * 2);
    for i in 0..n {
        let rect = &rectangles[i];
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
        if a.pos > b.pos {
            Ordering::Greater
        } else if a.pos < b.pos {
            return Ordering::Less;
        } else if a.is_open {
            Ordering::Less
        } else if b.is_open {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
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
            while let Some(&w) = nodes[u.index].prev.iter().rev().nth(0) {
                u = w;
                let i = u.index;
                let j = v.index;
                let gap = (nodes[i].r.height() + nodes[j].r.height()) / 2. + 1e-6;
                constraints.push(Constraint::new(i, j, gap));
                nodes[i].next.remove(&v);
            }

            // Create constraints with 'next' neighbors
            let v = nodes[event.v].index();
            let mut u = v;
            while let Some(&w) = nodes[u.index].next.iter().nth(0) {
                u = w;
                let i = v.index;
                let j = u.index;
                let gap = (nodes[i].r.height() + nodes[j].r.height()) / 2. + 1e-6;
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
    let size = vec![
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
    let size = vec![vec![10.0, 10.0], vec![10.0, 10.0], vec![10.0, 10.0]];
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
