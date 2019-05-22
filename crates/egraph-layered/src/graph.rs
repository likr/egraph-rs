use petgraph::graph::{EdgeIndex, IndexType, NodeIndex};

#[derive(Clone)]
pub struct Node<Ix: IndexType> {
    pub layer: usize,
    pub order: usize,
    pub width: usize,
    pub height: usize,
    pub orig_width: usize,
    pub orig_height: usize,
    pub x: i32,
    pub y: i32,
    pub dummy: bool,
    pub edge_index: Option<EdgeIndex<Ix>>,
    pub align: Option<NodeIndex<Ix>>,
    pub root: Option<NodeIndex<Ix>>,
    pub sink: Option<NodeIndex<Ix>>,
    pub shift: i32,
}

impl<Ix: IndexType> Node<Ix> {
    pub fn new() -> Node<Ix> {
        Node {
            layer: 0,
            order: 0,
            width: 0,
            height: 0,
            orig_width: 0,
            orig_height: 0,
            x: 0,
            y: 0,
            dummy: false,
            edge_index: None,
            align: None,
            root: None,
            sink: None,
            shift: i32::min_value(),
        }
    }

    pub fn new_dummy(e: EdgeIndex<Ix>) -> Node<Ix> {
        let mut node = Node::new();
        node.dummy = true;
        node.edge_index = Some(e);
        node
    }
}

#[derive(Clone)]
pub struct Edge {
    pub conflict: bool,
    pub reversed: bool,
}

impl Edge {
    pub fn new() -> Edge {
        Edge {
            conflict: false,
            reversed: false,
        }
    }

    pub fn new_reversed() -> Edge {
        let mut edge = Edge::new();
        edge.reversed = true;
        edge
    }

    pub fn new_split(original: &Edge) -> Edge {
        if original.reversed {
            Edge::new_reversed()
        } else {
            Edge::new()
        }
    }
}
