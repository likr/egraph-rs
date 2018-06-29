use petgraph::graph::NodeIndex;

#[derive(Clone)]
pub struct Node {
    pub layer: usize,
    pub order: usize,
    pub width: usize,
    pub height: usize,
    pub orig_width: usize,
    pub orig_height: usize,
    pub x: i32,
    pub y: i32,
    pub dummy: bool,
    pub align: Option<NodeIndex>,
    pub root: Option<NodeIndex>,
    pub sink: Option<NodeIndex>,
    pub shift: i32,
}

impl Node {
    pub fn new() -> Node {
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
            align: None,
            root: None,
            sink: None,
            shift: i32::min_value(),
        }
    }

    pub fn new_dummy() -> Node {
        let mut node = Node::new();
        node.dummy = true;
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
