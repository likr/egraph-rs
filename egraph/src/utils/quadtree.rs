#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub enum Region {
    TL,
    TR,
    BL,
    BR,
}

#[derive(Copy, Clone, Debug)]
pub struct Rect {
    pub cx: f32,
    pub cy: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    fn quad(self, x: f32, y: f32) -> Region {
        let is_left = x < self.cx;
        let is_bottom = y < self.cy;
        match (is_bottom, is_left) {
            (true, true) => Region::BL,
            (true, false) => Region::BR,
            (false, true) => Region::TL,
            (false, false) => Region::TR,
        }
    }

    fn top_left(self) -> Rect {
        Rect {
            cx: self.cx - self.width / 4.,
            cy: self.cy + self.height / 4.,
            width: self.width / 2.,
            height: self.height / 2.,
        }
    }

    fn top_right(self) -> Rect {
        Rect {
            cx: self.cx + self.width / 4.,
            cy: self.cy + self.height / 4.,
            width: self.width / 2.,
            height: self.height / 2.,
        }
    }

    fn bottom_left(self) -> Rect {
        Rect {
            cx: self.cx - self.width / 4.,
            cy: self.cy - self.height / 4.,
            width: self.width / 2.,
            height: self.height / 2.,
        }
    }

    fn bottom_right(self) -> Rect {
        Rect {
            cx: self.cx + self.width / 4.,
            cy: self.cy - self.height / 4.,
            width: self.width / 2.,
            height: self.height / 2.,
        }
    }

    pub fn sub_rect(self, region: Region) -> Rect {
        match region {
            Region::TL => self.top_left(),
            Region::TR => self.top_right(),
            Region::BL => self.bottom_left(),
            Region::BR => self.bottom_right(),
        }
    }

    pub fn left(self) -> f32 {
        self.cx - self.width / 2.
    }

    pub fn bottom(self) -> f32 {
        self.cy - self.height / 2.
    }

    pub fn width(self) -> f32 {
        self.width
    }

    pub fn height(self) -> f32 {
        self.height
    }
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct NodeId {
    index: usize,
}

#[derive(Copy, Clone, Debug)]
pub enum Element {
    Empty,
    Leaf { x: f32, y: f32, n: usize },
    Node { node_id: NodeId },
}

#[derive(Clone, Debug)]
pub struct Node<T> {
    parent: Option<NodeId>,
    top_left: Box<Element>,
    top_right: Box<Element>,
    bottom_left: Box<Element>,
    bottom_right: Box<Element>,
    rect: Rect,
    data: T,
}

#[derive(Clone, Debug)]
pub struct Quadtree<T> {
    root: NodeId,
    nodes: Vec<Node<T>>,
}

impl<T: Default> Node<T> {
    pub fn new(rect: Rect) -> Node<T> {
        Node {
            parent: None,
            top_left: Box::new(Element::Empty),
            top_right: Box::new(Element::Empty),
            bottom_left: Box::new(Element::Empty),
            bottom_right: Box::new(Element::Empty),
            rect: rect,
            data: T::default(),
        }
    }

    pub fn new_with_parent(rect: Rect, parent: NodeId) -> Node<T> {
        Node {
            parent: Some(parent),
            top_left: Box::new(Element::Empty),
            top_right: Box::new(Element::Empty),
            bottom_left: Box::new(Element::Empty),
            bottom_right: Box::new(Element::Empty),
            rect: rect,
            data: T::default(),
        }
    }

    pub fn child(&self, region: Region) -> Element {
        match region {
            Region::TL => *self.top_left,
            Region::TR => *self.top_right,
            Region::BL => *self.bottom_left,
            Region::BR => *self.bottom_right,
        }
    }

    pub fn insert(&mut self, region: Region, element: Element) {
        match region {
            Region::TL => self.top_left = Box::new(element),
            Region::TR => self.top_right = Box::new(element),
            Region::BL => self.bottom_left = Box::new(element),
            Region::BR => self.bottom_right = Box::new(element),
        }
    }
}

impl<T: Default> Quadtree<T> {
    pub fn new(rect: Rect) -> Quadtree<T> {
        let mut nodes = Vec::new();
        nodes.push(Node::<T>::new(rect));
        Quadtree {
            root: NodeId { index: 0 },
            nodes: nodes,
        }
    }

    pub fn root(&self) -> NodeId {
        self.root
    }

    pub fn find(&self, u: NodeId, x: f32, y: f32) -> (NodeId, Region) {
        let node = &self.nodes[u.index];
        let region = node.rect.quad(x, y);
        let child = node.child(region);
        match child {
            Element::Node { node_id: v } => self.find(v, x, y),
            _ => (u, region),
        }
    }

    pub fn insert(&mut self, u: NodeId, x: f32, y: f32) -> (NodeId, Region) {
        let (v, region) = self.find(u, x, y);
        match self.nodes[v.index].child(region) {
            Element::Empty => self.insert_to_empty(v, region, x, y),
            Element::Leaf { x: x0, y: y0, n } => {
                if x == x && y == y0 {
                    self.increment_leaf(v, region, x, y, n)
                } else {
                    self.insert_to_leaf(v, region, x, y, x0, y0, n)
                }
            }
            _ => {
                panic!("unexpected");
            }
        }
    }

    fn insert_to_empty(&mut self, u: NodeId, region: Region, x: f32, y: f32) -> (NodeId, Region) {
        let node = self.nodes.get_mut(u.index).unwrap();
        node.insert(region, Element::Leaf { x: x, y: y, n: 1 });
        (u, region)
    }

    fn insert_to_leaf(
        &mut self,
        u: NodeId,
        region: Region,
        x: f32,
        y: f32,
        x0: f32,
        y0: f32,
        n: usize,
    ) -> (NodeId, Region) {
        let index = self.nodes.len();
        let rect = self.nodes[u.index].rect.sub_rect(region);
        self.nodes.push(Node::new_with_parent(rect, u));
        let new_node = NodeId { index: index };
        self.nodes[u.index].insert(region, Element::Node { node_id: new_node });
        let region = self.nodes[new_node.index].rect.quad(x0, y0);
        self.nodes[new_node.index].insert(region, Element::Leaf { x: x0, y: y0, n: n });
        self.insert(new_node, x, y)
    }

    fn increment_leaf(
        &mut self,
        u: NodeId,
        region: Region,
        x: f32,
        y: f32,
        n: usize,
    ) -> (NodeId, Region) {
        self.nodes[u.index].insert(
            region,
            Element::Leaf {
                x: x,
                y: y,
                n: n + 1,
            },
        );
        (u, region)
    }

    pub fn rect(&self, u: NodeId) -> Rect {
        self.nodes[u.index].rect
    }

    pub fn element(&self, u: NodeId, region: Region) -> Element {
        self.nodes[u.index].child(region)
    }

    pub fn elements(&self, u: NodeId) -> [(Box<Element>, Region); 4] {
        let node = &self.nodes[u.index];
        [
            (node.top_left.clone(), Region::TL),
            (node.top_right.clone(), Region::TR),
            (node.bottom_left.clone(), Region::BL),
            (node.bottom_right.clone(), Region::BR),
        ]
    }

    pub fn data(&self, u: NodeId) -> &T {
        &self.nodes[u.index].data
    }

    pub fn data_mut(&mut self, u: NodeId) -> &mut T {
        &mut self.nodes[u.index].data
    }
}

#[cfg(test)]
fn make_tree() -> Quadtree<()> {
    Quadtree::new(Rect {
        cx: 0.,
        cy: 0.,
        width: 100.,
        height: 100.,
    })
}

#[test]
fn test_find() {
    let tree = make_tree();
    let root = tree.root();
    let (node_id, region) = tree.find(root, 10., 10.);
    assert!(node_id.index == 0);
    assert!(region == Region::TR);
}

#[test]
fn test_insert() {
    let mut tree = make_tree();
    let root = tree.root();
    let (node_id, region) = tree.insert(root, 10., 10.);
    assert!(node_id.index == 0);
    assert!(region == Region::TR);
    let (node_id, region) = tree.insert(root, 20., 40.);
    assert!(node_id.index == 1);
    assert!(region == Region::TL);
    let (node_id, region) = tree.insert(root, 10., 30.);
    assert!(node_id.index == 2);
    assert!(region == Region::BL);
}

#[test]
fn test_elements() {
    let tree = make_tree();
    let root = tree.root();
    for &(ref e, _) in tree.elements(root).iter() {
        assert!(match **e {
            Element::Empty => true,
            _ => false,
        });
    }
}
