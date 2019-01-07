use js_sys::{Object, Reflect, Symbol};
use petgraph::graph::{edge_index, node_index};
use petgraph::prelude::*;
use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;
use wasm_bindgen::prelude::*;

pub type Node = Object;
pub type Edge = Object;
pub type EdgeType = Directed;
pub type IndexType = usize;
type GraphType = petgraph::Graph<Node, Edge, EdgeType, IndexType>;

fn next(value: Option<JsValue>) -> JsValue {
    let obj = Object::new();
    if let Some(v) = value {
        Reflect::set(&obj, &"done".into(), &false.into())
            .ok()
            .unwrap();
        Reflect::set(&obj, &"value".into(), &v).ok().unwrap();
    } else {
        Reflect::set(&obj, &"done".into(), &true.into())
            .ok()
            .unwrap();
    }
    obj.into()
}

fn iterable(f: Box<Fn() -> JsValue>) -> JsValue {
    let obj = Object::new();
    let closure = Closure::wrap(f);
    Reflect::set(&obj, &Symbol::iterator(), closure.as_ref())
        .ok()
        .unwrap();
    closure.forget(); // XXX ?
    obj.into()
}

#[wasm_bindgen]
struct Neighbors {
    iter: petgraph::graph::WalkNeighbors<usize>,
    graph: Rc<RefCell<GraphType>>,
}

#[wasm_bindgen]
impl Neighbors {
    pub fn next(&mut self) -> JsValue {
        next(
            self.iter
                .next_node(&self.graph.borrow())
                .map(|index| (index.index() as u32).into()),
        )
    }
}

#[wasm_bindgen]
pub struct NodeIndices {
    iter: petgraph::graph::NodeIndices<usize>,
}

#[wasm_bindgen]
impl NodeIndices {
    pub fn next(&mut self) -> JsValue {
        next(self.iter.next().map(|index| (index.index() as u32).into()))
    }
}

#[wasm_bindgen]
pub struct EdgeIndices {
    iter: petgraph::graph::EdgeIndices<usize>,
}

#[wasm_bindgen]
impl EdgeIndices {
    pub fn next(&mut self) -> JsValue {
        next(self.iter.next().map(|index| (index.index() as u32).into()))
    }
}

#[wasm_bindgen]
pub struct Graph {
    graph: Rc<RefCell<GraphType>>,
}

impl Graph {
    pub fn graph(&self) -> Ref<GraphType> {
        self.graph.borrow()
    }

    pub fn graph_mut(&self) -> RefMut<GraphType> {
        self.graph.borrow_mut()
    }
}

#[wasm_bindgen]
impl Graph {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Graph {
        Graph {
            graph: Rc::new(RefCell::new(GraphType::with_capacity(0, 0))),
        }
    }

    #[wasm_bindgen(js_name = addNode)]
    pub fn add_node(&mut self) -> usize {
        self.graph_mut().add_node(Object::new()).index()
    }

    #[wasm_bindgen(js_name = addEdge)]
    pub fn add_edge(&mut self, u: usize, v: usize) -> usize {
        let u = node_index(u);
        let v = node_index(v);
        self.graph_mut().add_edge(u, v, Object::new()).index()
    }

    #[wasm_bindgen(js_name = removeNode)]
    pub fn remove_node(&mut self, u: usize) {
        self.graph_mut().remove_node(node_index(u));
    }

    #[wasm_bindgen(js_name = removeEdge)]
    pub fn remove_edge(&mut self, u: usize) {
        self.graph_mut().remove_edge(edge_index(u));
    }

    #[wasm_bindgen(js_name = nodeCount)]
    pub fn node_count(&self) -> usize {
        self.graph().node_count()
    }

    #[wasm_bindgen(js_name = edgeCount)]
    pub fn edge_count(&self) -> usize {
        self.graph().edge_count()
    }

    #[wasm_bindgen(js_name = neighbors)]
    pub fn neighbors(&self, a: usize) -> JsValue {
        let graph = self.graph.clone();
        iterable(Box::new(move || {
            (Neighbors {
                iter: graph.borrow().neighbors(node_index(a)).detach(),
                graph: graph.clone(),
            })
            .into()
        }) as Box<Fn() -> JsValue>)
    }

    #[wasm_bindgen(js_name = nodeIndices)]
    pub fn node_indices(&self) -> JsValue {
        let graph = self.graph.clone();
        iterable(Box::new(move || {
            (NodeIndices {
                iter: graph.borrow().node_indices(),
            })
            .into()
        }))
    }

    #[wasm_bindgen(js_name = edgeIndices)]
    pub fn edge_indices(&self) -> JsValue {
        let graph = self.graph.clone();
        iterable(Box::new(move || {
            (EdgeIndices {
                iter: graph.borrow().edge_indices(),
            })
            .into()
        }))
    }

    pub fn node(&self, a: usize) -> Object {
        self.graph()[node_index(a)].clone()
    }

    pub fn edge(&self, e: usize) -> Object {
        self.graph()[edge_index(e)].clone()
    }

    pub fn source(&self, e: usize) -> usize {
        self.graph()
            .edge_endpoints(edge_index(e))
            .unwrap()
            .0
            .index()
    }

    pub fn target(&self, e: usize) -> usize {
        self.graph()
            .edge_endpoints(edge_index(e))
            .unwrap()
            .1
            .index()
    }
}
