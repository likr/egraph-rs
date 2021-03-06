use js_sys::Array;
use petgraph::graph::{edge_index, node_index};
use petgraph::{Directed, Direction};
use wasm_bindgen::prelude::*;

pub type Node = JsValue;
pub type Edge = JsValue;
pub type EdgeType = Directed;
pub type IndexType = u32;
type GraphType = petgraph::Graph<Node, Edge, EdgeType, IndexType>;

#[wasm_bindgen(js_name = Graph)]
pub struct JsGraph {
    graph: GraphType,
}

impl JsGraph {
    pub fn graph(&self) -> &GraphType {
        &self.graph
    }
}

#[wasm_bindgen(js_class = Graph)]
impl JsGraph {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsGraph {
        JsGraph {
            graph: GraphType::with_capacity(0, 0),
        }
    }

    #[wasm_bindgen(js_name = nodeCount)]
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    #[wasm_bindgen(js_name = edgeCount)]
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    #[wasm_bindgen(js_name = addNode)]
    pub fn add_node(&mut self, value: JsValue) -> usize {
        self.graph.add_node(value).index()
    }

    #[wasm_bindgen(js_name = nodeWeight)]
    pub fn node_weight(&self, a: usize) -> Result<JsValue, JsValue> {
        let a = node_index(a);
        self.graph
            .node_weight(a)
            .map(|node| node.clone())
            .ok_or_else(|| "invalid node index".into())
    }

    #[wasm_bindgen(js_name = addEdge)]
    pub fn add_edge(&mut self, a: usize, b: usize, value: JsValue) -> usize {
        let a = node_index(a);
        let b = node_index(b);
        self.graph.add_edge(a, b, value).index()
    }

    #[wasm_bindgen(js_name = edgeWeight)]
    pub fn edge_weight(&mut self, e: usize) -> Result<JsValue, JsValue> {
        let e = edge_index(e);
        self.graph
            .edge_weight(e)
            .map(|edge| edge.clone())
            .ok_or_else(|| "invalid edge index".into())
    }

    #[wasm_bindgen(js_name = edgeEndpoints)]
    pub fn edge_endpoints(&self, e: usize) -> Result<Array, JsValue> {
        let e = edge_index(e);
        self.graph
            .edge_endpoints(e)
            .map(|(u, v)| {
                [u, v]
                    .iter()
                    .map(|a| JsValue::from_f64(a.index() as f64))
                    .collect::<Array>()
            })
            .ok_or_else(|| "invalid edge index".into())
    }

    #[wasm_bindgen(js_name = removeNode)]
    pub fn remove_node(&mut self, a: usize) -> Result<JsValue, JsValue> {
        let a = node_index(a);
        self.graph
            .remove_node(a)
            .ok_or_else(|| "invalid node index".into())
    }

    #[wasm_bindgen(js_name = removeEdge)]
    pub fn remove_edge(&mut self, e: usize) -> Result<JsValue, JsValue> {
        let e = edge_index(e);
        self.graph
            .remove_edge(e)
            .ok_or_else(|| "invalid node index".into())
    }

    pub fn neighbors(&self, a: usize) -> Array {
        self.graph
            .neighbors(node_index(a))
            .map(|u| JsValue::from_f64(u.index() as f64))
            .collect::<Array>()
    }

    #[wasm_bindgen(js_name = neighborsDirected)]
    pub fn neighbors_directed(&self, a: usize, dir: usize) -> Array {
        let a = node_index(a);
        let dir = match dir {
            0 => Direction::Outgoing,
            _ => Direction::Incoming,
        };
        self.graph
            .neighbors_directed(a, dir)
            .map(|u| JsValue::from_f64(u.index() as f64))
            .collect::<Array>()
    }

    #[wasm_bindgen(js_name = neighborsUndirected)]
    pub fn neighbors_undirected(&self, a: usize) -> Array {
        self.graph
            .neighbors_undirected(node_index(a))
            .map(|u| JsValue::from_f64(u.index() as f64))
            .collect::<Array>()
    }

    pub fn edges(&self, a: usize) -> Array {
        self.graph
            .edges(node_index(a))
            .map(|e| e.weight().clone())
            .collect::<Array>()
    }

    #[wasm_bindgen(js_name = containsEdge)]
    pub fn contains_edge(&self, a: usize, b: usize) -> bool {
        let a = node_index(a);
        let b = node_index(b);
        self.graph.contains_edge(a, b)
    }

    #[wasm_bindgen(js_name = findEdge)]
    pub fn find_edge(&self, a: usize, b: usize) -> Result<usize, JsValue> {
        let a = node_index(a);
        let b = node_index(b);
        self.graph
            .find_edge(a, b)
            .map(|e| e.index())
            .ok_or_else(|| "invalid edge index".into())
    }

    pub fn externals(&self, dir: usize) -> Array {
        let dir = match dir {
            0 => Direction::Outgoing,
            _ => Direction::Incoming,
        };
        self.graph
            .externals(dir)
            .map(|u| JsValue::from_f64(u.index() as f64))
            .collect::<Array>()
    }

    #[wasm_bindgen(js_name = nodeIndices)]
    pub fn node_indices(&self) -> Array {
        self.graph
            .node_indices()
            .map(|u| JsValue::from_f64(u.index() as f64))
            .collect::<Array>()
    }

    #[wasm_bindgen(js_name = edgeIndices)]
    pub fn edge_indices(&self) -> Array {
        self.graph
            .edge_indices()
            .map(|e| JsValue::from_f64(e.index() as f64))
            .collect::<Array>()
    }
}
