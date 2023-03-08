use crate::graph::{Edge, IndexType, Node};
use js_sys::{Array, Function};
use petgraph::{
    graph::{edge_index, node_index, Graph},
    Directed, Direction, EdgeType, Undirected,
};
use wasm_bindgen::prelude::*;

struct GraphBase<Ty: EdgeType> {
    graph: Graph<Node, Edge, Ty, IndexType>,
}

impl<Ty: EdgeType> GraphBase<Ty> {
    pub fn new() -> Self {
        Self::new_from_graph(Graph::<Node, Edge, Ty, IndexType>::with_capacity(0, 0))
    }

    pub fn new_from_graph(graph: Graph<Node, Edge, Ty, IndexType>) -> Self {
        Self { graph }
    }

    pub fn graph(&self) -> &Graph<Node, Edge, Ty, IndexType> {
        &self.graph
    }

    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    pub fn add_node(&mut self, value: JsValue) -> usize {
        self.graph.add_node(value).index()
    }

    pub fn node_weight(&self, a: usize) -> Result<JsValue, JsValue> {
        let a = node_index(a);
        self.graph
            .node_weight(a)
            .map(|node| node.clone())
            .ok_or_else(|| "invalid node index".into())
    }

    pub fn add_edge(&mut self, a: usize, b: usize, value: JsValue) -> usize {
        let a = node_index(a);
        let b = node_index(b);
        self.graph.add_edge(a, b, value).index()
    }

    pub fn edge_weight(&mut self, e: usize) -> Result<JsValue, JsValue> {
        let e = edge_index(e);
        self.graph
            .edge_weight(e)
            .map(|edge| edge.clone())
            .ok_or_else(|| "invalid edge index".into())
    }

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

    pub fn remove_node(&mut self, a: usize) -> Result<JsValue, JsValue> {
        let a = node_index(a);
        self.graph
            .remove_node(a)
            .ok_or_else(|| "invalid node index".into())
    }

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

    pub fn contains_edge(&self, a: usize, b: usize) -> bool {
        let a = node_index(a);
        let b = node_index(b);
        self.graph.contains_edge(a, b)
    }

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

    pub fn node_indices(&self) -> Array {
        self.graph
            .node_indices()
            .map(|u| JsValue::from_f64(u.index() as f64))
            .collect::<Array>()
    }

    pub fn edge_indices(&self) -> Array {
        self.graph
            .edge_indices()
            .map(|e| JsValue::from_f64(e.index() as f64))
            .collect::<Array>()
    }

    pub fn map(&self, node_map: &Function, edge_map: &Function) -> Self {
        Self {
            graph: self.graph.map(
                |u, node| {
                    node_map
                        .call2(&JsValue::null(), &(u.index() as f64).into(), node)
                        .unwrap()
                },
                |e, edge| {
                    edge_map
                        .call2(&JsValue::null(), &(e.index() as f64).into(), edge)
                        .unwrap()
                },
            ),
        }
    }

    pub fn filter_map(&self, node_map: &Function, edge_map: &Function) -> Self {
        Self {
            graph: self.graph.filter_map(
                |u, node| {
                    let result = node_map
                        .call2(&JsValue::null(), &(u.index() as f64).into(), node)
                        .unwrap();
                    if result.is_null() {
                        None
                    } else {
                        Some(result)
                    }
                },
                |e, edge| {
                    let result = edge_map
                        .call2(&JsValue::null(), &(e.index() as f64).into(), edge)
                        .unwrap();
                    if result.is_null() {
                        None
                    } else {
                        Some(result)
                    }
                },
            ),
        }
    }
}

#[wasm_bindgen(js_name = Graph)]
pub struct JsGraph {
    graph: GraphBase<Undirected>,
}

impl JsGraph {
    pub fn new_from_graph(graph: Graph<Node, Edge, Undirected, IndexType>) -> Self {
        Self {
            graph: GraphBase::<Undirected>::new_from_graph(graph),
        }
    }

    pub fn graph(&self) -> &Graph<Node, Edge, Undirected, IndexType> {
        &self.graph.graph()
    }
}

#[wasm_bindgen(js_class = Graph)]
impl JsGraph {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            graph: GraphBase::<Undirected>::new(),
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
        self.graph.add_node(value)
    }

    #[wasm_bindgen(js_name = nodeWeight)]
    pub fn node_weight(&self, a: usize) -> Result<JsValue, JsValue> {
        self.graph.node_weight(a)
    }

    #[wasm_bindgen(js_name = addEdge)]
    pub fn add_edge(&mut self, a: usize, b: usize, value: JsValue) -> usize {
        self.graph.add_edge(a, b, value)
    }

    #[wasm_bindgen(js_name = edgeWeight)]
    pub fn edge_weight(&mut self, e: usize) -> Result<JsValue, JsValue> {
        self.graph.edge_weight(e)
    }

    #[wasm_bindgen(js_name = edgeEndpoints)]
    pub fn edge_endpoints(&self, e: usize) -> Result<Array, JsValue> {
        self.graph.edge_endpoints(e)
    }

    #[wasm_bindgen(js_name = removeNode)]
    pub fn remove_node(&mut self, a: usize) -> Result<JsValue, JsValue> {
        self.graph.remove_node(a)
    }

    #[wasm_bindgen(js_name = removeEdge)]
    pub fn remove_edge(&mut self, e: usize) -> Result<JsValue, JsValue> {
        self.graph.remove_edge(e)
    }

    pub fn neighbors(&self, a: usize) -> Array {
        self.graph.neighbors(a)
    }

    #[wasm_bindgen(js_name = neighborsDirected)]
    pub fn neighbors_directed(&self, a: usize, dir: usize) -> Array {
        self.graph.neighbors_directed(a, dir)
    }

    #[wasm_bindgen(js_name = neighborsUndirected)]
    pub fn neighbors_undirected(&self, a: usize) -> Array {
        self.graph.neighbors_undirected(a)
    }

    pub fn edges(&self, a: usize) -> Array {
        self.graph.edges(a)
    }

    #[wasm_bindgen(js_name = containsEdge)]
    pub fn contains_edge(&self, a: usize, b: usize) -> bool {
        self.graph.contains_edge(a, b)
    }

    #[wasm_bindgen(js_name = findEdge)]
    pub fn find_edge(&self, a: usize, b: usize) -> Result<usize, JsValue> {
        self.graph.find_edge(a, b)
    }

    pub fn externals(&self, dir: usize) -> Array {
        self.graph.externals(dir)
    }

    #[wasm_bindgen(js_name = nodeIndices)]
    pub fn node_indices(&self) -> Array {
        self.graph.node_indices()
    }

    #[wasm_bindgen(js_name = edgeIndices)]
    pub fn edge_indices(&self) -> Array {
        self.graph.edge_indices()
    }

    pub fn map(&self, node_map: &Function, edge_map: &Function) -> Self {
        Self {
            graph: self.graph.map(node_map, edge_map),
        }
    }

    #[wasm_bindgen(js_name = filterMap)]
    pub fn filter_map(&self, node_map: &Function, edge_map: &Function) -> Self {
        Self {
            graph: self.graph.filter_map(node_map, edge_map),
        }
    }
}

#[wasm_bindgen(js_name = DiGraph)]
pub struct JsDiGraph {
    graph: GraphBase<Directed>,
}

impl JsDiGraph {
    pub fn new_from_graph(graph: Graph<Node, Edge, Directed, IndexType>) -> Self {
        Self {
            graph: GraphBase::<Directed>::new_from_graph(graph),
        }
    }

    pub fn graph(&self) -> &Graph<Node, Edge, Directed, IndexType> {
        &self.graph.graph()
    }
}

#[wasm_bindgen(js_class = DiGraph)]
impl JsDiGraph {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            graph: GraphBase::<Directed>::new(),
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
        self.graph.add_node(value)
    }

    #[wasm_bindgen(js_name = nodeWeight)]
    pub fn node_weight(&self, a: usize) -> Result<JsValue, JsValue> {
        self.graph.node_weight(a)
    }

    #[wasm_bindgen(js_name = addEdge)]
    pub fn add_edge(&mut self, a: usize, b: usize, value: JsValue) -> usize {
        self.graph.add_edge(a, b, value)
    }

    #[wasm_bindgen(js_name = edgeWeight)]
    pub fn edge_weight(&mut self, e: usize) -> Result<JsValue, JsValue> {
        self.graph.edge_weight(e)
    }

    #[wasm_bindgen(js_name = edgeEndpoints)]
    pub fn edge_endpoints(&self, e: usize) -> Result<Array, JsValue> {
        self.graph.edge_endpoints(e)
    }

    #[wasm_bindgen(js_name = removeNode)]
    pub fn remove_node(&mut self, a: usize) -> Result<JsValue, JsValue> {
        self.graph.remove_node(a)
    }

    #[wasm_bindgen(js_name = removeEdge)]
    pub fn remove_edge(&mut self, e: usize) -> Result<JsValue, JsValue> {
        self.graph.remove_edge(e)
    }

    pub fn neighbors(&self, a: usize) -> Array {
        self.graph.neighbors(a)
    }

    #[wasm_bindgen(js_name = neighborsDirected)]
    pub fn neighbors_directed(&self, a: usize, dir: usize) -> Array {
        self.graph.neighbors_directed(a, dir)
    }

    #[wasm_bindgen(js_name = neighborsUndirected)]
    pub fn neighbors_undirected(&self, a: usize) -> Array {
        self.graph.neighbors_undirected(a)
    }

    pub fn edges(&self, a: usize) -> Array {
        self.graph.edges(a)
    }

    #[wasm_bindgen(js_name = containsEdge)]
    pub fn contains_edge(&self, a: usize, b: usize) -> bool {
        self.graph.contains_edge(a, b)
    }

    #[wasm_bindgen(js_name = findEdge)]
    pub fn find_edge(&self, a: usize, b: usize) -> Result<usize, JsValue> {
        self.graph.find_edge(a, b)
    }

    pub fn externals(&self, dir: usize) -> Array {
        self.graph.externals(dir)
    }

    #[wasm_bindgen(js_name = nodeIndices)]
    pub fn node_indices(&self) -> Array {
        self.graph.node_indices()
    }

    #[wasm_bindgen(js_name = edgeIndices)]
    pub fn edge_indices(&self) -> Array {
        self.graph.edge_indices()
    }

    pub fn map(&self, node_map: &Function, edge_map: &Function) -> Self {
        Self {
            graph: self.graph.map(node_map, edge_map),
        }
    }

    #[wasm_bindgen(js_name = filterMap)]
    pub fn filter_map(&self, node_map: &Function, edge_map: &Function) -> Self {
        Self {
            graph: self.graph.filter_map(node_map, edge_map),
        }
    }
}
