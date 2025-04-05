//! Directed graph implementation for WebAssembly.
//!
//! This module provides a WebAssembly binding for a directed graph data structure,
//! exposing petgraph's Graph with Directed edge type to JavaScript.

use crate::graph::base::GraphBase;
use crate::graph::types::{Edge, IndexType, Node};
use js_sys::{Array, Function};
use petgraph::graph::Graph;
use petgraph::Directed;
use wasm_bindgen::prelude::*;

/// WebAssembly binding for a directed graph.
///
/// This struct provides a JavaScript interface to a directed graph implementation
/// based on petgraph's Graph<Node, Edge, Directed, IndexType>.
#[wasm_bindgen(js_name = DiGraph)]
pub struct JsDiGraph {
    graph: GraphBase<Directed>,
}

impl Default for JsDiGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl JsDiGraph {
    pub fn new_from_graph(graph: Graph<Node, Edge, Directed, IndexType>) -> Self {
        Self {
            graph: GraphBase::<Directed>::new_from_graph(graph),
        }
    }

    pub fn graph(&self) -> &Graph<Node, Edge, Directed, IndexType> {
        self.graph.graph()
    }
}

#[wasm_bindgen(js_class = DiGraph)]
impl JsDiGraph {
    /// Creates a new empty directed graph.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            graph: GraphBase::<Directed>::new(),
        }
    }

    /// Returns the number of nodes in the graph.
    #[wasm_bindgen(js_name = nodeCount)]
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Returns the number of edges in the graph.
    #[wasm_bindgen(js_name = edgeCount)]
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Adds a new node to the graph with the given value.
    ///
    /// The value can be any JavaScript value.
    ///
    /// Returns the index of the newly added node.
    #[wasm_bindgen(js_name = addNode)]
    pub fn add_node(&mut self, value: JsValue) -> usize {
        self.graph.add_node(value)
    }

    /// Returns the value associated with the node at the given index.
    ///
    /// Returns an error if the node index is invalid.
    #[wasm_bindgen(js_name = nodeWeight)]
    pub fn node_weight(&self, a: usize) -> Result<JsValue, JsValue> {
        self.graph.node_weight(a)
    }

    /// Adds a directed edge from node at index a to node at index b with the given value.
    ///
    /// The value can be any JavaScript value.
    ///
    /// Returns the index of the newly added edge.
    #[wasm_bindgen(js_name = addEdge)]
    pub fn add_edge(&mut self, a: usize, b: usize, value: JsValue) -> usize {
        self.graph.add_edge(a, b, value)
    }

    /// Returns the value associated with the edge at the given index.
    ///
    /// Returns an error if the edge index is invalid.
    #[wasm_bindgen(js_name = edgeWeight)]
    pub fn edge_weight(&mut self, e: usize) -> Result<JsValue, JsValue> {
        self.graph.edge_weight(e)
    }

    /// Returns the endpoints (source and target node indices) of the edge at the given index.
    ///
    /// Returns an array containing two node indices [source, target].
    /// Returns an error if the edge index is invalid.
    #[wasm_bindgen(js_name = edgeEndpoints)]
    pub fn edge_endpoints(&self, e: usize) -> Result<Array, JsValue> {
        self.graph.edge_endpoints(e)
    }

    /// Removes the node at the given index from the graph.
    ///
    /// Returns the node value if successful, or an error if the node index is invalid.
    /// Note that removing a node will invalidate any edge indices that pointed to edges
    /// connected to the removed node.
    #[wasm_bindgen(js_name = removeNode)]
    pub fn remove_node(&mut self, a: usize) -> Result<JsValue, JsValue> {
        self.graph.remove_node(a)
    }

    /// Removes the edge at the given index from the graph.
    ///
    /// Returns the edge value if successful, or an error if the edge index is invalid.
    #[wasm_bindgen(js_name = removeEdge)]
    pub fn remove_edge(&mut self, e: usize) -> Result<JsValue, JsValue> {
        self.graph.remove_edge(e)
    }

    /// Returns an array of node indices that are neighbors of the node at the given index.
    ///
    /// For a directed graph, this includes both successor and predecessor nodes.
    pub fn neighbors(&self, a: usize) -> Array {
        self.graph.neighbors(a)
    }

    /// Returns an array of node indices that are directed neighbors of the node at the given index.
    ///
    /// The direction is specified by the dir parameter:
    /// - 0: outgoing neighbors (successors of node a)
    /// - 1: incoming neighbors (predecessors of node a)
    #[wasm_bindgen(js_name = neighborsDirected)]
    pub fn neighbors_directed(&self, a: usize, dir: usize) -> Array {
        self.graph.neighbors_directed(a, dir)
    }

    /// Returns an array of node indices that are undirected neighbors of the node at the given index.
    ///
    /// This includes all nodes connected by an edge in either direction.
    #[wasm_bindgen(js_name = neighborsUndirected)]
    pub fn neighbors_undirected(&self, a: usize) -> Array {
        self.graph.neighbors_undirected(a)
    }

    /// Returns an array of edge values for all edges connected to the node at the given index.
    pub fn edges(&self, a: usize) -> Array {
        self.graph.edges(a)
    }

    /// Returns true if there is a directed edge from node at index a to node at index b.
    #[wasm_bindgen(js_name = containsEdge)]
    pub fn contains_edge(&self, a: usize, b: usize) -> bool {
        self.graph.contains_edge(a, b)
    }

    /// Returns the index of the directed edge from node at index a to node at index b.
    ///
    /// Returns an error if no such edge exists.
    #[wasm_bindgen(js_name = findEdge)]
    pub fn find_edge(&self, a: usize, b: usize) -> Result<usize, JsValue> {
        self.graph.find_edge(a, b)
    }

    /// Returns an array of node indices that are external (have no neighbors in the specified direction).
    ///
    /// The direction is specified by the dir parameter:
    /// - 0: source nodes (nodes with no outgoing edges)
    /// - 1: sink nodes (nodes with no incoming edges)
    pub fn externals(&self, dir: usize) -> Array {
        self.graph.externals(dir)
    }

    /// Returns an array of all node indices in the graph.
    #[wasm_bindgen(js_name = nodeIndices)]
    pub fn node_indices(&self) -> Array {
        self.graph.node_indices()
    }

    /// Returns an array of all edge indices in the graph.
    #[wasm_bindgen(js_name = edgeIndices)]
    pub fn edge_indices(&self) -> Array {
        self.graph.edge_indices()
    }

    /// Creates a new graph by applying mapping functions to all nodes and edges.
    ///
    /// @param {Function} node_map - Function that takes (node_index, node_value) and returns a new node value
    /// @param {Function} edge_map - Function that takes (edge_index, edge_value) and returns a new edge value
    /// @returns {DiGraph} A new graph with mapped values
    pub fn map(&self, node_map: &Function, edge_map: &Function) -> Self {
        Self {
            graph: self.graph.map(node_map, edge_map),
        }
    }

    /// Creates a new graph by selectively mapping and filtering nodes and edges.
    ///
    /// Similar to map(), but allows removing nodes and edges by returning null.
    ///
    /// @param {Function} node_map - Function that takes (node_index, node_value) and returns a new node value or null to remove the node
    /// @param {Function} edge_map - Function that takes (edge_index, edge_value) and returns a new edge value or null to remove the edge
    /// @returns {DiGraph} A new filtered graph
    #[wasm_bindgen(js_name = filterMap)]
    pub fn filter_map(&self, node_map: &Function, edge_map: &Function) -> Self {
        Self {
            graph: self.graph.filter_map(node_map, edge_map),
        }
    }
}
