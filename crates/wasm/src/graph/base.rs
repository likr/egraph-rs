//! Base graph implementation shared by both directed and undirected graphs.
//!
//! This module provides the core graph functionality that is common to
//! both directed and undirected graph variants.

use crate::graph::types::{Edge, IndexType, Node};
use js_sys::{Array, Function};
use petgraph::{
    graph::{edge_index, node_index, Graph},
    Direction, EdgeType,
};
use wasm_bindgen::prelude::*;

/// Base graph implementation that can be parameterized by edge type.
///
/// This struct wraps a petgraph Graph and provides common functionality
/// for both directed and undirected graph variants.
pub struct GraphBase<Ty: EdgeType> {
    pub(crate) graph: Graph<Node, Edge, Ty, IndexType>,
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
            .cloned()
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
            .cloned()
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
