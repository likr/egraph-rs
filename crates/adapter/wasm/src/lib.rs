use egraph_adapter::{Graph, NodeIndex};
use js_sys::{try_iter, Reflect};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = Graph)]
    pub type JsGraph;

    #[wasm_bindgen(method, js_class = "Graph", js_name = "addNode")]
    pub fn add_node(this: &JsGraph, u: usize, data: JsValue);
    #[wasm_bindgen(method, js_class = "Graph", js_name = "addEdge")]
    pub fn add_edge(this: &JsGraph, u: usize, v: usize, data: JsValue);

    #[wasm_bindgen(method, js_class = "Graph", js_name = "node")]
    fn node(this: &JsGraph, u: usize) -> JsValue;
    #[wasm_bindgen(method, js_class = "Graph", js_name = "edge")]
    fn edge(this: &JsGraph, u: usize, v: usize) -> JsValue;
    #[wasm_bindgen(method, js_class = "Graph", js_name = "nodes")]
    fn nodes(this: &JsGraph) -> js_sys::Iterator;
    #[wasm_bindgen(method, js_class = "Graph", js_name = "edges")]
    fn edges(this: &JsGraph) -> js_sys::Iterator;
    #[wasm_bindgen(method, js_class = "Graph", js_name = "outNodes")]
    fn out_nodes(this: &JsGraph, u: usize) -> js_sys::Iterator;
    #[wasm_bindgen(method, js_class = "Graph", js_name = "inNodes")]
    fn in_nodes(this: &JsGraph, u: usize) -> js_sys::Iterator;
    #[wasm_bindgen(method, js_class = "Graph", js_name = "nodeCount")]
    fn node_count(this: &JsGraph) -> usize;
    #[wasm_bindgen(method, js_class = "Graph", js_name = "edgeCount")]
    fn edge_count(this: &JsGraph) -> usize;
    #[wasm_bindgen(method, js_class = "Graph", js_name = "outDegree")]
    fn out_degree(this: &JsGraph, u: usize) -> usize;
    #[wasm_bindgen(method, js_class = "Graph", js_name = "inDegree")]
    fn in_degree(this: &JsGraph, u: usize) -> usize;
}

pub struct JsGraphAdapter {
    graph: JsGraph,
}

impl JsGraphAdapter {
    pub fn new(graph: JsGraph) -> JsGraphAdapter {
        JsGraphAdapter { graph }
    }
}

impl Graph<JsGraph> for JsGraphAdapter {
    fn data(&self) -> &JsGraph {
        &self.graph
    }

    fn data_mut(&mut self) -> &mut JsGraph {
        &mut self.graph
    }

    fn nodes(&self) -> Box<dyn Iterator<Item = NodeIndex>> {
        Box::new(
            try_iter(&self.graph.nodes())
                .unwrap()
                .unwrap()
                .map(|obj| obj.unwrap().as_f64().unwrap() as usize),
        )
    }

    fn edges<'a>(&'a self) -> Box<dyn Iterator<Item = (NodeIndex, NodeIndex)> + 'a> {
        Box::new(try_iter(&self.graph.edges()).unwrap().unwrap().map(|obj| {
            let obj = obj.unwrap();
            let u = Reflect::get_u32(&obj, 0).ok().unwrap().as_f64().unwrap() as usize;
            let v = Reflect::get_u32(&obj, 1).ok().unwrap().as_f64().unwrap() as usize;
            (u, v)
        }))
    }

    fn out_nodes<'a>(&'a self, u: NodeIndex) -> Box<dyn Iterator<Item = NodeIndex> + 'a> {
        Box::new(
            try_iter(&self.graph.out_nodes(u))
                .unwrap()
                .unwrap()
                .map(|obj| obj.unwrap().as_f64().unwrap() as usize),
        )
    }

    fn in_nodes<'a>(&'a self, u: NodeIndex) -> Box<dyn Iterator<Item = NodeIndex> + 'a> {
        Box::new(
            try_iter(&self.graph.in_nodes(u))
                .unwrap()
                .unwrap()
                .map(|obj| obj.unwrap().as_f64().unwrap() as usize),
        )
    }

    fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    fn out_degree<'a>(&'a self, u: NodeIndex) -> usize {
        self.graph.out_degree(u)
    }

    fn in_degree<'a>(&'a self, u: NodeIndex) -> usize {
        self.graph.in_degree(u)
    }

    fn has_edge(&self, u: NodeIndex, v: NodeIndex) -> bool {
        !self.graph.edge(u, v).is_null()
    }
}
