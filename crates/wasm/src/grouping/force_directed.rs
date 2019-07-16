use super::{GroupLinkObject, GroupNodeObject, GroupObject};
use crate::layout::force_directed::simulation::JsSimulationBuilder;
use egraph::grouping::force_directed::ForceDirectedGrouping;
use egraph::grouping::{GroupLink, GroupNode};
use egraph::Graph;
use egraph_wasm_adapter::{JsGraph, JsGraphAdapter};
use js_sys::Function;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = ForceDirectedGrouping)]
pub struct JsForceDirectedGrouping {
    grouping: ForceDirectedGrouping<JsGraph, JsGraphAdapter>,
}

#[wasm_bindgen(js_class = ForceDirectedGrouping)]
impl JsForceDirectedGrouping {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsForceDirectedGrouping {
        JsForceDirectedGrouping {
            grouping: ForceDirectedGrouping::new(),
        }
    }

    pub fn call(
        &self,
        graph: JsGraph,
        builder: JsSimulationBuilder,
        new_graph: Function,
    ) -> JsValue {
        let graph = JsGraphAdapter::new(graph);
        let f: Box<dyn Fn(&Vec<GroupNode>, &Vec<GroupLink>) -> JsGraphAdapter> =
            Box::new(move |nodes, links| {
                let this = JsValue::NULL;
                let graph: JsGraph = new_graph.call0(&this).ok().unwrap().into();
                for node in nodes {
                    let obj = GroupNodeObject {
                        id: node.id,
                        weight: node.weight as f64,
                    };
                    graph.add_node(node.id, JsValue::from_serde(&obj).unwrap())
                }
                for link in links {
                    let obj = GroupLinkObject {
                        source: link.source,
                        target: link.target,
                        weight: link.weight as f64,
                    };
                    graph.add_edge(link.source, link.target, JsValue::from_serde(&obj).unwrap())
                }
                JsGraphAdapter::new(graph)
            });
        let result = self
            .grouping
            .call(&graph, builder.builder(), &f)
            .iter()
            .map(|(&i, g)| {
                (
                    i,
                    GroupObject {
                        shape: "circle".into(),
                        x: g.x as f64,
                        y: g.y as f64,
                        width: g.width as f64,
                        height: g.height as f64,
                    },
                )
            })
            .collect::<HashMap<_, _>>();
        JsValue::from_serde(&result).unwrap()
    }

    #[wasm_bindgen(setter = group)]
    pub fn set_group(&mut self, f: &Function) {
        let f = f.clone();
        self.grouping.group = Box::new(move |graph, u| {
            let this = JsValue::NULL;
            let graph = graph.data();
            let u = JsValue::from_f64(u as f64);
            f.call2(&this, &graph, &u).ok().unwrap().as_f64().unwrap() as usize
        });
    }

    #[wasm_bindgen(setter = nodeWeight)]
    pub fn set_node_weight(&mut self, f: &Function) {
        let f = f.clone();
        self.grouping.node_weight = Box::new(move |graph, u| {
            let this = JsValue::NULL;
            let graph = graph.data();
            let u = JsValue::from_f64(u as f64);
            f.call2(&this, &graph, &u).ok().unwrap().as_f64().unwrap() as f32
        });
    }

    #[wasm_bindgen(setter = linkWeight)]
    pub fn set_link_weight(&mut self, f: &Function) {
        let f = f.clone();
        self.grouping.link_weight = Box::new(move |graph, u, v| {
            let this = JsValue::NULL;
            let graph = graph.data();
            let u = JsValue::from_f64(u as f64);
            let v = JsValue::from_f64(v as f64);
            f.call3(&this, &graph, &u, &v)
                .ok()
                .unwrap()
                .as_f64()
                .unwrap() as f32
        });
    }
}
