use egraph_layered::Layout;
use js_sys::{Array, Object, Reflect};
use wasm_bindgen::prelude::*;

pub fn convert_to_object(layout: Layout) -> JsValue {
    let result = Object::new();

    let nodes = Array::new();
    for node in layout.nodes {
        let obj = Object::new();
        Reflect::set(&obj, &"x".into(), &node.x.into())
            .ok()
            .unwrap();
        Reflect::set(&obj, &"y".into(), &node.y.into())
            .ok()
            .unwrap();
        Reflect::set(&obj, &"width".into(), &node.width.into())
            .ok()
            .unwrap();
        Reflect::set(&obj, &"height".into(), &node.height.into())
            .ok()
            .unwrap();
        nodes.push(&obj);
    }

    let edges = Array::new();
    for edge in layout.edges {
        let obj = Object::new();
        let bends = Array::new();
        for point in edge.bends {
            let p = Object::new();
            Reflect::set(&p, &"x".into(), &point.x.into()).ok().unwrap();
            Reflect::set(&p, &"y".into(), &point.y.into()).ok().unwrap();
            bends.push(&p);
        }
        Reflect::set(&obj, &"bends".into(), &bends.into())
            .ok()
            .unwrap();
        edges.push(&obj);
    }

    Reflect::set(&result, &"nodes".into(), &nodes.into())
        .ok()
        .unwrap();
    Reflect::set(&result, &"edges".into(), &edges.into())
        .ok()
        .unwrap();

    result.into()
}
