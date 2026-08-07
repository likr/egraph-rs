//! Distance matrix bindings for WebAssembly.
//!
//! This module provides WebAssembly bindings for distance matrices and kernels,
//! enabling their use in layout algorithms like tsNET.

use crate::graph::JsGraph;
use crate::rng::JsRng;
use js_sys::Function;
use petgraph::graph::{node_index, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph_algorithm_shortest_path::{
    all_sources_dijkstra, FullDistanceMatrix, SubDistanceMatrix,
};
use petgraph_distance::{Distance, GaussianKernel, KernelDistance, SparseSymmetricMatrix};
use petgraph_linalg_diffusion_kernel::{
    DiffusionKernel, DiffusionKernelBuilder, NegLogSimDistance, NegLogSimDistanceBuilder,
};
use petgraph_linalg_embedding_distance::EmbeddingDistanceMatrix;
use wasm_bindgen::prelude::*;

/// Helper enum representing different types of distance matrices in WASM bindings.
#[derive(Clone)]
pub enum InnerDistanceMatrix {
    Full(FullDistanceMatrix<NodeIndex<u32>, f32>),
    Sub(SubDistanceMatrix<NodeIndex<u32>, f32>),
    Diffusion(NegLogSimDistance<NodeIndex<u32>, f32, DiffusionKernel<f32>>),
    Embedding(EmbeddingDistanceMatrix<NodeIndex<u32>, f32>),
    Kernel(Box<KernelDistance<GaussianKernel<NodeIndex<u32>, InnerDistanceMatrix, f32>>>),
}

impl Distance<NodeIndex<u32>, f32> for InnerDistanceMatrix {
    fn get(&self, u: NodeIndex<u32>, v: NodeIndex<u32>) -> Option<f32> {
        match self {
            Self::Full(d) => Distance::get(d, u, v),
            Self::Sub(d) => Distance::get(d, u, v),
            Self::Diffusion(d) => Distance::get(d, u, v),
            Self::Embedding(d) => Distance::get(d, u, v),
            Self::Kernel(d) => Distance::get(d.as_ref(), u, v),
        }
    }

    fn get_by_index(&self, i: usize, j: usize) -> f32 {
        match self {
            Self::Full(d) => Distance::get_by_index(d, i, j),
            Self::Sub(d) => Distance::get_by_index(d, i, j),
            Self::Diffusion(d) => Distance::get_by_index(d, i, j),
            Self::Embedding(d) => Distance::get_by_index(d, i, j),
            Self::Kernel(d) => Distance::<NodeIndex<u32>, f32>::get_by_index(d.as_ref(), i, j),
        }
    }

    fn shape(&self) -> (usize, usize) {
        match self {
            Self::Full(d) => Distance::shape(d),
            Self::Sub(d) => Distance::shape(d),
            Self::Diffusion(d) => Distance::shape(d),
            Self::Embedding(d) => Distance::shape(d),
            Self::Kernel(d) => Distance::<NodeIndex<u32>, f32>::shape(d.as_ref()),
        }
    }

    fn row_index(&self, u: NodeIndex<u32>) -> Option<usize> {
        match self {
            Self::Full(d) => Distance::row_index(d, u),
            Self::Sub(d) => Distance::row_index(d, u),
            Self::Diffusion(d) => Distance::row_index(d, u),
            Self::Embedding(d) => Distance::row_index(d, u),
            Self::Kernel(d) => Distance::row_index(d.as_ref(), u),
        }
    }

    fn col_index(&self, u: NodeIndex<u32>) -> Option<usize> {
        match self {
            Self::Full(d) => Distance::col_index(d, u),
            Self::Sub(d) => Distance::col_index(d, u),
            Self::Diffusion(d) => Distance::col_index(d, u),
            Self::Embedding(d) => Distance::col_index(d, u),
            Self::Kernel(d) => Distance::col_index(d.as_ref(), u),
        }
    }
}

/// WebAssembly binding for querying diffusion kernel matrix elements
#[wasm_bindgen(js_name = "DiffusionKernel")]
pub struct JsDiffusionKernel {
    pub(crate) kernel: DiffusionKernel<f32>,
}

#[wasm_bindgen(js_class = "DiffusionKernel")]
impl JsDiffusionKernel {
    #[wasm_bindgen(constructor)]
    pub fn new(
        graph: &JsGraph,
        length: &Function,
        t: f32,
        degree: usize,
        _num_vectors: usize,
        rng: &mut JsRng,
    ) -> Result<JsDiffusionKernel, JsError> {
        let mut length_map = std::collections::HashMap::new();
        for e in graph.graph().edge_indices() {
            let c = length
                .call1(&JsValue::null(), &JsValue::from_f64(e.index() as f64))
                .unwrap()
                .as_f64()
                .unwrap() as f32;
            length_map.insert(e, c);
        }
        let laplacian =
            SparseSymmetricMatrix::standard_laplacian(graph.graph(), |e| length_map[&e.id()]);
        let kernel = DiffusionKernelBuilder::new(&laplacian, t, degree)
            .build(rng.get_mut())
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(JsDiffusionKernel { kernel })
    }

    #[wasm_bindgen(js_name = "newWithLambdaMax")]
    pub fn new_with_lambda_max(
        graph: &JsGraph,
        length: &Function,
        t: f32,
        degree: usize,
        lambda_max: f32,
        _num_vectors: usize,
        rng: &mut JsRng,
    ) -> Result<JsDiffusionKernel, JsError> {
        let mut length_map = std::collections::HashMap::new();
        for e in graph.graph().edge_indices() {
            let c = length
                .call1(&JsValue::null(), &JsValue::from_f64(e.index() as f64))
                .unwrap()
                .as_f64()
                .unwrap() as f32;
            length_map.insert(e, c);
        }
        let laplacian =
            SparseSymmetricMatrix::standard_laplacian(graph.graph(), |e| length_map[&e.id()]);
        let kernel = DiffusionKernelBuilder::new(&laplacian, t, degree)
            .lambda_max(lambda_max)
            .build(rng.get_mut())
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(JsDiffusionKernel { kernel })
    }

    pub fn get(&self, i: usize, j: usize) -> f32 {
        petgraph_distance::Kernel::get(&self.kernel, i, j)
    }

    #[wasm_bindgen]
    pub fn n(&self) -> usize {
        petgraph_distance::Kernel::n(&self.kernel)
    }
}

/// WebAssembly binding for representing distance matrices.
#[wasm_bindgen(js_name = "DistanceMatrix")]
#[derive(Clone)]
pub struct JsDistanceMatrix {
    pub(crate) inner: InnerDistanceMatrix,
}

#[wasm_bindgen(js_class = "DistanceMatrix")]
impl JsDistanceMatrix {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph) -> JsDistanceMatrix {
        JsDistanceMatrix {
            inner: InnerDistanceMatrix::Full(FullDistanceMatrix::new(graph.graph())),
        }
    }

    #[wasm_bindgen(js_name = "allSourcesDijkstra")]
    pub fn all_sources_dijkstra(graph: &JsGraph, length: &Function) -> JsDistanceMatrix {
        let mut length_map = std::collections::HashMap::new();
        for e in graph.graph().edge_indices() {
            let c = length
                .call1(&JsValue::null(), &JsValue::from_f64(e.index() as f64))
                .unwrap()
                .as_f64()
                .unwrap() as f32;
            length_map.insert(e, c);
        }
        let full_d = all_sources_dijkstra(graph.graph(), |e| length_map[&e.id()]);
        JsDistanceMatrix {
            inner: InnerDistanceMatrix::Full(full_d),
        }
    }

    pub fn get(&self, u: usize, v: usize) -> Option<f32> {
        Distance::get(&self.inner, node_index(u), node_index(v))
    }

    #[wasm_bindgen(js_name = "diffusion")]
    pub fn diffusion(
        graph: &JsGraph,
        kernel: &JsDiffusionKernel,
        _min_dist: f32, // Ignored, kept for API compatibility
    ) -> Result<JsDistanceMatrix, JsError> {
        let matrix = NegLogSimDistanceBuilder::new(kernel.kernel.clone())
            .build(graph.graph())
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(JsDistanceMatrix {
            inner: InnerDistanceMatrix::Diffusion(matrix),
        })
    }

    #[wasm_bindgen(js_name = "embedding")]
    pub fn embedding(
        graph: &JsGraph,
        coordinates: &[f32],
        d: usize,
        min_dist: f32,
    ) -> Result<JsDistanceMatrix, JsError> {
        let n = graph.graph().node_count();
        if coordinates.len() != n * d {
            return Err(JsError::new(&format!(
                "coordinates length must be n * d (n: {}, d: {}, expected: {}, got: {})",
                n,
                d,
                n * d,
                coordinates.len()
            )));
        }
        let array = ndarray::Array2::from_shape_vec((n, d), coordinates.to_vec())
            .map_err(|e| JsError::new(&format!("Failed to create ndarray shape: {:?}", e)))?;
        let matrix = EmbeddingDistanceMatrix::new(graph.graph(), array, min_dist);
        Ok(JsDistanceMatrix {
            inner: InnerDistanceMatrix::Embedding(matrix),
        })
    }

    #[wasm_bindgen(js_name = "kernel")]
    pub fn kernel(distance_matrix: &JsDistanceMatrix, gamma: f32) -> JsDistanceMatrix {
        let kernel = GaussianKernel::new(distance_matrix.inner.clone(), gamma);
        let matrix = KernelDistance::new(kernel);
        JsDistanceMatrix {
            inner: InnerDistanceMatrix::Kernel(Box::new(matrix)),
        }
    }
}
