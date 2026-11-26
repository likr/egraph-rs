//! Built-in dataset catalog and metadata lookup for `egraph-dataset`.

/// Static metadata describing a built-in embedded dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DatasetInfo {
    /// Canonical dataset identifier returned by [`list_datasets`].
    pub name: &'static str,
    /// Number of vertices in the dataset.
    pub num_nodes: usize,
    /// Number of undirected edges in the dataset.
    pub num_edges: usize,
    /// Source collection or benchmark origin.
    pub source: &'static str,
    /// Concise topological summary.
    pub description: &'static str,
}

const DATASET_NAMES: &[&str] = &[
    "1138_bus",
    "3elt",
    "USpowerGrid",
    "dwt_1005",
    "dwt_2680",
    "karate",
    "lesmis",
    "poli",
    "qh882",
];

const DATASET_CATALOG: &[DatasetInfo] = &[
    DatasetInfo {
        name: "1138_bus",
        num_nodes: 1138,
        num_edges: 1458,
        source: "SuiteSparse / HB",
        description: "Power system admittance bus network",
    },
    DatasetInfo {
        name: "3elt",
        num_nodes: 4720,
        num_edges: 13722,
        source: "SuiteSparse / AG-Monien",
        description: "2D finite-element airfoil triangular mesh",
    },
    DatasetInfo {
        name: "USpowerGrid",
        num_nodes: 4941,
        num_edges: 6594,
        source: "SuiteSparse / Pajek",
        description: "Western States power grid topology",
    },
    DatasetInfo {
        name: "dwt_1005",
        num_nodes: 1005,
        num_edges: 3808,
        source: "SuiteSparse / HB",
        description: "Structural engineering finite-element mesh",
    },
    DatasetInfo {
        name: "dwt_2680",
        num_nodes: 2680,
        num_edges: 11173,
        source: "SuiteSparse / HB",
        description: "Structural engineering finite-element mesh",
    },
    DatasetInfo {
        name: "karate",
        num_nodes: 34,
        num_edges: 78,
        source: "Zachary (1977)",
        description: "Zachary's Karate Club social network",
    },
    DatasetInfo {
        name: "lesmis",
        num_nodes: 77,
        num_edges: 254,
        source: "Knuth (1993)",
        description: "Les Miserables character co-occurrence network",
    },
    DatasetInfo {
        name: "poli",
        num_nodes: 3915,
        num_edges: 4119,
        source: "SuiteSparse / Grund",
        description: "Macroeconomic sparse structural model",
    },
    DatasetInfo {
        name: "qh882",
        num_nodes: 882,
        num_edges: 1533,
        source: "SuiteSparse / HB",
        description: "Quebec hydro-electric network Jacobian sparsity graph",
    },
];

/// Returns the canonical names of all built-in embedded datasets.
#[inline]
pub fn list_datasets() -> &'static [&'static str] {
    DATASET_NAMES
}

/// Returns static metadata records for all built-in embedded datasets.
#[inline]
pub fn dataset_catalog() -> &'static [DatasetInfo] {
    DATASET_CATALOG
}

/// Looks up static metadata for a built-in dataset by canonical name.
#[inline]
pub fn dataset_info(name: &str) -> Option<&'static DatasetInfo> {
    DATASET_CATALOG.iter().find(|entry| entry.name == name)
}
