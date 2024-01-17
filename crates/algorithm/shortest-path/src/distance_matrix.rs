use ndarray::{concatenate, prelude::*};
use petgraph::visit::IntoNodeIdentifiers;
use std::{collections::HashMap, hash::Hash};

pub trait DistanceMatrix<N, S> {
    fn get(&self, u: N, v: N) -> Option<S>;

    fn set(&mut self, u: N, v: N, d: S) -> Option<()>;

    fn get_by_index(&self, i: usize, j: usize) -> S;

    fn set_by_index(&mut self, i: usize, j: usize, d: S);

    fn shape(&self) -> (usize, usize);

    fn row_index(&self, u: N) -> Option<usize>;

    fn col_index(&self, u: N) -> Option<usize>;
}

pub struct FullDistanceMatrix<N, S> {
    indices: Vec<N>,
    index_map: HashMap<N, usize>,
    d: Array2<S>,
}

impl<N, S> DistanceMatrix<N, S> for FullDistanceMatrix<N, S>
where
    N: Eq + Hash,
    S: NdFloat,
{
    fn get(&self, u: N, v: N) -> Option<S> {
        self.index(u, v).map(|(i, j)| self.d[[i, j]])
    }

    fn set(&mut self, u: N, v: N, d: S) -> Option<()> {
        self.index(u, v).map(|(i, j)| self.d[[i, j]] = d)
    }

    fn get_by_index(&self, i: usize, j: usize) -> S {
        self.d[[i, j]]
    }

    fn set_by_index(&mut self, i: usize, j: usize, d: S) {
        self.d[[i, j]] = d;
    }

    fn shape(&self) -> (usize, usize) {
        (self.indices.len(), self.indices.len())
    }

    fn row_index(&self, u: N) -> Option<usize> {
        self.index_map.get(&u).copied()
    }

    fn col_index(&self, u: N) -> Option<usize> {
        self.index_map.get(&u).copied()
    }
}

impl<N, S> FullDistanceMatrix<N, S>
where
    N: Eq + Hash,
    S: NdFloat,
{
    pub fn new<G>(graph: G) -> Self
    where
        G: IntoNodeIdentifiers,
        G::NodeId: Into<N>,
        N: Copy,
    {
        let indices = graph
            .node_identifiers()
            .map(|u| u.into())
            .collect::<Vec<_>>();
        let mut index_map = HashMap::new();
        for (i, &u) in indices.iter().enumerate() {
            index_map.insert(u, i);
        }
        let n = indices.len();
        Self {
            indices,
            index_map,
            d: Array::from_elem((n, n), S::infinity()),
        }
    }

    fn index(&self, u: N, v: N) -> Option<(usize, usize)> {
        self.index_map
            .get(&u)
            .zip(self.index_map.get(&v))
            .map(|(&i, &j)| (i, j))
    }
}

pub struct SubDistanceMatrix<N, S> {
    row_indices: Vec<N>,
    row_index_map: HashMap<N, usize>,
    col_indices: Vec<N>,
    col_index_map: HashMap<N, usize>,
    d: Array2<S>,
}

impl<N, S> DistanceMatrix<N, S> for SubDistanceMatrix<N, S>
where
    N: Eq + Hash,
    S: NdFloat,
{
    fn get(&self, u: N, v: N) -> Option<S> {
        self.index(u, v).map(|(i, j)| self.d[[i, j]])
    }

    fn set(&mut self, u: N, v: N, d: S) -> Option<()> {
        self.index(u, v).map(|(i, j)| self.d[[i, j]] = d)
    }

    fn get_by_index(&self, i: usize, j: usize) -> S {
        self.d[[i, j]]
    }

    fn set_by_index(&mut self, i: usize, j: usize, d: S) {
        self.d[[i, j]] = d;
    }

    fn shape(&self) -> (usize, usize) {
        (self.row_indices.len(), self.col_indices.len())
    }

    fn row_index(&self, u: N) -> Option<usize> {
        self.row_index_map.get(&u).copied()
    }

    fn col_index(&self, u: N) -> Option<usize> {
        self.col_index_map.get(&u).copied()
    }
}

impl<N, S> SubDistanceMatrix<N, S>
where
    N: Eq + Hash,
    S: NdFloat,
{
    pub fn empty<G>(graph: G) -> Self
    where
        G: IntoNodeIdentifiers,
        G::NodeId: Into<N>,
        N: Copy,
    {
        Self::new(graph, &[])
    }

    pub fn new<G>(graph: G, sources: &[G::NodeId]) -> Self
    where
        G: IntoNodeIdentifiers,
        G::NodeId: Into<N>,
        N: Copy,
    {
        let row_indices = graph
            .node_identifiers()
            .map(|u| u.into())
            .collect::<Vec<_>>();
        let mut row_index_map = HashMap::new();
        for (i, &u) in row_indices.iter().enumerate() {
            row_index_map.insert(u, i);
        }
        let col_indices = sources.iter().map(|&u| u.into()).collect::<Vec<_>>();
        let mut col_index_map = HashMap::new();
        for (i, &u) in col_indices.iter().enumerate() {
            col_index_map.insert(u, i);
        }
        let d = Array::from_elem((row_indices.len(), col_indices.len()), S::infinity());
        Self {
            row_indices,
            row_index_map,
            col_indices,
            col_index_map,
            d,
        }
    }

    pub fn push(&mut self, u: N)
    where
        N: Copy,
    {
        self.col_index_map.insert(u, self.col_indices.len());
        self.col_indices.push(u);
        concatenate![
            Axis(1),
            Array::from_elem((self.row_indices.len(), 1), S::infinity())
        ];
    }

    fn index(&self, u: N, v: N) -> Option<(usize, usize)> {
        self.row_index_map
            .get(&u)
            .zip(self.col_index_map.get(&v))
            .map(|(&i, &j)| (i, j))
    }
}
