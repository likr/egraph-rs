use std::collections::VecDeque;

use super::CsrGraph;
use crate::error::{CoreError, CoreResult};
use crate::visit_stamp::VisitStamp;

impl<W> CsrGraph<W> {
    /// Returns the sorted original vertex IDs belonging to the largest (weakly) connected component.
    ///
    /// Ties are broken deterministically in favor of the component containing the smallest node ID.
    pub fn largest_connected_component_nodes(&self) -> Vec<u32> {
        if self.num_nodes == 0 {
            return Vec::new();
        }

        // For directed graphs, build lightweight incoming adjacency so BFS visits weak components.
        let reverse_adj = if self.directed {
            Some(self.build_reverse_adjacency())
        } else {
            None
        };

        let mut visited = VisitStamp::new(self.num_nodes);
        let mut queue = VecDeque::new();
        let mut best: Vec<u32> = Vec::new();
        let mut current: Vec<u32> = Vec::new();

        for start in 0..self.num_nodes {
            if !visited.visit(start) {
                continue;
            }
            current.clear();
            queue.push_back(start as u32);
            while let Some(u) = queue.pop_front() {
                current.push(u);
                for &v in self.neighbors(u as usize) {
                    if visited.visit(v as usize) {
                        queue.push_back(v);
                    }
                }
                if let Some((rev_indptr, rev_indices)) = &reverse_adj {
                    let rs = rev_indptr[u as usize] as usize;
                    let re = rev_indptr[u as usize + 1] as usize;
                    for &v in &rev_indices[rs..re] {
                        if visited.visit(v as usize) {
                            queue.push_back(v);
                        }
                    }
                }
            }
            if current.len() > best.len() {
                std::mem::swap(&mut best, &mut current);
            }
        }

        best.sort_unstable();
        best
    }

    /// Extracts the induced subgraph on `nodes` (with compact `0..nodes.len()` vertex IDs).
    ///
    /// Returns an error if any entry in `nodes` is out of range or duplicated.
    pub fn induced_subgraph(&self, nodes: &[u32]) -> CoreResult<Self>
    where
        W: Clone,
    {
        let mut remap = vec![u32::MAX; self.num_nodes];
        for (new_idx, &old_u) in nodes.iter().enumerate() {
            let old = old_u as usize;
            if old >= self.num_nodes {
                return Err(CoreError::index_out_of_bounds(
                    old,
                    self.num_nodes,
                    "CsrGraph::induced_subgraph",
                ));
            }
            if remap[old] != u32::MAX {
                return Err(CoreError::invalid_input(format!(
                    "duplicate vertex {old_u} in induced_subgraph node subset"
                )));
            }
            remap[old] = new_idx as u32;
        }

        let is_sorted = nodes.windows(2).all(|w| w[0] < w[1]);
        let sub_n = nodes.len();
        let mut out_indptr = Vec::with_capacity(sub_n + 1);
        out_indptr.push(0u32);
        let mut out_indices = Vec::new();
        let mut out_weights = self.weights.as_ref().map(|_| Vec::new());

        for &old_u in nodes {
            let start = self.indptr[old_u as usize] as usize;
            let end = self.indptr[old_u as usize + 1] as usize;
            let row_start_pos = out_indices.len();

            if let Some(w_src) = &self.weights {
                let w_dst = out_weights.as_mut().unwrap();
                for pos in start..end {
                    let mapped_v = remap[self.indices[pos] as usize];
                    if mapped_v != u32::MAX {
                        out_indices.push(mapped_v);
                        w_dst.push(w_src[pos].clone());
                    }
                }
                if !is_sorted && out_indices.len() - row_start_pos > 1 {
                    let mut pairs: Vec<(u32, W)> = out_indices
                        .drain(row_start_pos..)
                        .zip(w_dst.drain(row_start_pos..))
                        .collect();
                    pairs.sort_unstable_by_key(|(v, _)| *v);
                    for (v, w) in pairs {
                        out_indices.push(v);
                        w_dst.push(w);
                    }
                }
            } else {
                for pos in start..end {
                    let mapped_v = remap[self.indices[pos] as usize];
                    if mapped_v != u32::MAX {
                        out_indices.push(mapped_v);
                    }
                }
                if !is_sorted && out_indices.len() - row_start_pos > 1 {
                    out_indices[row_start_pos..].sort_unstable();
                }
            }

            let ptr = u32::try_from(out_indices.len())
                .map_err(|_| CoreError::invalid_csr("subgraph edge count exceeds u32::MAX"))?;
            out_indptr.push(ptr);
        }

        Ok(Self {
            num_nodes: sub_n,
            directed: self.directed,
            indptr: out_indptr,
            indices: out_indices,
            weights: out_weights,
        })
    }

    /// Extracts the largest (weakly) connected component as `(subgraph, original_node_ids)`.
    ///
    /// `original_node_ids[new_id]` maps each compact vertex index in the returned subgraph back
    /// to its original vertex index in `self`, sorted in ascending order.
    pub fn largest_connected_component(&self) -> (Self, Vec<u32>)
    where
        W: Clone,
    {
        let lcc_nodes = self.largest_connected_component_nodes();
        if lcc_nodes.len() == self.num_nodes {
            return (self.clone(), lcc_nodes);
        }
        let subgraph = self
            .induced_subgraph(&lcc_nodes)
            .expect("LCC nodes are strictly valid and unique");
        (subgraph, lcc_nodes)
    }

    fn build_reverse_adjacency(&self) -> (Vec<u32>, Vec<u32>) {
        let mut counts = vec![0u32; self.num_nodes];
        for &v in &self.indices {
            counts[v as usize] += 1;
        }
        let mut rev_indptr = Vec::with_capacity(self.num_nodes + 1);
        rev_indptr.push(0u32);
        let mut acc = 0u32;
        for c in &counts {
            acc += *c;
            rev_indptr.push(acc);
        }
        let mut cursor = rev_indptr[..self.num_nodes].to_vec();
        let mut rev_indices = vec![0u32; self.indices.len()];
        for u in 0..self.num_nodes {
            let start = self.indptr[u] as usize;
            let end = self.indptr[u + 1] as usize;
            for &v in &self.indices[start..end] {
                let dst = cursor[v as usize] as usize;
                rev_indices[dst] = u as u32;
                cursor[v as usize] += 1;
            }
        }
        (rev_indptr, rev_indices)
    }
}
