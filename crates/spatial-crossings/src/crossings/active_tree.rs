use std::cmp::Ordering;

use super::cross_pred::CrossPred;
use super::geometry::Seg;

const NONE_U32: u32 = u32::MAX;

#[derive(Debug, Clone, Copy)]
pub(super) struct AvlNode {
    parent: u32,
    left: u32,
    right: u32,
    height: u16,
    size: u32,
    pub(super) edge_idx: usize,
}

/// Arena-backed AVL tree maintaining active non-vertical segments ordered by `(y(x), slope, id)`.
///
/// Storing `node_of_edge` lets segment removal at right endpoints run in $O(\log |A|)$ by handle
/// without relying on floating-point `y(x)` search, and swaps adjacent crossing segments in $O(1)$
/// by exchanging edge payloads.
pub(super) struct ActiveSegmentTree {
    pub(super) nodes: Vec<AvlNode>,
    free_list: Vec<u32>,
    root: u32,
    pub(super) node_of_edge: Vec<u32>,
}

impl ActiveSegmentTree {
    pub(super) fn new(num_edges: usize) -> Self {
        Self {
            nodes: Vec::new(),
            free_list: Vec::new(),
            root: NONE_U32,
            node_of_edge: vec![NONE_U32; num_edges],
        }
    }

    #[inline]
    pub(super) fn is_active(&self, edge_idx: usize) -> bool {
        self.node_of_edge[edge_idx] != NONE_U32
    }

    #[inline]
    fn height(&self, n: u32) -> u16 {
        if n == NONE_U32 {
            0
        } else {
            self.nodes[n as usize].height
        }
    }

    #[inline]
    fn size(&self, n: u32) -> u32 {
        if n == NONE_U32 {
            0
        } else {
            self.nodes[n as usize].size
        }
    }

    #[inline]
    fn update(&mut self, n: u32) {
        let l = self.nodes[n as usize].left;
        let r = self.nodes[n as usize].right;
        self.nodes[n as usize].height = 1 + self.height(l).max(self.height(r));
        self.nodes[n as usize].size = 1 + self.size(l) + self.size(r);
    }

    #[inline]
    fn balance_factor(&self, n: u32) -> i32 {
        self.height(self.nodes[n as usize].left) as i32
            - self.height(self.nodes[n as usize].right) as i32
    }

    fn alloc_node(&mut self, parent: u32, edge_idx: usize) -> u32 {
        let node = AvlNode {
            parent,
            left: NONE_U32,
            right: NONE_U32,
            height: 1,
            size: 1,
            edge_idx,
        };
        if let Some(id) = self.free_list.pop() {
            self.nodes[id as usize] = node;
            id
        } else {
            let id = self.nodes.len() as u32;
            self.nodes.push(node);
            id
        }
    }

    fn rotate_left(&mut self, x: u32) -> u32 {
        let y = self.nodes[x as usize].right;
        let t2 = self.nodes[y as usize].left;
        let p = self.nodes[x as usize].parent;

        self.nodes[y as usize].left = x;
        self.nodes[x as usize].parent = y;
        self.nodes[x as usize].right = t2;
        if t2 != NONE_U32 {
            self.nodes[t2 as usize].parent = x;
        }
        self.nodes[y as usize].parent = p;
        if p == NONE_U32 {
            self.root = y;
        } else if self.nodes[p as usize].left == x {
            self.nodes[p as usize].left = y;
        } else {
            self.nodes[p as usize].right = y;
        }
        self.update(x);
        self.update(y);
        y
    }

    fn rotate_right(&mut self, y: u32) -> u32 {
        let x = self.nodes[y as usize].left;
        let t2 = self.nodes[x as usize].right;
        let p = self.nodes[y as usize].parent;

        self.nodes[x as usize].right = y;
        self.nodes[y as usize].parent = x;
        self.nodes[y as usize].left = t2;
        if t2 != NONE_U32 {
            self.nodes[t2 as usize].parent = y;
        }
        self.nodes[x as usize].parent = p;
        if p == NONE_U32 {
            self.root = x;
        } else if self.nodes[p as usize].left == y {
            self.nodes[p as usize].left = x;
        } else {
            self.nodes[p as usize].right = x;
        }
        self.update(y);
        self.update(x);
        x
    }

    fn rebalance_up(&mut self, mut cur: u32) {
        while cur != NONE_U32 {
            self.update(cur);
            let bf = self.balance_factor(cur);
            if bf > 1 {
                let left = self.nodes[cur as usize].left;
                if self.balance_factor(left) < 0 {
                    self.rotate_left(left);
                }
                cur = self.rotate_right(cur);
            } else if bf < -1 {
                let right = self.nodes[cur as usize].right;
                if self.balance_factor(right) > 0 {
                    self.rotate_right(right);
                }
                cur = self.rotate_left(cur);
            }
            cur = self.nodes[cur as usize].parent;
        }
    }

    pub(super) fn insert<P: CrossPred>(
        &mut self,
        edge_idx: usize,
        sweep_x: f64,
        segs: &[Seg],
        pred: &P,
    ) {
        if self.node_of_edge[edge_idx] != NONE_U32 {
            return;
        }
        if self.root == NONE_U32 {
            let id = self.alloc_node(NONE_U32, edge_idx);
            self.root = id;
            self.node_of_edge[edge_idx] = id;
            return;
        }

        let s_new = &segs[edge_idx];
        let mut cur = self.root;
        loop {
            let cur_edge = self.nodes[cur as usize].edge_idx;
            let s_cur = &segs[cur_edge];
            let ord = pred.cmp_post_x(s_new, s_cur, sweep_x);
            if ord == Ordering::Less {
                let left = self.nodes[cur as usize].left;
                if left == NONE_U32 {
                    let id = self.alloc_node(cur, edge_idx);
                    self.nodes[cur as usize].left = id;
                    self.node_of_edge[edge_idx] = id;
                    self.rebalance_up(cur);
                    return;
                }
                cur = left;
            } else {
                let right = self.nodes[cur as usize].right;
                if right == NONE_U32 {
                    let id = self.alloc_node(cur, edge_idx);
                    self.nodes[cur as usize].right = id;
                    self.node_of_edge[edge_idx] = id;
                    self.rebalance_up(cur);
                    return;
                }
                cur = right;
            }
        }
    }

    fn leftmost(&self, mut n: u32) -> u32 {
        while n != NONE_U32 {
            let l = self.nodes[n as usize].left;
            if l == NONE_U32 {
                break;
            }
            n = l;
        }
        n
    }

    fn rightmost(&self, mut n: u32) -> u32 {
        while n != NONE_U32 {
            let r = self.nodes[n as usize].right;
            if r == NONE_U32 {
                break;
            }
            n = r;
        }
        n
    }

    fn prev_node(&self, mut n: u32) -> u32 {
        if n == NONE_U32 {
            return NONE_U32;
        }
        let l = self.nodes[n as usize].left;
        if l != NONE_U32 {
            return self.rightmost(l);
        }
        let mut p = self.nodes[n as usize].parent;
        while p != NONE_U32 && self.nodes[p as usize].left == n {
            n = p;
            p = self.nodes[p as usize].parent;
        }
        p
    }

    fn next_node(&self, mut n: u32) -> u32 {
        if n == NONE_U32 {
            return NONE_U32;
        }
        let r = self.nodes[n as usize].right;
        if r != NONE_U32 {
            return self.leftmost(r);
        }
        let mut p = self.nodes[n as usize].parent;
        while p != NONE_U32 && self.nodes[p as usize].right == n {
            n = p;
            p = self.nodes[p as usize].parent;
        }
        p
    }

    #[inline]
    pub(super) fn prev_edge(&self, edge_idx: usize) -> Option<usize> {
        let n = self.node_of_edge[edge_idx];
        if n == NONE_U32 {
            return None;
        }
        let p = self.prev_node(n);
        if p == NONE_U32 {
            None
        } else {
            Some(self.nodes[p as usize].edge_idx)
        }
    }

    #[inline]
    pub(super) fn next_edge(&self, edge_idx: usize) -> Option<usize> {
        let n = self.node_of_edge[edge_idx];
        if n == NONE_U32 {
            return None;
        }
        let nx = self.next_node(n);
        if nx == NONE_U32 {
            None
        } else {
            Some(self.nodes[nx as usize].edge_idx)
        }
    }

    pub(super) fn rank_of_edge(&self, edge_idx: usize) -> Option<usize> {
        let mut cur = self.node_of_edge[edge_idx];
        if cur == NONE_U32 {
            return None;
        }
        let mut rank = self.size(self.nodes[cur as usize].left) as usize;
        while self.nodes[cur as usize].parent != NONE_U32 {
            let p = self.nodes[cur as usize].parent;
            if self.nodes[p as usize].right == cur {
                rank += 1 + self.size(self.nodes[p as usize].left) as usize;
            }
            cur = p;
        }
        Some(rank)
    }

    pub(super) fn swap_adjacent_edges(&mut self, e1: usize, e2: usize) {
        let n1 = self.node_of_edge[e1];
        let n2 = self.node_of_edge[e2];
        debug_assert!(n1 != NONE_U32 && n2 != NONE_U32);
        self.nodes[n1 as usize].edge_idx = e2;
        self.nodes[n2 as usize].edge_idx = e1;
        self.node_of_edge[e1] = n2;
        self.node_of_edge[e2] = n1;
    }

    /// Removes `edge_idx` by handle and returns the newly adjacent `(prev, next)` edge indices.
    pub(super) fn remove_by_edge(&mut self, edge_idx: usize) -> (Option<usize>, Option<usize>) {
        let n = self.node_of_edge[edge_idx];
        if n == NONE_U32 {
            return (None, None);
        }
        let prev = self.prev_edge(edge_idx);
        let next = self.next_edge(edge_idx);

        // If node has two children, swap payload with its successor so deletion has <= 1 child.
        let mut target = n;
        if self.nodes[n as usize].left != NONE_U32 && self.nodes[n as usize].right != NONE_U32 {
            let succ = self.leftmost(self.nodes[n as usize].right);
            let succ_edge = self.nodes[succ as usize].edge_idx;
            self.nodes[n as usize].edge_idx = succ_edge;
            self.node_of_edge[succ_edge] = n;
            target = succ;
        }

        let child = if self.nodes[target as usize].left != NONE_U32 {
            self.nodes[target as usize].left
        } else {
            self.nodes[target as usize].right
        };
        let parent = self.nodes[target as usize].parent;
        if child != NONE_U32 {
            self.nodes[child as usize].parent = parent;
        }
        if parent == NONE_U32 {
            self.root = child;
        } else if self.nodes[parent as usize].left == target {
            self.nodes[parent as usize].left = child;
        } else {
            self.nodes[parent as usize].right = child;
        }

        self.node_of_edge[edge_idx] = NONE_U32;
        self.free_list.push(target);
        if parent != NONE_U32 {
            self.rebalance_up(parent);
        }

        (prev, next)
    }

    pub(super) fn lower_bound_y(&self, target_y: f64, sweep_x: f64, segs: &[Seg]) -> Option<usize> {
        let mut cur = self.root;
        let mut candidate = NONE_U32;
        while cur != NONE_U32 {
            let e = self.nodes[cur as usize].edge_idx;
            let y = segs[e].y_at(sweep_x);
            if y >= target_y {
                candidate = cur;
                cur = self.nodes[cur as usize].left;
            } else {
                cur = self.nodes[cur as usize].right;
            }
        }
        if candidate == NONE_U32 {
            None
        } else {
            Some(self.nodes[candidate as usize].edge_idx)
        }
    }
}
