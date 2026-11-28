use std::collections::VecDeque;

use super::coding_tree::{CodingTreeNode, LeafUpMapping};
use super::entropy::tree_structural_entropy;
use super::k_tree::{build_k_tree, check_balance};

#[allow(clippy::type_complexity)]
pub(super) fn compute_leaf_up(
    nodes: &[Option<CodingTreeNode>],
    n: usize,
    total_vol: f64,
    vert_edge_indptr: &[usize],
    vert_edge_nbr: &[u32],
    vert_edge_w: &[f64],
    leaf_to_tree: &mut [usize],
) -> (f64, Vec<LeafUpMapping>) {
    let mut h1_ids: Vec<usize> = Vec::new();
    for node_opt in nodes.iter().take(n) {
        if let Some(p_id) = node_opt.as_ref().and_then(|node| node.parent) {
            h1_ids.push(p_id);
        }
    }
    h1_ids.sort_unstable();
    h1_ids.dedup();

    let mut delta = 0.0_f64;
    let mut mappings = Vec::with_capacity(h1_ids.len());

    for &h1_id in &h1_ids {
        let Some(cand) = nodes[h1_id].as_ref() else {
            continue;
        };
        if cand.partition.len() < 3 {
            mappings.push((h1_id, None));
            continue;
        }

        let sub_g_vol = cand.vol - cand.g;
        if sub_g_vol <= 0.0 {
            mappings.push((h1_id, None));
            continue;
        }

        let m_sub = cand.partition.len();
        for (local_i, &v_u32) in cand.partition.iter().enumerate() {
            leaf_to_tree[v_u32 as usize] = local_i;
        }

        let mut ori_ent = 0.0_f64;
        let mut sub_nodes: Vec<Option<CodingTreeNode>> = Vec::with_capacity(m_sub * 3);
        let mut sub_adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m_sub];

        for (local_u, &u_u32) in cand.partition.iter().enumerate() {
            let u = u_u32 as usize;
            if let Some(orig_leaf) = nodes[u].as_ref() {
                ori_ent +=
                    -(orig_leaf.g / total_vol) * ((orig_leaf.vol + 1.0) / (cand.vol + 1.0)).log2();
            }
            let mut sub_vol = 0.0_f64;
            for pos in vert_edge_indptr[u]..vert_edge_indptr[u + 1] {
                let v = vert_edge_nbr[pos] as usize;
                let w = vert_edge_w[pos];
                if w > 0.0 && cand.partition.binary_search(&(v as u32)).is_ok() {
                    let local_v = leaf_to_tree[v];
                    sub_vol += w;
                    sub_adj[local_u].push((local_v, w));
                }
            }
            sub_adj[local_u].sort_unstable_by_key(|x| x.0);
            sub_nodes.push(Some(CodingTreeNode {
                parent: None,
                children: Vec::new(),
                partition: vec![u_u32],
                vol: sub_vol,
                g: sub_vol,
                merged: false,
                child_h: 0,
                child_cut: 0.0,
                depth: 0,
                e_increase: 0.0,
                a_increase: 0.0,
            }));
        }

        let active: Vec<usize> = (0..m_sub).collect();
        let sub_root = build_k_tree(&mut sub_nodes, &active, &mut sub_adj, sub_g_vol, Some(2));
        check_balance(&mut sub_nodes, sub_root);

        let new_ent = leaf_up_entropy(nodes, &mut sub_nodes, sub_root, cand.vol, cand.g, total_vol);
        delta += ori_ent - new_ent;
        mappings.push((h1_id, Some((sub_root, sub_nodes))));
    }

    delta /= n.max(1) as f64;
    (delta, mappings)
}

fn leaf_up_entropy(
    orig_nodes: &[Option<CodingTreeNode>],
    sub_nodes: &mut [Option<CodingTreeNode>],
    sub_root: usize,
    cand_vol: f64,
    cand_g: f64,
    total_vol: f64,
) -> f64 {
    let mut ent = 0.0_f64;
    let mut queue = VecDeque::new();
    queue.push_back(sub_root);
    while let Some(sid) = queue.pop_front() {
        let (children, ch, part) = match sub_nodes[sid].as_ref() {
            Some(s) => (s.children.clone(), s.child_h, s.partition.clone()),
            None => continue,
        };
        for &c in &children {
            queue.push_back(c);
        }
        if sid == sub_root {
            if let Some(s) = sub_nodes[sid].as_mut() {
                s.vol = cand_vol;
                s.g = cand_g;
            }
            continue;
        }
        if ch == 1 {
            let (inner_vol, orig_vol) = {
                let s = sub_nodes[sid].as_ref();
                let inner = s.map_or(0.0, |x| x.vol - x.g);
                let ov: f64 = part
                    .iter()
                    .filter_map(|&u| orig_nodes[u as usize].as_ref().map(|n| n.vol))
                    .sum();
                (inner, ov)
            };
            let orig_g = (orig_vol - inner_vol).max(0.0);
            if let Some(s) = sub_nodes[sid].as_mut() {
                s.vol = orig_vol;
                s.g = orig_g;
            }
        } else if ch == 0 && part.len() == 1 {
            let u = part[0] as usize;
            if let (Some(orig), Some(s)) = (orig_nodes[u].as_ref(), sub_nodes[sid].as_mut()) {
                s.vol = orig.vol;
                s.g = orig.g;
            }
        }

        if let Some(s) = sub_nodes[sid].as_ref()
            && let Some(pid) = s.parent
            && let Some(p) = sub_nodes[pid].as_ref()
        {
            ent += -(s.g / total_vol) * ((s.vol + 1.0) / (p.vol + 1.0)).log2();
        }
    }
    ent
}

pub(super) fn apply_leaf_up(
    nodes: &mut Vec<Option<CodingTreeNode>>,
    root_id: usize,
    mappings: Vec<LeafUpMapping>,
) {
    for (h1_id, sub_opt) in mappings {
        match sub_opt {
            None => {
                let children = nodes[h1_id]
                    .as_ref()
                    .map(|n| n.children.clone())
                    .unwrap_or_default();
                for c in children {
                    super::k_tree::single_up(nodes, c);
                }
            }
            Some((sub_root, mut sub_nodes)) => {
                // Re-index non-leaf subtree nodes above `nodes.len()`, and map leaf nodes back to original `u32` IDs.
                let mut id_remap = vec![usize::MAX; sub_nodes.len()];
                let mut next_global = nodes.len();
                for (sid, s_opt) in sub_nodes.iter().enumerate() {
                    if sid == sub_root {
                        id_remap[sid] = h1_id;
                        continue;
                    }
                    let Some(s) = s_opt.as_ref() else {
                        continue;
                    };
                    if s.child_h == 0 && s.partition.len() == 1 {
                        id_remap[sid] = s.partition[0] as usize;
                    } else {
                        id_remap[sid] = next_global;
                        next_global += 1;
                    }
                }
                nodes.resize(next_global, None);

                let sub_root_children = sub_nodes[sub_root]
                    .as_ref()
                    .map(|r| r.children.clone())
                    .unwrap_or_default();

                for (sid, s_opt) in sub_nodes.iter_mut().enumerate() {
                    if sid == sub_root {
                        continue;
                    }
                    let Some(mut s) = s_opt.take() else {
                        continue;
                    };
                    let gid = id_remap[sid];
                    if s.child_h == 0 && s.partition.len() == 1 {
                        if let Some(orig) = nodes[gid].as_mut() {
                            orig.parent = s.parent.map(|p| id_remap[p]);
                        }
                    } else {
                        s.parent = s.parent.map(|p| id_remap[p]);
                        for c in &mut s.children {
                            *c = id_remap[*c];
                        }
                        s.children.sort_unstable();
                        nodes[gid] = Some(s);
                    }
                }

                if let Some(h1_node) = nodes[h1_id].as_mut() {
                    h1_node.children = sub_root_children.into_iter().map(|c| id_remap[c]).collect();
                    h1_node.children.sort_unstable();
                }
            }
        }
    }
    if let Some(root) = nodes[root_id].as_mut() {
        root.child_h += 1;
    }
}

#[allow(clippy::type_complexity)]
pub(super) fn compute_root_down(
    nodes: &[Option<CodingTreeNode>],
    root_id: usize,
    n: usize,
    vert_edge_indptr: &[usize],
    vert_edge_nbr: &[u32],
    vert_edge_w: &[f64],
    leaf_to_tree: &mut [usize],
) -> Option<(f64, Option<(usize, Vec<Option<CodingTreeNode>>)>)> {
    let root = nodes[root_id].as_ref()?;
    if root.children.len() < 3 {
        return Some((0.0, None));
    }

    let k_c = root.children.len();
    let g_vol = root.vol;
    if g_vol <= 0.0 {
        return Some((0.0, None));
    }

    for (local_c, &cid) in root.children.iter().enumerate() {
        if let Some(c) = nodes[cid].as_ref() {
            for &u in &c.partition {
                leaf_to_tree[u as usize] = local_c;
            }
        }
    }

    let mut ori_ent = 0.0_f64;
    let mut sub_nodes: Vec<Option<CodingTreeNode>> = Vec::with_capacity(k_c * 3);
    let mut sub_adj_pairs: Vec<Vec<(usize, f64)>> = vec![Vec::new(); k_c];

    for (local_c, &cid) in root.children.iter().enumerate() {
        let Some(c) = nodes[cid].as_ref() else {
            return Some((0.0, None));
        };
        ori_ent += -(c.g / g_vol) * ((c.vol + 1.0) / g_vol).log2();
        for &u_u32 in &c.partition {
            let u = u_u32 as usize;
            for pos in vert_edge_indptr[u]..vert_edge_indptr[u + 1] {
                let v = vert_edge_nbr[pos] as usize;
                let w = vert_edge_w[pos];
                let local_v = leaf_to_tree[v];
                if w > 0.0 && local_v != local_c {
                    sub_adj_pairs[local_c].push((local_v, w));
                }
            }
        }
        sub_adj_pairs[local_c].sort_unstable_by_key(|x| x.0);
        // Aggregate parallel edges between distinct root children.
        let mut merged_row: Vec<(usize, f64)> = Vec::new();
        for &(nbr_c, w) in &sub_adj_pairs[local_c] {
            if let Some(last) = merged_row.last_mut()
                && last.0 == nbr_c
            {
                last.1 += w;
            } else {
                merged_row.push((nbr_c, w));
            }
        }
        sub_adj_pairs[local_c] = merged_row;

        sub_nodes.push(Some(CodingTreeNode {
            parent: None,
            children: vec![cid],
            partition: c.partition.clone(),
            vol: c.vol,
            g: c.g,
            merged: false,
            child_h: 0,
            child_cut: 0.0,
            depth: 0,
            e_increase: 0.0,
            a_increase: 0.0,
        }));
    }

    let _ = n;
    let active: Vec<usize> = (0..k_c).collect();
    let sub_root = build_k_tree(&mut sub_nodes, &active, &mut sub_adj_pairs, g_vol, Some(2));
    check_balance(&mut sub_nodes, sub_root);

    let new_ent = tree_structural_entropy(&sub_nodes, g_vol);
    let delta = (ori_ent - new_ent) / (k_c as f64);
    Some((delta, Some((sub_root, sub_nodes))))
}

pub(super) fn apply_root_down(
    nodes: &mut Vec<Option<CodingTreeNode>>,
    root_id: usize,
    sub_root: usize,
    mut sub_nodes: Vec<Option<CodingTreeNode>>,
) {
    let orig_children = nodes[root_id]
        .as_ref()
        .map(|r| r.children.clone())
        .unwrap_or_default();
    let k_c = orig_children.len();

    let mut id_remap = vec![usize::MAX; sub_nodes.len()];
    let mut next_global = nodes.len();
    for sid in 0..sub_nodes.len() {
        if sid == sub_root {
            id_remap[sid] = root_id;
        } else if sid < k_c {
            id_remap[sid] = orig_children[sid];
        } else if sub_nodes[sid].is_some() {
            id_remap[sid] = next_global;
            next_global += 1;
        }
    }
    nodes.resize(next_global, None);

    let sub_root_children = sub_nodes[sub_root]
        .as_ref()
        .map(|r| r.children.clone())
        .unwrap_or_default();

    for (sid, s_opt) in sub_nodes.iter_mut().enumerate() {
        if sid == sub_root {
            continue;
        }
        let Some(mut s) = s_opt.take() else {
            continue;
        };
        let gid = id_remap[sid];
        if sid < k_c {
            if let Some(orig_c) = nodes[gid].as_mut() {
                orig_c.parent = s.parent.map(|p| id_remap[p]);
            }
        } else {
            s.parent = s.parent.map(|p| id_remap[p]);
            for c in &mut s.children {
                *c = id_remap[*c];
            }
            s.children.sort_unstable();
            nodes[gid] = Some(s);
        }
    }

    if let Some(root) = nodes[root_id].as_mut() {
        root.children = sub_root_children.into_iter().map(|c| id_remap[c]).collect();
        root.children.sort_unstable();
        root.child_h += 1;
    }
}
