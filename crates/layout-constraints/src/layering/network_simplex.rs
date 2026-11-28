use std::collections::VecDeque;

use egraph_core::{CoreError, CoreResult, VisitStamp};

pub(super) fn network_simplex_on_dag(
    n: usize,
    dag_indptr: &[u32],
    dag_indices: &[u32],
    lp_layers: &[u32],
) -> CoreResult<Vec<u32>> {
    if dag_indices.is_empty() {
        return Ok(vec![0u32; n]);
    }

    // Build reverse CSR of the DAG for weakly connected component discovery.
    let mut rev_counts = vec![0u32; n];
    for &v in dag_indices {
        rev_counts[v as usize] += 1;
    }
    let mut rev_indptr = Vec::with_capacity(n + 1);
    rev_indptr.push(0u32);
    let mut acc = 0u32;
    for &c in &rev_counts {
        acc += c;
        rev_indptr.push(acc);
    }
    let mut rev_cursor = rev_indptr[..n].to_vec();
    let mut rev_indices = vec![0u32; dag_indices.len()];
    for u in 0..n {
        let start = dag_indptr[u] as usize;
        let end = dag_indptr[u + 1] as usize;
        for &v in &dag_indices[start..end] {
            let dst = rev_cursor[v as usize] as usize;
            rev_indices[dst] = u as u32;
            rev_cursor[v as usize] += 1;
        }
    }

    let mut out_layers = vec![0u32; n];
    let mut visited = VisitStamp::new(n);
    let mut global_to_local = vec![u32::MAX; n];
    let mut comp_nodes: Vec<u32> = Vec::new();
    let mut comp_queue: VecDeque<u32> = VecDeque::new();

    for start_node in 0..n {
        if !visited.visit(start_node) {
            continue;
        }
        comp_nodes.clear();
        comp_queue.push_back(start_node as u32);
        while let Some(u) = comp_queue.pop_front() {
            comp_nodes.push(u);
            let us = u as usize;
            for &v in &dag_indices[dag_indptr[us] as usize..dag_indptr[us + 1] as usize] {
                if visited.visit(v as usize) {
                    comp_queue.push_back(v);
                }
            }
            for &v in &rev_indices[rev_indptr[us] as usize..rev_indptr[us + 1] as usize] {
                if visited.visit(v as usize) {
                    comp_queue.push_back(v);
                }
            }
        }

        let k = comp_nodes.len();
        if k <= 1 {
            out_layers[comp_nodes[0] as usize] = 0;
            continue;
        }

        comp_nodes.sort_unstable();
        for (local_idx, &global_u) in comp_nodes.iter().enumerate() {
            global_to_local[global_u as usize] = local_idx as u32;
        }

        let mut comp_edges: Vec<(u32, u32)> = Vec::new();
        let mut rank: Vec<i64> = Vec::with_capacity(k);
        for (local_u, &global_u) in comp_nodes.iter().enumerate() {
            rank.push(lp_layers[global_u as usize] as i64);
            let gs = global_u as usize;
            for &global_v in &dag_indices[dag_indptr[gs] as usize..dag_indptr[gs + 1] as usize] {
                let local_v = global_to_local[global_v as usize];
                comp_edges.push((local_u as u32, local_v));
            }
        }

        solve_component_network_simplex(k, &comp_edges, &mut rank);

        let min_rank = rank.iter().copied().min().unwrap_or(0);
        for (local_u, &global_u) in comp_nodes.iter().enumerate() {
            let delta = rank[local_u] - min_rank;
            out_layers[global_u as usize] = u32::try_from(delta)
                .map_err(|_| CoreError::invalid_input("assigned layer index exceeds u32::MAX"))?;
            global_to_local[global_u as usize] = u32::MAX;
        }
    }

    // Ensure global min layer is 0 (each component already has min_rank == 0).
    Ok(out_layers)
}

fn solve_component_network_simplex(k: usize, edges: &[(u32, u32)], rank: &mut [i64]) {
    let m = edges.len();
    if k <= 1 || m == 0 {
        return;
    }

    // Build undirected incidence CSR of component edges: each edge e=(u,v) appears at u and v.
    let mut deg = vec![0u32; k];
    let mut node_balance = vec![0i64; k];
    for &(u, v) in edges {
        deg[u as usize] += 1;
        deg[v as usize] += 1;
        node_balance[u as usize] -= 1;
        node_balance[v as usize] += 1;
    }
    let mut adj_indptr = Vec::with_capacity(k + 1);
    adj_indptr.push(0u32);
    let mut acc = 0u32;
    for &d in &deg {
        acc += d;
        adj_indptr.push(acc);
    }
    let mut adj_cursor = adj_indptr[..k].to_vec();
    let mut adj_to = vec![0u32; acc as usize];
    let mut adj_eidx = vec![0u32; acc as usize];
    for (e_idx, &(u, v)) in edges.iter().enumerate() {
        let eu = adj_cursor[u as usize] as usize;
        adj_to[eu] = v;
        adj_eidx[eu] = e_idx as u32;
        adj_cursor[u as usize] += 1;

        let ev = adj_cursor[v as usize] as usize;
        adj_to[ev] = u;
        adj_eidx[ev] = e_idx as u32;
        adj_cursor[v as usize] += 1;
    }

    // 1. Feasible tight spanning tree construction.
    let mut in_tree = vec![false; k];
    let mut is_tree_edge = vec![false; m];
    in_tree[0] = true;
    let mut tree_count = 1usize;
    let mut bfs_q = VecDeque::with_capacity(k);
    bfs_q.push_back(0u32);

    while tree_count < k {
        while let Some(u) = bfs_q.pop_front() {
            let start = adj_indptr[u as usize] as usize;
            let end = adj_indptr[u as usize + 1] as usize;
            for pos in start..end {
                let v = adj_to[pos];
                if in_tree[v as usize] {
                    continue;
                }
                let e_idx = adj_eidx[pos] as usize;
                let (ea, eb) = edges[e_idx];
                if rank[eb as usize] - rank[ea as usize] == 1 {
                    in_tree[v as usize] = true;
                    is_tree_edge[e_idx] = true;
                    tree_count += 1;
                    bfs_q.push_back(v);
                }
            }
        }

        if tree_count >= k {
            break;
        }

        // Find minimum-slack crossing edge to bring into the tight tree.
        let mut best_slack = i64::MAX;
        let mut best_delta = 0i64;
        let mut best_seed = 0u32;

        for &(ea, eb) in edges {
            let a_in = in_tree[ea as usize];
            let b_in = in_tree[eb as usize];
            if a_in == b_in {
                continue;
            }
            let slack = rank[eb as usize] - rank[ea as usize] - 1;
            if slack < best_slack {
                best_slack = slack;
                if a_in {
                    best_delta = slack;
                    best_seed = ea;
                } else {
                    best_delta = -slack;
                    best_seed = eb;
                }
            }
        }

        if best_slack == i64::MAX {
            break;
        }
        if best_delta != 0 {
            for (x, in_t) in in_tree.iter().enumerate() {
                if *in_t {
                    rank[x] += best_delta;
                }
            }
        }
        bfs_q.push_back(best_seed);
    }

    // 2. Network Simplex pivoting loop.
    let mut parent = vec![u32::MAX; k];
    let mut parent_eidx = vec![u32::MAX; k];
    let mut parent_dir_down = vec![false; k];
    let mut tin = vec![0u32; k];
    let mut tout = vec![0u32; k];
    let mut dfs_order = Vec::with_capacity(k);
    let mut sub_balance = vec![0i64; k];
    let mut dfs_stack: Vec<(u32, usize)> = Vec::with_capacity(k);

    let max_pivots = (k.saturating_mul(m)).clamp(10_000, 500_000);
    for _pivot in 0..max_pivots {
        // Iterative DFS to compute Euler tour [tin, tout], parent pointers, and pre-order.
        parent.fill(u32::MAX);
        parent_eidx.fill(u32::MAX);
        parent_dir_down.fill(false);
        dfs_order.clear();
        dfs_stack.clear();

        let mut timer = 0u32;
        parent[0] = 0;
        tin[0] = timer;
        timer += 1;
        dfs_order.push(0);
        dfs_stack.push((0, adj_indptr[0] as usize));

        while let Some(&mut (u, ref mut pos)) = dfs_stack.last_mut() {
            let end = adj_indptr[u as usize + 1] as usize;
            if *pos < end {
                let cur_pos = *pos;
                *pos += 1;
                let e_idx = adj_eidx[cur_pos] as usize;
                if !is_tree_edge[e_idx] {
                    continue;
                }
                let v = adj_to[cur_pos];
                if v == parent[u as usize] {
                    continue;
                }
                parent[v as usize] = u;
                parent_eidx[v as usize] = e_idx as u32;
                parent_dir_down[v as usize] = edges[e_idx].0 == u;
                tin[v as usize] = timer;
                timer += 1;
                dfs_order.push(v);
                dfs_stack.push((v, adj_indptr[v as usize] as usize));
            } else {
                tout[u as usize] = timer - 1;
                dfs_stack.pop();
            }
        }

        if dfs_order.len() != k {
            break;
        }

        // Bottom-up subtree net balance = cut values for all tree edges.
        sub_balance.copy_from_slice(&node_balance);
        let mut best_leave: Option<(u32, i64, u32)> = None; // (node, cut_val, e_idx)

        for idx in (1..dfs_order.len()).rev() {
            let x = dfs_order[idx];
            let xs = x as usize;
            let p = parent[xs] as usize;
            let sb = sub_balance[xs];
            sub_balance[p] += sb;

            let cut_val = if parent_dir_down[xs] { sb } else { -sb };
            if cut_val < 0 {
                let e_idx = parent_eidx[xs];
                match best_leave {
                    None => best_leave = Some((x, cut_val, e_idx)),
                    Some((_, best_cut, best_e)) => {
                        if cut_val < best_cut || (cut_val == best_cut && e_idx < best_e) {
                            best_leave = Some((x, cut_val, e_idx));
                        }
                    }
                }
            }
        }

        let Some((leave_node, _, e_leave_u32)) = best_leave else {
            break; // All tree edges have cut value >= 0 => optimal!
        };
        let e_leave = e_leave_u32 as usize;
        let ln_tin = tin[leave_node as usize];
        let ln_tout = tout[leave_node as usize];
        let head_is_subtree = parent_dir_down[leave_node as usize];

        let is_in_head = |y: u32| -> bool {
            let ty = tin[y as usize];
            let in_sub = ty >= ln_tin && ty <= ln_tout;
            if head_is_subtree { in_sub } else { !in_sub }
        };

        // Find minimum-slack entering edge crossing from head to tail.
        let mut best_enter: Option<(usize, i64)> = None;
        for (e_idx, &(ea, eb)) in edges.iter().enumerate() {
            if is_tree_edge[e_idx] {
                continue;
            }
            if is_in_head(ea) && !is_in_head(eb) {
                let slack = rank[eb as usize] - rank[ea as usize] - 1;
                match best_enter {
                    None => best_enter = Some((e_idx, slack)),
                    Some((be, bs)) => {
                        if slack < bs || (slack == bs && e_idx < be) {
                            best_enter = Some((e_idx, slack));
                        }
                    }
                }
            }
        }

        let Some((e_enter, delta)) = best_enter else {
            break;
        };

        if delta != 0 {
            for (y, r) in rank.iter_mut().enumerate() {
                if is_in_head(y as u32) {
                    *r += delta;
                }
            }
        }
        is_tree_edge[e_leave] = false;
        is_tree_edge[e_enter] = true;
    }
}
