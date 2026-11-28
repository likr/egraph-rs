use egraph_core::{CoreResult, CsrGraph, EuclideanLayout, EuclideanStorageRef, Scalar};

const COLLAPSED_SPAN_EPS: f64 = 1e-12;

#[derive(Debug, Clone, Copy)]
pub(super) struct MaxentEdge {
    pub(super) u: u32,
    pub(super) v: u32,
    pub(super) d: f64,
    pub(super) w: f64,
}

#[inline]
pub(super) fn norm2_slice(xs: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    for &x in xs {
        sum += x * x;
    }
    sum.sqrt()
}

#[inline]
fn span_collapsed<S: Scalar>(slice: &[S]) -> bool {
    let mut min_v = slice[0].to_f64();
    let mut max_v = min_v;
    for &v in &slice[1..] {
        let f = v.to_f64();
        min_v = min_v.min(f);
        max_v = max_v.max(f);
    }
    (max_v - min_v) < COLLAPSED_SPAN_EPS
}

pub(super) fn is_layout_collapsed_euclidean<L: EuclideanLayout>(layout: &L) -> bool {
    let n = layout.len();
    if n < 2 {
        return false;
    }
    match layout.as_euclidean_storage() {
        EuclideanStorageRef::Soa2d { x, y } => span_collapsed(x) && span_collapsed(y),
        EuclideanStorageRef::Soa3d { x, y, z } => {
            span_collapsed(x) && span_collapsed(y) && span_collapsed(z)
        }
        EuclideanStorageRef::FlatNd {
            coords,
            num_nodes,
            dims,
        } => {
            for a in 0..dims {
                let mut min_v = coords[a].to_f64();
                let mut max_v = min_v;
                for i in 1..num_nodes {
                    let v = coords[i * dims + a].to_f64();
                    min_v = min_v.min(v);
                    max_v = max_v.max(v);
                }
                if (max_v - min_v) >= COLLAPSED_SPAN_EPS {
                    return false;
                }
            }
            true
        }
    }
}

pub(super) fn extract_canonical_positive_edges<W: Scalar>(
    graph: &CsrGraph<W>,
) -> CoreResult<Vec<(u32, u32, f64)>> {
    let n = graph.num_nodes();
    if n <= 1 || graph.nnz() == 0 {
        return Ok(Vec::new());
    }
    let weights_opt = graph.weights();
    if !graph.is_directed() {
        let mut edges = Vec::with_capacity(graph.num_edges());
        let mut is_sorted = true;
        for u in 0..n {
            let u32_u = u as u32;
            let start = graph.indptr()[u] as usize;
            let end = graph.indptr()[u + 1] as usize;
            let mut prev_v: Option<u32> = None;
            for pos in start..end {
                let v = graph.indices()[pos];
                if let Some(pv) = prev_v
                    && v <= pv
                {
                    is_sorted = false;
                }
                prev_v = Some(v);
                if u32_u < v {
                    let w = weights_opt.map_or(1.0, |ws| ws[pos].to_f64());
                    if w > 0.0 {
                        edges.push((u32_u, v, w));
                    }
                }
            }
        }
        if !is_sorted {
            edges.sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
        }
        return Ok(edges);
    }

    let mut raw = Vec::with_capacity(graph.nnz());
    for u in 0..n {
        let u32_u = u as u32;
        let start = graph.indptr()[u] as usize;
        let end = graph.indptr()[u + 1] as usize;
        for pos in start..end {
            let v = graph.indices()[pos];
            if u32_u == v {
                continue;
            }
            let w = weights_opt.map_or(1.0, |ws| ws[pos].to_f64());
            if w <= 0.0 {
                continue;
            }
            let (a, b) = if u32_u < v { (u32_u, v) } else { (v, u32_u) };
            raw.push((a, b, w));
        }
    }
    raw.sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
    let mut out: Vec<(u32, u32, f64)> = Vec::with_capacity(raw.len());
    for (a, b, w) in raw {
        if let Some(last) = out.last_mut()
            && last.0 == a
            && last.1 == b
        {
            last.2 = last.2.min(w);
        } else {
            out.push((a, b, w));
        }
    }
    Ok(out)
}

pub(super) fn extract_canonical_unweighted_edges<W>(
    graph: &CsrGraph<W>,
) -> CoreResult<Vec<(u32, u32, f64)>> {
    let n = graph.num_nodes();
    if n <= 1 || graph.nnz() == 0 {
        return Ok(Vec::new());
    }
    if !graph.is_directed() {
        let mut edges = Vec::with_capacity(graph.num_edges());
        let mut is_sorted = true;
        for u in 0..n {
            let u32_u = u as u32;
            let start = graph.indptr()[u] as usize;
            let end = graph.indptr()[u + 1] as usize;
            let mut prev_v: Option<u32> = None;
            for pos in start..end {
                let v = graph.indices()[pos];
                if let Some(pv) = prev_v
                    && v <= pv
                {
                    is_sorted = false;
                }
                prev_v = Some(v);
                if u32_u < v {
                    edges.push((u32_u, v, 1.0));
                }
            }
        }
        if !is_sorted {
            edges.sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
        }
        return Ok(edges);
    }

    let mut raw = Vec::with_capacity(graph.nnz());
    for u in 0..n {
        let u32_u = u as u32;
        let start = graph.indptr()[u] as usize;
        let end = graph.indptr()[u + 1] as usize;
        for pos in start..end {
            let v = graph.indices()[pos];
            if u32_u == v {
                continue;
            }
            let (a, b) = if u32_u < v { (u32_u, v) } else { (v, u32_u) };
            raw.push((a, b, 1.0));
        }
    }
    raw.sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
    let mut out: Vec<(u32, u32, f64)> = Vec::with_capacity(raw.len());
    for (a, b, w) in raw {
        if let Some(last) = out.last_mut()
            && last.0 == a
            && last.1 == b
        {
            continue;
        }
        out.push((a, b, w));
    }
    Ok(out)
}
