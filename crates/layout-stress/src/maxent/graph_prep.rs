use egraph_core::{CoreResult, CsrGraph, Layout2d, LayoutNd, Scalar};

const COLLAPSED_SPAN_EPS: f64 = 1e-12;

#[derive(Debug, Clone, Copy)]
pub(super) struct MaxentEdge {
    pub(super) u: u32,
    pub(super) v: u32,
    pub(super) d: f64,
    pub(super) w: f64,
}

#[inline]
pub(super) fn norm2_pair(xs: &[f64], ys: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        sum += x * x + y * y;
    }
    sum.sqrt()
}

#[inline]
pub(super) fn norm2_slice(xs: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    for &x in xs {
        sum += x * x;
    }
    sum.sqrt()
}

pub(super) fn is_layout_collapsed<S: Scalar>(layout: &Layout2d<S>) -> bool {
    let n = layout.len();
    if n < 2 {
        return false;
    }
    let xs = layout.x();
    let ys = layout.y();
    let mut min_x = xs[0].to_f64();
    let mut max_x = min_x;
    let mut min_y = ys[0].to_f64();
    let mut max_y = min_y;
    for i in 1..n {
        let x = xs[i].to_f64();
        let y = ys[i].to_f64();
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    (max_x - min_x) < COLLAPSED_SPAN_EPS && (max_y - min_y) < COLLAPSED_SPAN_EPS
}

pub(super) fn is_layout_collapsed_nd<S: Scalar>(layout: &LayoutNd<S>) -> bool {
    let n = layout.num_nodes();
    if n < 2 {
        return false;
    }
    let d = layout.dims();
    let coords = layout.coords();
    for a in 0..d {
        let mut min_v = coords[a].to_f64();
        let mut max_v = min_v;
        for i in 1..n {
            let v = coords[i * d + a].to_f64();
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        if (max_v - min_v) >= COLLAPSED_SPAN_EPS {
            return false;
        }
    }
    true
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
