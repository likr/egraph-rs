use egraph_core::Scalar;

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct SapRect {
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    idx: u32,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn sap_find_overlaps<T: Scalar>(
    xs: &[T],
    ys: &[T],
    hw: &[T],
    hh: &[T],
    sap_bounds: &mut Vec<SapRect>,
    sap_order: &mut Vec<u32>,
    sap_sorted: &mut Vec<SapRect>,
    out: &mut Vec<(u32, u32)>,
) {
    out.clear();
    let n = xs.len();
    if n < 2 {
        return;
    }
    sap_bounds.resize(n, SapRect::default());
    for i in 0..n {
        let xi = xs[i];
        let yi = ys[i];
        let hwi = hw[i];
        let hhi = hh[i];
        sap_bounds[i] = SapRect {
            x_min: (xi - hwi).to_f64(),
            x_max: (xi + hwi).to_f64(),
            y_min: (yi - hhi).to_f64(),
            y_max: (yi + hhi).to_f64(),
            idx: i as u32,
        };
    }

    if sap_order.len() != n {
        sap_order.clear();
        sap_order.extend(0..n as u32);
    }
    sap_order.sort_unstable_by(|&a, &b| {
        sap_bounds[a as usize]
            .x_min
            .total_cmp(&sap_bounds[b as usize].x_min)
            .then_with(|| a.cmp(&b))
    });

    sap_sorted.clear();
    for &idx in sap_order.iter() {
        sap_sorted.push(sap_bounds[idx as usize]);
    }

    for p in 0..n {
        let a = sap_sorted[p];
        for b in sap_sorted.iter().skip(p + 1) {
            if b.x_min >= a.x_max {
                break;
            }
            if a.x_min < b.x_max && a.y_min < b.y_max && b.y_min < a.y_max {
                if a.idx < b.idx {
                    out.push((a.idx, b.idx));
                } else {
                    out.push((b.idx, a.idx));
                }
            }
        }
    }
}
