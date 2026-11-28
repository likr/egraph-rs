#[derive(Debug, Clone, Copy)]
pub(super) struct GridEntry {
    pub(super) cy: i64,
    pub(super) cx: i64,
    pub(super) node: usize,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct GridCell {
    pub(super) cy: i64,
    pub(super) cx: i64,
    pub(super) start: usize,
    pub(super) end: usize,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Grid3Entry {
    pub(super) cz: i64,
    pub(super) cy: i64,
    pub(super) cx: i64,
    pub(super) node: usize,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Grid3Cell {
    pub(super) cz: i64,
    pub(super) cy: i64,
    pub(super) cx: i64,
    pub(super) start: usize,
    pub(super) end: usize,
}

#[inline]
pub(super) fn clamp_cell_coord(v: f64) -> i64 {
    if !v.is_finite() {
        return 0;
    }
    let min_c = -1_000_000_000_000.0_f64;
    let max_c = 1_000_000_000_000.0_f64;
    v.clamp(min_c, max_c) as i64
}
