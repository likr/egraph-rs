use ordered_float::OrderedFloat;
use petgraph_drawing::{Delta, Drawing, Metric};

struct Constraint {
    left: usize,
    right: usize,
    gap: f32,
}

struct Block {
    variables: Vec<usize>,
    position: f32,
    active: Vec<Constraint>,
}

struct ConstraintGraph {
    blocks: Vec<Block>,
    constraints: Vec<Constraint>,
    neighbors: Vec<Vec<usize>>,
}

fn optimal_position(block: &Block, x: &[f32], offset: &[f32]) -> f32 {
    let mut s = 0.;
    for &j in block.variables.iter() {
        s += x[j] - offset[j]
    }
    s / block.variables.len() as f32
}

fn comp_dfdv(
    v: usize,
    active_constraints: &[Constraint],
    u: Option<usize>,
    lm: &mut [f32],
    x: &[f32],
    blocks: &[Block],
) -> f32 {
    let mut dfdv = blocks[v].position - x[v];
    for (i, c) in active_constraints.iter().enumerate() {
        if v == c.left && u != Some(c.right) {
            lm[i] = comp_dfdv(c.right, active_constraints, Some(v), lm, x, blocks);
            dfdv += lm[i];
        } else if v == c.right && u != Some(c.left) {
            lm[i] = -comp_dfdv(c.left, active_constraints, Some(v), lm, x, blocks);
            dfdv -= lm[i];
        }
    }
    dfdv
}

pub fn split_blocks(x: &[f32], blocks: &mut [Block], offset: &[f32]) -> bool {
    let mut nosplit = true;
    for i in 0..blocks.len() {
        if blocks[i].variables.is_empty() {
            continue;
        }
        blocks[i].position = optimal_position(&blocks[i], x, offset);
        let mut lm = vec![0.; blocks[i].active.len()];
        let v = blocks[i].variables[0];
        comp_dfdv(v, &blocks[i].active, None, &mut lm, x, blocks);
        let sc = (0..blocks[i].active.len())
            .min_by_key(|&i| OrderedFloat(lm[i]))
            .unwrap();
        if lm[sc] >= 0. {
            break;
        }
        nosplit = false;
        let c = constraints[blocks[i].active[sc]];
        blocks[i].active.remove(sc);
        let s = constraints[c].right;
    }
    nosplit
}
