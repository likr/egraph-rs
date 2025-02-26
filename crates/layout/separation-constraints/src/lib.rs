use ordered_float::OrderedFloat;
use petgraph_drawing::{Delta, Drawing, Metric};
use std::collections::{HashMap, HashSet, VecDeque};

struct Variable {
    block: usize,
    offset: f32,
}

struct Constraint {
    left: usize,
    right: usize,
    gap: f32,
}

struct Block {
    variables: HashSet<usize>,
    position: f32,
    active: HashSet<usize>,
}

pub struct ConstraintGraph {
    variables: Vec<Variable>,
    constraints: Vec<Constraint>,
    blocks: Vec<Block>,
    neighbors: Vec<Vec<usize>>,
}

impl ConstraintGraph {
    fn optimal_position(&self, i: usize, x: &[f32]) -> f32 {
        let mut s = 0.;
        for &j in self.blocks[i].variables.iter() {
            s += x[j] - self.variables[j].offset
        }
        s / self.blocks[i].variables.len() as f32
    }

    fn comp_dfdv(
        &self,
        v: usize,
        active_constraints: &HashSet<usize>,
        u: Option<usize>,
        lm: &mut HashMap<usize, f32>,
        x: &[f32],
    ) -> f32 {
        let mut dfdv = self.variable_position(v) - x[v];
        for &i in active_constraints.iter() {
            let c = &self.constraints[i];
            if v == c.left && u != Some(c.right) {
                let value = self.comp_dfdv(c.right, active_constraints, Some(v), lm, x);
                lm.insert(i, value);
                dfdv += value;
            } else if v == c.right && u != Some(c.left) {
                let value = -self.comp_dfdv(c.left, active_constraints, Some(v), lm, x);
                lm.insert(i, value);
                dfdv -= value;
            }
        }
        dfdv
    }

    pub fn split_blocks(&mut self, x: &[f32]) -> bool {
        let mut nosplit = true;
        for i in 0..self.blocks.len() {
            if self.blocks[i].variables.is_empty() {
                continue;
            }
            self.blocks[i].position = self.optimal_position(i, x);
            let mut ac = self.blocks[i].active.clone();
            let mut lm = HashMap::new();
            let v = *self.blocks[i].variables.iter().nth(0).unwrap();
            self.comp_dfdv(v, &self.blocks[i].active, None, &mut lm, x);
            let Some(&sc) = self.blocks[i]
                .active
                .iter()
                .min_by_key(|&&i| OrderedFloat(lm[&i]))
            else {
                break;
            };
            if lm[&sc] >= 0. {
                break;
            }
            nosplit = false;
            ac.remove(&sc);
            let s = self.constraints[sc].right;
            self.blocks[s].variables = self.connected(s, &ac);
            for &v in self.blocks[s].variables.iter() {
                self.variables[v].block = s;
            }
            self.blocks[i].variables = self.blocks[i]
                .variables
                .difference(&self.blocks[s].variables)
                .cloned()
                .collect();
            self.blocks[i].position = self.optimal_position(i, x);
            self.blocks[s].position = self.optimal_position(s, x);
            self.blocks[s].active = ac
                .iter()
                .filter(|&&c| {
                    self.blocks[s].variables.contains(&self.constraints[c].left)
                        && self.blocks[s]
                            .variables
                            .contains(&self.constraints[c].right)
                })
                .cloned()
                .collect();
            for &c in self.blocks[s].active.iter() {
                ac.remove(&c);
            }
            self.blocks[i].active = ac;
        }
        nosplit
    }

    fn connected(&self, s: usize, ac: &HashSet<usize>) -> HashSet<usize> {
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        queue.push_back(s);
        visited.insert(s);
        while let Some(u) = queue.pop_front() {
            for &v in self.neighbors[u].iter() {
                if ac.contains(&v) && !visited.contains(&v) {
                    queue.push_back(v);
                    visited.insert(v);
                }
            }
        }
        visited
    }

    pub fn project(&mut self, x: &mut [f32]) {
        while let Some(c) = (0..self.constraints.len())
            .filter(|&c| self.violation(c) >= 0.)
            .max_by_key(|&c| OrderedFloat(self.violation(c)))
        {
            if self.variables[self.constraints[c].left].block
                != self.variables[self.constraints[c].right].block
            {
                self.merge_block(
                    self.variables[self.constraints[c].left].block,
                    self.variables[self.constraints[c].right].block,
                    c,
                )
            } else {
                self.expand_block(self.variables[self.constraints[c].left].block, c, x);
            }
        }
        for i in 0..self.variables.len() {
            x[i] = self.blocks[self.variables[i].block].position + self.variables[i].offset;
        }
    }

    fn variable_position(&self, i: usize) -> f32 {
        self.blocks[self.variables[i].block].position + self.variables[i].offset
    }

    fn violation(&self, c: usize) -> f32 {
        self.variable_position(self.constraints[c].left) + self.constraints[c].gap
            - self.variable_position(self.constraints[c].right)
    }

    fn merge_block(&mut self, l: usize, r: usize, c: usize) {
        let d = self.variables[self.constraints[c].left].offset + self.constraints[c].gap
            - self.variables[self.constraints[c].right].offset;
        let nvar_l = self.blocks[l].variables.len() as f32;
        let nvar_r = self.blocks[r].variables.len() as f32;
        self.blocks[l].position = (self.blocks[l].position * nvar_l
            + (self.blocks[r].position - d) * nvar_r)
            / (nvar_l + nvar_r);
        self.blocks[l].active = self.blocks[l]
            .active
            .union(&self.blocks[r].active)
            .cloned()
            .collect();
        self.blocks[l].active.insert(c);
        for &i in self.blocks[r].variables.iter() {
            self.variables[i].block = l;
            self.variables[i].offset += d;
        }
        self.blocks[l].variables = self.blocks[l]
            .variables
            .union(&self.blocks[r].variables)
            .cloned()
            .collect();
        self.blocks[r].variables.clear();
    }

    fn expand_block(&mut self, b: usize, c: usize, x: &[f32]) {
        let mut lm = HashMap::new();
        let mut ac = self.blocks[b].active.clone();
        self.comp_dfdv(c, &ac, None, &mut lm, x);
        let vs = self.comp_path(self.constraints[c].left, self.constraints[c].right, &ac);
        let vs_adj = (1..vs.len())
            .map(|i| (vs[i - 1], vs[i]))
            .collect::<HashSet<_>>();
        let ps = ac
            .iter()
            .filter(|&&c| vs_adj.contains(&(self.constraints[c].left, self.constraints[c].right)))
            .cloned()
            .collect::<Vec<_>>();
        if let Some(&sc) = ps.iter().min_by_key(|&c| OrderedFloat(lm[&c])) {
            ac.remove(&sc);
            for &v in self.connected(self.constraints[c].right, &ac).iter() {
                self.variables[v].offset += self.violation(c);
            }
            ac.insert(c);
            self.blocks[b].active = ac;
            self.blocks[b].position = self.optimal_position(b, x);
        }
    }

    fn comp_path(&self, s: usize, t: usize, ac: &HashSet<usize>) -> Vec<usize> {
        let mut queue = VecDeque::new();
        let mut parent = HashMap::new();
        queue.push_back(s);
        parent.insert(s, None);
        while let Some(u) = queue.pop_front() {
            for &v in self.neighbors[u].iter() {
                if ac.contains(&v) && !parent.contains_key(&v) {
                    queue.push_back(v);
                    parent.insert(v, Some(u));
                }
            }
        }
        let mut path = vec![t];
        let mut u = t;
        while let Some(v) = parent[&u] {
            path.push(v);
            u = v;
        }
        path.reverse();
        path
    }
}
