use crate::layout::force_directed::force::Point;
use crate::Graph;
use std::f32;

pub struct LineSegment {
    source: usize,
    target: usize,
    point_indices: Vec<usize>,
}

pub struct Line {
    pub source: usize,
    pub target: usize,
    pub points: Vec<Point>,
}

impl LineSegment {
    fn new(source: usize, target: usize) -> LineSegment {
        LineSegment {
            source: source,
            target: target,
            point_indices: Vec::new(),
        }
    }
}

struct EdgePair {
    p: usize,
    q: usize,
    compatibility: f32,
    theta: f32,
}

impl EdgePair {
    fn new(p: usize, q: usize, compatibility: f32, theta: f32) -> EdgePair {
        EdgePair {
            p,
            q,
            compatibility,
            theta,
        }
    }
}

fn distance(p1x: f32, p1y: f32, p2x: f32, p2y: f32) -> f32 {
    let dx = p2x - p1x;
    let dy = p2y - p1y;
    (dx * dx + dy * dy).sqrt().max(1e-6)
}

fn angle(p1: Point, p2: Point, q1: Point, q2: Point) -> f32 {
    let p_norm = distance(p1.x, p1.y, p2.x, p2.y);
    let q_norm = distance(q1.x, q1.y, q2.x, q2.y);
    let pq = (p2.x - p1.x) * (q2.x - q1.x) + (p2.y - p1.y) * (q2.y - q1.y);
    (pq / p_norm / q_norm).acos()
}

fn compatibility(p1: Point, p2: Point, q1: Point, q2: Point) -> f32 {
    let p_norm = distance(p1.x, p1.y, p2.x, p2.y);
    let q_norm = distance(q1.x, q1.y, q2.x, q2.y);
    let l_avg = (p_norm + q_norm) / 2.;
    let pmx = (p1.x + p2.x) / 2.;
    let pmy = (p1.y + p2.y) / 2.;
    let qmx = (q1.x + q2.x) / 2.;
    let qmy = (q1.y + q2.y) / 2.;
    let c_a = {
        let pq = (p2.x - p1.x) * (q2.x - q1.x) + (p2.y - p1.y) * (q2.y - q1.y);
        (pq / p_norm / q_norm).abs()
    };
    let c_s = 2. / (l_avg / p_norm.min(q_norm) + p_norm.max(q_norm) / l_avg);
    let c_p = {
        let mpq2 = distance(pmx, pmy, qmx, qmy);
        l_avg / (l_avg + mpq2)
    };
    let c_v = {
        let vp = {
            let i0r =
                ((p1.y - q1.y) * (p1.y - p2.y) - (p1.x - q1.x) * (p2.x - p1.x)) / p_norm / p_norm;
            let i0x = p1.x + i0r * (p2.x - p1.x);
            let i0y = p1.y + i0r * (p2.y - p1.y);
            let i1r =
                ((p1.y - q2.y) * (p1.y - p2.y) - (p1.x - q2.x) * (p2.x - p1.x)) / p_norm / p_norm;
            let i1x = p1.x + i1r * (p2.x - p1.x);
            let i1y = p1.y + i1r * (p2.y - p1.y);
            let imx = (i0x + i1x) / 2.;
            let imy = (i0y + i1y) / 2.;
            (1. - 2. * distance(pmx, pmy, imx, imy) / distance(i0x, i0y, i1x, i1y)).max(0.0)
        };
        let vq = {
            let i0r =
                ((q1.y - p1.y) * (q1.y - q2.y) - (q1.x - p1.x) * (q2.x - q1.x)) / q_norm / q_norm;
            let i0x = q1.x + i0r * (q2.x - q1.x);
            let i0y = q1.y + i0r * (q2.y - q1.y);
            let i1r =
                ((q1.y - p2.y) * (q1.y - q2.y) - (q1.x - p2.x) * (q2.x - q1.x)) / q_norm / q_norm;
            let i1x = q1.x + i1r * (q2.x - q1.x);
            let i1y = q1.y + i1r * (q2.y - q1.y);
            let imx = (i0x + i1x) / 2.;
            let imy = (i0y + i1y) / 2.;
            (1. - 2. * distance(qmx, qmy, imx, imy) / distance(i0x, i0y, i1x, i1y)).max(0.0)
        };
        vp.min(vq)
    };
    c_a * c_s * c_p * c_v
}

fn apply_spring_force(
    mid_points: &mut Vec<Point>,
    segments: &Vec<LineSegment>,
    points: &Vec<Point>,
    num_p: usize,
    k: f32,
) {
    for segment in segments.iter() {
        let d = distance(
            points[segment.source].x,
            points[segment.source].y,
            points[segment.target].x,
            points[segment.target].y,
        );
        let kp = k / (num_p as usize as f32) / d;
        let n = segment.point_indices.len();
        for i in 0..n {
            let p0 = if i == 0 {
                points[segment.source]
            } else {
                mid_points[segment.point_indices[i - 1]]
            };
            let p2 = if i == n - 1 {
                points[segment.target]
            } else {
                mid_points[segment.point_indices[i + 1]]
            };
            let ref mut p1 = mid_points[segment.point_indices[i]];
            p1.vx += kp * (p0.x - p1.x + p2.x - p1.x);
            p1.vy += kp * (p0.y - p1.y + p2.y - p1.y);
        }
    }
}

fn apply_electrostatic_force(
    mid_points: &mut Vec<Point>,
    segments: &Vec<LineSegment>,
    edge_pairs: &Vec<EdgePair>,
    num_p: usize,
) {
    for pair in edge_pairs {
        let EdgePair {
            p,
            q,
            theta,
            compatibility: c_e,
        } = pair;
        let segment_p = &segments[*p];
        let segment_q = &segments[*q];
        for i in 0..num_p {
            let j = if *theta < f32::consts::PI / 2.0 {
                i
            } else {
                num_p - i - 1
            };
            let pi = mid_points[segment_p.point_indices[i as usize]];
            let qi = mid_points[segment_q.point_indices[j as usize]];
            let dx = qi.x - pi.x;
            let dy = qi.y - pi.y;
            if dx.abs() > 1e-6 || dy.abs() > 1e-6 {
                let w = c_e / (dx * dx + dy * dy).sqrt();
                {
                    let ref mut qi = mid_points[segment_q.point_indices[j as usize]];
                    qi.vx -= dx * w;
                    qi.vy -= dy * w;
                }
                {
                    let ref mut pi = mid_points[segment_p.point_indices[i as usize]];
                    pi.vx += dx * w;
                    pi.vy += dy * w;
                }
            }
        }
    }
}

pub struct EdgeBundlingOptions {
    cycles: usize,
    s0: f32,
    i0: usize,
    s_step: f32,
    i_step: f32,
    minimum_edge_compatibility: f32,
}

pub fn edge_bundling<D, G: Graph<D>>(
    graph: &G,
    points: &Vec<Point>,
    options: &EdgeBundlingOptions,
) -> Vec<Line> {
    let EdgeBundlingOptions {
        cycles,
        s0,
        i0,
        s_step,
        i_step,
        minimum_edge_compatibility,
    } = options;
    let mut mid_points = Vec::new();
    let mut segments: Vec<LineSegment> =
        graph.edges().map(|(u, v)| LineSegment::new(u, v)).collect();

    let mut num_iter = *i0;
    let mut alpha = *s0;

    let edge_pairs = {
        let mut edge_pairs = Vec::new();
        let m = segments.len();
        for p in 0..m {
            let segment_p = &segments[p];
            for q in (p + 1)..m {
                let segment_q = &segments[q];
                let c_e = compatibility(
                    points[segment_p.source],
                    points[segment_p.target],
                    points[segment_q.source],
                    points[segment_q.target],
                );
                if c_e >= *minimum_edge_compatibility {
                    let theta = angle(
                        points[segment_p.source],
                        points[segment_p.target],
                        points[segment_q.source],
                        points[segment_q.target],
                    );
                    edge_pairs.push(EdgePair::new(p, q, c_e, theta));
                }
            }
        }
        edge_pairs
    };

    for cycle in 0..*cycles {
        let dp = (2 as i32).pow(cycle as u32);
        for segment in segments.iter_mut() {
            for j in 0..dp {
                let p0 = if j == 0 {
                    points[segment.source]
                } else {
                    mid_points[segment.point_indices[(j * 2 - 1) as usize]]
                };
                let p1 = if j == dp - 1 {
                    points[segment.target]
                } else {
                    mid_points[segment.point_indices[(j * 2) as usize]]
                };
                mid_points.push(Point::new((p0.x + p1.x) / 2., (p0.y + p1.y) / 2.));
                segment
                    .point_indices
                    .insert((j * 2) as usize, mid_points.len() - 1);
            }
        }

        let num_p = (dp * 2 - 1) as usize;
        for _ in 0..num_iter {
            for point in mid_points.iter_mut() {
                point.vx = 0.;
                point.vy = 0.;
            }

            apply_spring_force(&mut mid_points, &segments, &points, num_p, 0.1);
            apply_electrostatic_force(&mut mid_points, &segments, &edge_pairs, num_p);

            for point in mid_points.iter_mut() {
                point.x += alpha * point.vx;
                point.y += alpha * point.vy;
            }
        }

        alpha *= s_step;
        num_iter = (num_iter as f32 * i_step) as usize;
    }

    segments
        .iter()
        .map(|segment| {
            let mut ps: Vec<Point> = segment
                .point_indices
                .iter()
                .map(|i| mid_points[*i])
                .collect();
            ps.insert(0, points[segment.source]);
            ps.push(points[segment.target]);
            Line {
                source: segment.source,
                target: segment.target,
                points: ps,
            }
        })
        .collect()
}

pub struct ForceDirectedEdgeBundling {
    pub cycles: usize,
    pub s0: f32,
    pub i0: usize,
    pub s_step: f32,
    pub i_step: f32,
    pub minimum_edge_compatibility: f32,
}

impl ForceDirectedEdgeBundling {
    pub fn new() -> ForceDirectedEdgeBundling {
        ForceDirectedEdgeBundling {
            cycles: 6,
            s0: 0.1,
            i0: 90,
            s_step: 0.5,
            i_step: 2. / 3.,
            minimum_edge_compatibility: 0.6,
        }
    }

    pub fn call<D, G: Graph<D>>(&self, graph: &G, points: &Vec<Point>) -> Vec<Line> {
        let options = self.options();
        edge_bundling(graph, &points, &options)
    }

    fn options(&self) -> EdgeBundlingOptions {
        EdgeBundlingOptions {
            cycles: self.cycles,
            s0: self.s0,
            i0: self.i0,
            s_step: self.s_step,
            i_step: self.i_step,
            minimum_edge_compatibility: self.minimum_edge_compatibility,
        }
    }
}
