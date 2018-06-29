use force::{Force, Link, Point};
use std::collections::HashMap;

pub struct LinkForce {
    links: Vec<Link>,
}

impl LinkForce {
    pub fn new(links: &Vec<Link>) -> LinkForce {
        LinkForce {
            links: links.clone().to_vec(),
        }
    }
}

impl Force for LinkForce {
    fn apply(&self, points: &mut Vec<Point>, alpha: f32) {
        let links = &self.links;
        let mut count: HashMap<usize, usize> = HashMap::new();
        for link in links.iter() {
            if !count.contains_key(&link.source) {
                count.insert(link.source, 0);
            }
            if !count.contains_key(&link.target) {
                count.insert(link.target, 0);
            }
            {
                let v = count.get_mut(&link.source).unwrap();
                *v += 1;
            }
            {
                let v = count.get_mut(&link.target).unwrap();
                *v += 1;
            }
        }
        let bias = links
            .iter()
            .map(|link| {
                let source_count = *count.get(&link.source).unwrap();
                let target_count = *count.get(&link.target).unwrap();
                source_count as f32 / (source_count + target_count) as f32
            })
            .collect::<Vec<f32>>();
        for (link, b) in links.iter().zip(bias.iter()) {
            let source = points[link.source];
            let target = points[link.target];
            let source_count = count.get(&link.source).unwrap();
            let target_count = count.get(&link.target).unwrap();
            // let dx = (target.x + target.vx) - (source.x + source.vx);
            // let dy = (target.y + target.vy) - (source.y + source.vy);
            let dx = (target.x - source.x).max(1e-6);
            let dy = (target.y - source.y).max(1e-6);
            let l = (dx * dx + dy * dy).sqrt();
            let strength = link.strength / *source_count.min(target_count) as f32;
            let w = (l - link.length) / l * alpha * strength;
            {
                let ref mut target = points[link.target];
                target.vx -= dx * w * b;
                target.vy -= dy * w * b;
            }
            {
                let ref mut source = points[link.source];
                source.vx += dx * w * (1. - b);
                source.vy += dy * w * (1. - b);
            }
        }
    }
}

#[test]
fn test_link() {
    let mut links = Vec::new();
    links.push(Link::new(0, 1));
    links.push(Link::new(1, 3));
    links.push(Link::new(3, 2));
    links.push(Link::new(2, 0));
    let force = LinkForce::new(&links);
    let mut points = Vec::new();
    points.push(Point::new(10., 10.));
    points.push(Point::new(10., -10.));
    points.push(Point::new(-10., 10.));
    points.push(Point::new(-10., -10.));
    force.apply(&mut points, 1.);
    assert!(points[0].vx == 2.5);
    assert!(points[0].vy == 2.5);
    assert!(points[1].vx == 2.5);
    assert!(points[1].vy == -2.5);
    assert!(points[2].vx == -2.5);
    assert!(points[2].vy == 2.5);
    assert!(points[3].vx == -2.5);
    assert!(points[3].vy == -2.5);
}
