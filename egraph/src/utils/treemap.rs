#[derive(Copy, Clone, Debug)]
pub struct Tile {
    pub x: f64,
    pub y: f64,
    pub dx: f64,
    pub dy: f64,
}

enum Direction {
    Vertical,
    Horizontal,
}

fn improve(row: &Vec<f64>, w: f64, item: f64) -> bool {
    let r_plus = row.first().unwrap();
    let r_minus0 = row.last().unwrap();
    let r_minus1 = item;
    let s0 : f64 = row.iter().sum();
    let s1 = s0 + item;
    let w2 = w * w;
    let v0 = s0 * s0 / w2;
    let v1 = s1 * s1 / w2;
    (r_plus / v1).max(v1 / r_minus1) < (r_plus / v0).max(v0 / r_minus0)
}

pub fn squarify(width: f64, height: f64, values: &Vec<f64>) -> Vec<Tile> {
    let mut result = vec![];
    let mut w = width;
    let mut h = height;
    let mut x = 0.;
    let mut y = 0.;
    let mut row = vec![];
    let mut direction = if w > h {
        Direction::Horizontal
    } else {
        Direction::Vertical
    };
    for value in values {
        let size = match direction {
            Direction::Vertical => w,
            Direction::Horizontal => h,
        };
        if row.is_empty() || improve(&row, size, *value) {
            row.push(*value);
            continue;
        }
        let s : f64 = row.iter().sum();
        let d = match direction {
            Direction::Vertical => s / w,
            Direction::Horizontal => s / h,
        };
        let mut offset = 0.;
        for z in &row {
            let inc = z / d;
            result.push(match direction {
                Direction::Vertical => {
                    Tile {
                        x: x + offset,
                        y: y,
                        dx: inc,
                        dy: d,
                    }
                },
                Direction::Horizontal => {
                    Tile {
                        x: x,
                        y: y + offset,
                        dx: d,
                        dy: inc,
                    }
                },
            });
            offset += inc;
        }
        row.clear();
        row.push(*value);
        match direction {
            Direction::Vertical => {
                y += d;
                h -= d;
            },
            Direction::Horizontal => {
                x += d;
                w -= d;
            },
        }
        direction = if w > h {
            Direction::Horizontal
        } else {
            Direction::Vertical
        };
    }
    if !row.is_empty() {
        let s : f64 = row.iter().sum();
        let d = match direction {
            Direction::Vertical => s / w,
            Direction::Horizontal => s / h,
        };
        let mut offset = 0.;
        for z in &row {
            let inc = z / d;
            result.push(match direction {
                Direction::Vertical => {
                    Tile {
                        x: x + offset,
                        y: y,
                        dx: inc,
                        dy: d,
                    }
                },
                Direction::Horizontal => {
                    Tile {
                        x: x,
                        y: y + offset,
                        dx: d,
                        dy: inc,
                    }
                },
            });
            offset += inc;
        }
    }
    result
}

pub fn normalize(values: &mut Vec<f64>, total_area: f64) {
    let total_size = values.iter().sum::<f64>();
    for i in 0..values.len() {
        values[i] = values[i] * total_area / total_size;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let values = vec![6., 6., 4., 3., 2., 2., 1.];
        for tile in squarify(6., 4., &values) {
            println!("{} {} {} {}", tile.x, tile.y, tile.dx, tile.dy);
        }
    }

    #[test]
    fn it_works2() {
        let mut values = vec![14., 13., 11., 10., 10., 10., 4., 2., 2., 1.];
        normalize(&mut values, 960. * 600.);
        for tile in squarify(960., 600., &values) {
            println!("{} {} {} {}", tile.x, tile.y, tile.dx, tile.dy);
        }
    }
}
