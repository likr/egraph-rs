use ::layout::force_directed::force::Group;
use ::utils::treemap;

pub fn assign(width: usize, height: usize, num_groups: usize, node_groups: &Vec<usize>) -> Vec<Group> {
    let mut group_sizes = vec![0 as usize; num_groups];
    for &g in node_groups {
        group_sizes[g] += 1;
    }

    let mut group_indices = (0..num_groups).collect::<Vec<_>>();
    group_indices.sort_by_key(|&i| group_sizes[i]);
    group_indices.reverse();

    let mut group_sizes = group_indices.iter().map(|&i| group_sizes[i] as f64).collect::<Vec<_>>();
    treemap::normalize(&mut group_sizes, (width * height) as f64);

    let tiles = treemap::squarify(width as f64, height as f64, &group_sizes);
    let mut groups = vec![Group::new(0., 0.); num_groups];
    for (&i, t) in group_indices.iter().zip(tiles) {
        groups[i].x = (t.x + t.dx / 2.) as f32;
        groups[i].y = (t.y + t.dy / 2.) as f32;
    }
    groups
}
