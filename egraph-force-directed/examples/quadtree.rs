extern crate rand;
extern crate fd_layout;

use fd_layout::quadtree::{Quadtree, NodeId, Element, Rect};
use rand::distributions::{Range, IndependentSample};

fn print_rect(rect: Rect, color: &str) {
    println!(
        "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{}\" stroke=\"black\"/>",
        rect.left() as i32,
        rect.bottom() as i32,
        rect.width() as i32,
        rect.height() as i32,
        color,
    );
}

fn print_circle(x: f32, y: f32, n: usize) {
    println!(
        "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"red\" />",
        x as i32,
        y as i32,
        (2. * (n as f32).sqrt()) as i32
    )
}

fn walk(tree: &Quadtree<()>, node_id: NodeId) {
    let rect = tree.rect(node_id);
    print_rect(rect, "none");
    for &(ref e, region) in tree.elements(node_id).iter() {
        match **e {
            Element::Leaf { x, y, n } => {
                let sub_rect = rect.sub_rect(region);
                print_rect(sub_rect, "#eee");
                print_circle(x, y, n);
            }
            Element::Node { node_id } => walk(tree, node_id),
            Element::Empty => {
                let sub_rect = rect.sub_rect(region);
                print_rect(sub_rect, "none");
            }
        }
    }
}

fn generate(width: f32, height: f32, n: usize) -> Quadtree<()> {
    let mut tree = Quadtree::new(Rect {
        cx: 0.,
        cy: 0.,
        width: width,
        height: height,
    });
    let root = tree.root();

    let width_range = Range::new(-width / 2., width / 2.);
    let height_range = Range::new(-height / 2., height / 2.);
    let mut rng = rand::thread_rng();
    for _ in 0..n {
        tree.insert(
            root,
            width_range.ind_sample(&mut rng),
            height_range.ind_sample(&mut rng),
        );
    }
    tree
}

fn main() {
    let width = 1000.;
    let height = 1000.;
    let margin = 10.;
    let tree = generate(width, height, 500);
    let root = tree.root();

    println!(
        "<svg version=\"1.1\" width=\"{}\" height=\"{}\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">",
        width + margin * 2., height + margin * 2.,
    );
    println!(
        "<g transform=\"translate({},{})\">",
        width / 2. + margin,
        height / 2. + margin,
    );
    walk(&tree, root);
    println!("</g>\n</svg>");
}
