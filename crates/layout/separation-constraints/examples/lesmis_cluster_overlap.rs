use egraph_dataset::dataset_lesmis;
use petgraph::graph::UnGraph;
use petgraph::visit::IntoNodeIdentifiers;
use petgraph_clustering::{CommunityDetection, LabelPropagation};
use petgraph_layout_mds::ClassicalMds;
use petgraph_layout_separation_constraints::{
    project_clustered_rectangle_no_overlap_constraints, project_rectangle_no_overlap_constraints_2d,
};
use petgraph_layout_sgd::{FullSgd, Scheduler, SchedulerExponential, Sgd};
use plotters::prelude::*;
use plotters::style::RGBColor;
use rand::thread_rng;
use std::collections::HashMap;
use std::error::Error;
use std::f32;

// Function to get a color for each community
fn get_community_color(community_index: usize) -> RGBColor {
    // Define a set of distinct colors
    let colors = [
        RGBColor(230, 25, 75),   // Red
        RGBColor(60, 180, 75),   // Green
        RGBColor(255, 225, 25),  // Yellow
        RGBColor(0, 130, 200),   // Blue
        RGBColor(245, 130, 48),  // Orange
        RGBColor(145, 30, 180),  // Purple
        RGBColor(70, 240, 240),  // Cyan
        RGBColor(240, 50, 230),  // Magenta
        RGBColor(210, 245, 60),  // Lime
        RGBColor(250, 190, 212), // Pink
    ];

    // Use modulo to cycle through colors if there are more communities than colors
    colors[community_index % colors.len()]
}

fn main() -> Result<(), Box<dyn Error>> {
    let graph: UnGraph<(), ()> = dataset_lesmis();
    println!(
        "Graph loaded: {} nodes, {} edges",
        graph.node_count(),
        graph.edge_count()
    );

    let community_detection = LabelPropagation::with_seed(0);
    let communities = community_detection.detect_communities(&graph);

    let edge_length = 150.;
    let iterations = 1000;
    let node_size = [20., 20.];
    let mds = ClassicalMds::new(&graph, |_| edge_length);
    let mut drawing = mds.run_2d();
    let mut sgd = FullSgd::new(&graph, |_| edge_length);
    let mut scheduler = sgd.scheduler::<SchedulerExponential<f32>>(iterations, 0.1);
    let mut rng = thread_rng();

    scheduler.run(&mut |eta| {
        sgd.shuffle(&mut rng);
        sgd.apply(&mut drawing, eta);

        project_rectangle_no_overlap_constraints_2d(&mut drawing, |_, d| node_size[d]);
        project_clustered_rectangle_no_overlap_constraints(
            &graph,
            &mut drawing,
            |node_id| communities[&node_id],
            |_, d| node_size[d],
        );
    });
    drawing.centralize();

    // Output final statistics
    println!("\nFinal layout statistics:");
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    // Calculate community bounding boxes
    let mut community_bounds: HashMap<usize, (f32, f32, f32, f32)> = HashMap::new();

    // Initialize community bounds
    for &community_id in communities.values() {
        community_bounds.insert(
            community_id,
            (
                f32::INFINITY,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
            ),
        );
    }

    // Calculate the bounding box of the layout and community bounds
    for u in graph.node_identifiers() {
        let x = drawing.x(u).unwrap();
        let y = drawing.y(u).unwrap();
        let half_width = node_size[0] / 2.0;
        let half_height = node_size[1] / 2.0;

        // Update overall layout bounds
        min_x = min_x.min(x - half_width);
        max_x = max_x.max(x + half_width);
        min_y = min_y.min(y - half_height);
        max_y = max_y.max(y + half_height);

        // Update community bounds
        let community_id = communities[&u];
        let (c_min_x, c_min_y, c_max_x, c_max_y) = community_bounds[&community_id];
        community_bounds.insert(
            community_id,
            (
                c_min_x.min(x - half_width),
                c_min_y.min(y - half_height),
                c_max_x.max(x + half_width),
                c_max_y.max(y + half_height),
            ),
        );
    }

    println!("X range: {} to {}", min_x, max_x);
    println!("Y range: {} to {}", min_y, max_y);
    println!("Width: {}", max_x - min_x);
    println!("Height: {}", max_y - min_y);

    // Calculate padding for the chart
    let padding = (max_x - min_x).max(max_y - min_y) * 0.05;
    let x_range = (min_x - padding)..(max_x + padding);
    let y_range = (min_y - padding)..(max_y + padding);

    // Plotting the layout
    println!("Drawing layout to lesmis_layout.png...");
    let output_path = "lesmis_layout.png";
    let root = BitMapBackend::new(
        output_path,
        (
            (x_range.end - x_range.start) as u32,
            (y_range.end - y_range.start) as u32,
        ),
    )
    .into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Lesmis Layout with Cluster Overlap Constraints",
            ("sans-serif", 20),
        )
        .margin(10)
        .x_label_area_size(40) // Add space for labels
        .y_label_area_size(40) // Add space for labels
        .build_cartesian_2d(x_range, y_range)?;

    chart.configure_mesh().draw()?;

    // Draw community rectangles first (lowest z-order)
    println!("Drawing {} community rectangles", community_bounds.len());
    for (community_id, (min_x, min_y, max_x, max_y)) in &community_bounds {
        let color = get_community_color(*community_id);
        chart.draw_series(std::iter::once(Rectangle::new(
            [(*min_x, *min_y), (*max_x, *max_y)],
            color.mix(0.2).filled(), // 20% opacity
        )))?;
    }

    // Draw edges
    chart.draw_series(graph.edge_indices().map(|edge_index| {
        let (source, target) = graph.edge_endpoints(edge_index).unwrap();
        PathElement::new(
            vec![
                (drawing.x(source).unwrap(), drawing.y(source).unwrap()),
                (drawing.x(target).unwrap(), drawing.y(target).unwrap()),
            ],
            ShapeStyle {
                color: RGBAColor(0, 0, 0, 0.3), // Translucent black
                filled: false,
                stroke_width: 1,
            },
        )
    }))?;

    // Draw nodes colored by community
    chart.draw_series(graph.node_indices().map(|node_index| {
        let community_index = communities[&node_index];
        let color = get_community_color(community_index);

        Rectangle::new(
            [
                (
                    drawing.x(node_index).unwrap() - node_size[0] / 2.,
                    drawing.y(node_index).unwrap() - node_size[1] / 2.,
                ),
                (
                    drawing.x(node_index).unwrap() + node_size[0] / 2.,
                    drawing.y(node_index).unwrap() + node_size[1] / 2.,
                ),
            ],
            color.filled().stroke_width(1),
        )
    }))?;

    root.present()?;
    println!("Layout saved to {}", output_path);

    Ok(()) // Add Ok(()) at the end of main function
}
