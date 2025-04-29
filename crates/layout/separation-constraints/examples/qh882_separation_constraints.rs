use egraph_dataset::dataset_qh882;
use petgraph::graph::{NodeIndex, UnGraph};
use petgraph_drawing::DrawingEuclidean2d;
use petgraph_layout_separation_constraints::{
    project_1d, project_rectangle_no_overlap_constraints_2d, Constraint,
};
use petgraph_layout_stress_majorization::StressMajorization;
use plotters::prelude::*;
use std::error::Error;
use std::f32;

fn main() -> Result<(), Box<dyn Error>> {
    // Load the QH882 graph dataset
    let graph: UnGraph<(), ()> = dataset_qh882();
    println!(
        "Graph loaded: {} nodes, {} edges",
        graph.node_count(),
        graph.edge_count()
    );

    // Initialize layout using DrawingEuclidean2d
    let mut drawing = DrawingEuclidean2d::initial_placement(&graph);
    // Initialize StressMajorization with DrawingEuclidean2d
    let mut stress = StressMajorization::new(&graph, &drawing, |_| 100.0);

    // Optimize initial layout using Stress Majorization
    println!("Optimizing initial layout...");
    for _ in 0..50 {
        stress.apply(&mut drawing);
    }
    drawing.centralize();

    // Generate constraints
    // Example: Set constraints between the left half and right half of the graph based on x-coordinates
    let n = graph.node_count();
    let mut constraints = Vec::new();

    // Sort nodes based on current x-coordinates
    let mut nodes: Vec<NodeIndex<u32>> = graph.node_indices().collect();
    nodes.sort_by(|&a, &b| {
        drawing
            .x(a)
            .unwrap()
            .partial_cmp(&drawing.x(b).unwrap())
            .unwrap()
    });

    // Set constraints between left half and right half nodes
    let mid = n / 2;
    let min_distance = (drawing.x(nodes[mid]).unwrap() - drawing.x(nodes[0]).unwrap()) * 2.0;
    for i in 0..mid {
        let j = mid + i;
        let left = nodes[i].index();
        let right = nodes[j].index();
        constraints.push(Constraint::new(left, right, min_distance));
    }
    let size = 30.;

    println!("Created {} separation constraints", constraints.len());

    // Optimize layout using stress-majorization and constraints
    println!("Optimizing layout with constraints...");
    for iteration in 0..100 {
        // Apply one step of stress-majorization
        let stress = stress.apply(&mut drawing);

        // Apply constraints by projecting coordinates
        project_1d(&mut drawing, 0, &constraints);
        project_rectangle_no_overlap_constraints_2d(&mut drawing, |_, _| size);

        drawing.centralize();

        if iteration % 10 == 0 {
            println!("Iteration {}: stress = {}", iteration, stress);
        }
    }

    // Output final statistics
    println!("\nFinal layout statistics:");
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    // Calculate the bounding box of the layout
    for i in 0..n {
        let idx = NodeIndex::new(i);
        let x = drawing.x(idx).unwrap();
        let y = drawing.y(idx).unwrap();
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
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
    println!("Drawing layout to qh882_layout.png...");
    let output_path = "qh882_layout.png";
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
            "QH882 Layout with Separation Constraints",
            ("sans-serif", 20),
        )
        .margin(10)
        .x_label_area_size(40) // Add space for labels
        .y_label_area_size(40) // Add space for labels
        .build_cartesian_2d(x_range, y_range)?;

    chart.configure_mesh().draw()?;

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

    // Draw nodes
    chart.draw_series(graph.node_indices().map(|node_index| {
        Rectangle::new(
            [
                (
                    drawing.x(node_index).unwrap() - size / 2.,
                    drawing.y(node_index).unwrap() - size / 2.,
                ),
                (
                    drawing.x(node_index).unwrap() + size / 2.,
                    drawing.y(node_index).unwrap() + size / 2.,
                ),
            ],
            RED.stroke_width(1),
        ) // Red filled circle
    }))?;

    root.present()?;
    println!("Layout saved to {}", output_path);

    Ok(()) // Add Ok(()) at the end of main function
}
