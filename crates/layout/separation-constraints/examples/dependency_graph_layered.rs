use petgraph::graph::{DiGraph, Graph};
use petgraph_algorithm_layering::{remove_cycle, LongestPath};
use petgraph_layout_mds::ClassicalMds;
use petgraph_layout_separation_constraints::project_rectangle_no_overlap_constraints_2d;
use petgraph_layout_sgd::{FullSgd, Scheduler, SchedulerExponential, Sgd};
use plotters::prelude::*;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::f32;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

// Define structures to deserialize the JSON data
#[derive(Serialize, Deserialize, Debug)]
struct Node {
    id: usize,
    name: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct Link {
    source: usize,
    target: usize,
}

#[derive(Serialize, Deserialize, Debug)]
struct DependencyGraph {
    nodes: Vec<Node>,
    links: Vec<Link>,
}

fn main() -> Result<(), Box<dyn Error>> {
    // Load the dependency graph from JSON file
    println!("Loading dependency graph from JSON...");
    // When running with cargo run --example, the current directory is the project root
    let file = File::open(Path::new(
        "crates/layout/separation-constraints/examples/dependency-graph.json",
    ))
    .or_else(|_| {
        // Fallback to a relative path if running from the crate directory
        File::open(Path::new("examples/dependency-graph.json"))
    })?;
    let reader = BufReader::new(file);
    let data: DependencyGraph = serde_json::from_reader(reader)?;

    println!(
        "Dependency graph loaded: {} nodes, {} links",
        data.nodes.len(),
        data.links.len()
    );

    // Create a directed graph from the JSON data
    let mut graph = DiGraph::<String, ()>::new();
    let mut node_indices = HashMap::new();

    // Add only a small subset of nodes to the graph
    for node in data.nodes.iter() {
        let idx = graph.add_node(node.name.clone());
        node_indices.insert(node.id, idx);
    }

    // Add edges to the graph, skipping self-loops
    let mut added_edges = std::collections::HashSet::new();
    for link in &data.links {
        if link.source == link.target {
            // Skip self-loops
            continue;
        }

        if let (Some(&source), Some(&target)) = (
            node_indices.get(&link.source),
            node_indices.get(&link.target),
        ) {
            // Skip duplicate edges
            let edge_key = (source.index(), target.index());
            if !added_edges.contains(&edge_key) {
                graph.add_edge(source, target, ());
                added_edges.insert(edge_key);
            }
        }
    }

    println!(
        "Graph created: {} nodes, {} edges",
        graph.node_count(),
        graph.edge_count()
    );

    // Create an undirected graph with the same structure for stress majorization
    println!("Creating undirected graph for stress majorization...");
    let mut undirected_graph = Graph::new_undirected();
    let mut undirected_node_indices = HashMap::new();

    // Add nodes to the undirected graph
    for node_idx in graph.node_indices() {
        let node_name = graph[node_idx].clone();
        let undirected_idx = undirected_graph.add_node(node_name);
        undirected_node_indices.insert(node_idx.index(), undirected_idx);
    }

    // Add edges to the undirected graph
    for edge_idx in graph.edge_indices() {
        let (source, target) = graph.edge_endpoints(edge_idx).unwrap();
        let undirected_source = undirected_node_indices[&source.index()];
        let undirected_target = undirected_node_indices[&target.index()];
        undirected_graph.add_edge(undirected_source, undirected_target, ());
    }

    // Constraint parameters
    let node_width = 50.0;
    let node_height = 20.0;
    let node_separation = 10.0;
    let layer_distance = 100.0;
    let edge_length = 100.0;

    // Generate layered constraints for hierarchical layout
    println!("Generating layered constraints...");
    remove_cycle(&mut graph);
    let layering = LongestPath::new();
    let layers = layering.assign_layers(&graph);
    let layer_y = layers
        .into_iter()
        .map(|(node_id, layer)| (node_id, layer as f32 * -layer_distance))
        .collect::<HashMap<_, _>>();

    // Initialize layout using DrawingEuclidean2d with the undirected graph
    let mds = ClassicalMds::new(&undirected_graph, |_| edge_length);
    let mut drawing = mds.run_2d();

    // Initialize SGD with the undirected graph
    let iterations = 100;
    let mut sgd = FullSgd::new(&undirected_graph, |_| edge_length);
    let mut scheduler = sgd.scheduler::<SchedulerExponential<f32>>(iterations, 0.5);
    let mut rng = thread_rng();

    // Optimize layout using stress-majorization and constraints
    println!("Optimizing layout with constraints...");
    for _i in 0..iterations {
        scheduler.step(&mut |t| {
            sgd.shuffle(&mut rng);
            sgd.apply(&mut drawing, t);

            // Apply layered constraints to the y-dimension (vertical)
            for node_id in undirected_graph.node_indices() {
                drawing.set_y(node_id, layer_y[&node_id]);
            }
            // Apply rectangle overlap constraints to both dimensions
            project_rectangle_no_overlap_constraints_2d(&mut drawing, |_, d| {
                if d == 0 {
                    node_width + node_separation
                } else {
                    node_height
                }
            });
        });
    }
    drawing.centralize();

    // Output final statistics
    println!("\nFinal layout statistics:");
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    // Calculate the bounding box of the layout
    for node_idx in graph.node_indices() {
        let x = drawing.x(node_idx).unwrap();
        let y = drawing.y(node_idx).unwrap();
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }

    println!("X range: {} to {}", min_x, max_x);
    println!("Y range: {} to {}", min_y, max_y);
    println!("Width: {}", max_x - min_x);
    println!("Height: {}", max_y - min_y);

    // Ensure we have a valid bounding box with non-zero dimensions
    if max_x <= min_x || max_y <= min_y {
        println!("Warning: Invalid bounding box. Adjusting dimensions...");
        min_x = -500.0;
        max_x = 500.0;
        min_y = -500.0;
        max_y = 500.0;
    }

    // Calculate padding for the chart
    let padding = (max_x - min_x).max(max_y - min_y) * 0.05;
    let x_range = (min_x - padding)..(max_x + padding);
    let y_range = (min_y - padding)..(max_y + padding);

    // Plotting the layout
    println!("Drawing layout to dependency_graph_layered.png...");
    let output_path = "dependency_graph_layered.png";

    // Create a bitmap with appropriate dimensions
    let width = (x_range.end - x_range.start) as u32;
    let height = (y_range.end - y_range.start) as u32;

    let root = BitMapBackend::new(output_path, (width, height)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Dependency Graph with Layered Constraints",
            ("sans-serif", 20),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_range, y_range)?;

    chart.configure_mesh().draw()?;

    // Draw edges
    chart.draw_series(graph.edge_indices().map(|edge_index| {
        let (source, target) = graph.edge_endpoints(edge_index).unwrap();
        let undirected_source = node_indices[&source.index()];
        let undirected_target = node_indices[&target.index()];

        let source_x = drawing.x(undirected_source).unwrap();
        let source_y = drawing.y(undirected_source).unwrap();
        let target_x = drawing.x(undirected_target).unwrap();
        let target_y = drawing.y(undirected_target).unwrap();

        // Draw an arrow from source to target
        PathElement::new(
            vec![(source_x, source_y), (target_x, target_y)],
            ShapeStyle {
                color: RGBAColor(0, 0, 0, 0.3), // Translucent black
                filled: false,
                stroke_width: 1,
            },
        )
    }))?;

    // Draw nodes as rectangles with colors based on their y-position
    for node_idx in graph.node_indices() {
        let x = drawing.x(node_idx).unwrap();
        let y = drawing.y(node_idx).unwrap();

        // Draw node rectangle
        chart.draw_series(std::iter::once(Rectangle::new(
            [
                (x - node_width / 2.0, y - node_height / 2.0),
                (x + node_width / 2.0, y + node_height / 2.0),
            ],
            RED.filled().stroke_width(1),
        )))?;
    }

    root.present()?;
    println!("Layout saved to {}", output_path);

    println!("Example completed successfully!");

    Ok(())
}
