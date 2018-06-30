extern crate petgraph;
extern crate petgraph_sugiyama_layout;
extern crate xml;

use petgraph::Graph;
use petgraph_sugiyama_layout::graph::{Edge, Node};
use petgraph_sugiyama_layout::sugiyama_layout::SugiyamaLayout;
use std::fs::File;
use std::io::Write;
use xml::writer::{EmitterConfig, EventWriter, XmlEvent};

// fn example_graph() -> Graph<(), ()> {
//     let mut graph = Graph::new();
//     let a1 = graph.add_node(());
//     let a2 = graph.add_node(());
//     let a3 = graph.add_node(());
//     let b1 = graph.add_node(());
//     let b2 = graph.add_node(());
//     let b3 = graph.add_node(());
//     let c1 = graph.add_node(());
//     let c2 = graph.add_node(());
//     let c3 = graph.add_node(());
//     let d1 = graph.add_node(());
//     let d2 = graph.add_node(());
//     let d3 = graph.add_node(());
//     graph.add_edge(a1, b2, ());
//     graph.add_edge(a2, b1, ());
//     graph.add_edge(a3, b1, ());
//     graph.add_edge(b1, c1, ());
//     graph.add_edge(b2, c1, ());
//     graph.add_edge(b2, c2, ());
//     graph.add_edge(b2, c3, ());
//     graph.add_edge(b3, c2, ());
//     graph.add_edge(c1, d3, ());
//     graph.add_edge(c2, d1, ());
//     graph.add_edge(c2, d2, ());
//     graph
// }

fn example_graph() -> Graph<(), ()> {
    let mut graph = Graph::new();
    let a1 = graph.add_node(());
    let a2 = graph.add_node(());
    let b1 = graph.add_node(());
    let b2 = graph.add_node(());
    let b3 = graph.add_node(());
    let b4 = graph.add_node(());
    let b5 = graph.add_node(());
    let b6 = graph.add_node(());
    let b7 = graph.add_node(());
    let b8 = graph.add_node(());
    let c1 = graph.add_node(());
    let c2 = graph.add_node(());
    let c3 = graph.add_node(());
    let c4 = graph.add_node(());
    let c5 = graph.add_node(());
    let c6 = graph.add_node(());
    let d1 = graph.add_node(());
    let d2 = graph.add_node(());
    let d3 = graph.add_node(());
    let d4 = graph.add_node(());
    let d5 = graph.add_node(());
    let d6 = graph.add_node(());
    let d7 = graph.add_node(());
    let e1 = graph.add_node(());
    let e2 = graph.add_node(());
    let e3 = graph.add_node(());
    graph.add_edge(a1, b1, ());
    graph.add_edge(a1, b6, ());
    graph.add_edge(a1, b8, ());
    graph.add_edge(a2, b3, ());
    graph.add_edge(a2, b5, ());
    graph.add_edge(b2, c2, ());
    graph.add_edge(b3, c2, ());
    graph.add_edge(b4, c2, ());
    graph.add_edge(b5, c3, ());
    graph.add_edge(b6, c4, ());
    graph.add_edge(b7, c2, ());
    graph.add_edge(b7, c6, ());
    graph.add_edge(b8, c2, ());
    graph.add_edge(b8, c5, ());
    graph.add_edge(c1, d1, ());
    graph.add_edge(c1, d2, ());
    graph.add_edge(c1, d6, ());
    graph.add_edge(c3, d4, ());
    graph.add_edge(c4, d5, ());
    graph.add_edge(c5, d6, ());
    graph.add_edge(c6, d3, ());
    graph.add_edge(c6, d7, ());
    graph.add_edge(d1, e1, ());
    graph.add_edge(d1, e2, ());
    graph.add_edge(d2, e2, ());
    graph.add_edge(d3, e1, ());
    graph.add_edge(d4, e3, ());
    graph.add_edge(d5, e3, ());
    graph.add_edge(d6, e3, ());
    graph.add_edge(d7, e3, ());
    graph
}

fn layout<N, E>(graph: &Graph<N, E>) -> Graph<Node, Edge> {
    let sugiyama_layout = SugiyamaLayout::new();
    sugiyama_layout.call(&graph)
}

fn write<'a, W: Write, E: Into<XmlEvent<'a>>>(w: &mut EventWriter<W>, event: E) {
    w.write(event).ok().unwrap()
}

fn write_to_svg(layout: &Graph<Node, Edge>) {
    let mut file = File::create("output.svg").unwrap();
    let mut writer = EmitterConfig::new()
        .perform_indent(true)
        .create_writer(&mut file);

    let svg = XmlEvent::start_element("svg").attr("xmlns", "http://www.w3.org/2000/svg");
    write(&mut writer, svg);
    {
        let edges = XmlEvent::start_element("g").attr("class", "edges");
        write(&mut writer, edges);
        for e in layout.edge_indices() {
            let (u, v) = layout.edge_endpoints(e).unwrap();
            let ref u_node = layout[u];
            let ref v_node = layout[v];
            let g = XmlEvent::start_element("g");
            write(&mut writer, g);
            {
                let x1 = &format!("{}", u_node.x);
                let y1 = &format!("{}", u_node.y);
                let x2 = &format!("{}", v_node.x);
                let y2 = &format!("{}", v_node.y);
                let line = XmlEvent::start_element("line")
                    .attr("x1", x1)
                    .attr("y1", y1)
                    .attr("x2", x2)
                    .attr("y2", y2)
                    .attr("stroke", "black");
                write(&mut writer, line);
                write(&mut writer, XmlEvent::end_element());
            }
            write(&mut writer, XmlEvent::end_element());
        }
        write(&mut writer, XmlEvent::end_element());
    }
    {
        let nodes = XmlEvent::start_element("g").attr("class", "nodes");
        write(&mut writer, nodes);
        for u in layout.node_indices() {
            let ref node = layout[u];
            let transform = &format!(
                "translate({},{})",
                node.x - node.width as i32 / 2,
                node.y - node.height as i32 / 2
            );
            let g = XmlEvent::start_element("g").attr("transform", transform);
            write(&mut writer, g);
            {
                let width = &format!("{}", node.width);
                let height = &format!("{}", node.height);
                let rect = XmlEvent::start_element("rect")
                    .attr("width", width)
                    .attr("height", height)
                    .attr("fill", "none")
                    .attr("stroke", "black")
                    .attr("stroke-width", "1");
                write(&mut writer, rect);
                write(&mut writer, XmlEvent::end_element());
            }
            {
                let x = &format!("{}", node.width / 2);
                let y = &format!("{}", node.height / 2 + 8);
                let text = XmlEvent::start_element("text")
                    .attr("x", x)
                    .attr("y", y)
                    .attr("text-anchor", "middle");
                write(&mut writer, text);
                let content = &format!("{}", u.index());
                write(&mut writer, XmlEvent::characters(content));
                write(&mut writer, XmlEvent::end_element());
            }
            write(&mut writer, XmlEvent::end_element());
        }
        write(&mut writer, XmlEvent::end_element());
    }
    write(&mut writer, XmlEvent::end_element());
}

fn main() {
    let graph = example_graph();
    let layout = layout(&graph);
    write_to_svg(&layout);
}
