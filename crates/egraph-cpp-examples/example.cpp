#include <iostream>
#include "../egraph-c-api/egraph.h"

int main(void) {
  Graph* graph = graph_new();
  graph_add_node(graph);
  graph_add_node(graph);
  graph_add_node(graph);
  graph_add_node(graph);
  graph_add_edge(graph, 0, 1);
  graph_add_edge(graph, 1, 2);
  graph_add_edge(graph, 2, 3);
  graph_add_edge(graph, 3, 0);

  printf("number of nodes: %d\n", graph_node_count(graph));
  printf("number of edges: %d\n", graph_edge_count(graph));

  Simulation* simulation = simulation_new();
  simulation_add_many_body_force(simulation);
  simulation_add_link_force(simulation, graph);
  simulation_add_center_force(simulation);
  simulation_start(simulation, graph);

  EdgeBundling* edge_bundling = edge_bundling_new();
  Line* bundling_result = edge_bundling_call(edge_bundling, graph);

  for (int i = 0, n = graph_node_count(graph); i < n; ++i) {
    std::cout << "node[" << i << "] = {x: " << graph_get_x(graph, i) << ", y: " << graph_get_y(graph, i) << "}" << std::endl;
  }


  for (int i = 0, n = graph_edge_count(graph); i < n; ++i) {
    std::cout << "edge[" << i << "] = [" << std::endl;
    Line* line = lines_at(bundling_result, i);
    for (int j = 0, length = line_points_length(line); j < length; ++j) {
      Point* point = line_points_at(line, j);
      std::cout << "  [" << point_x(point) << ", " << point_y(point) << "]," << std::endl;
    }
    std::cout << "]" << std::endl;
  }
}
