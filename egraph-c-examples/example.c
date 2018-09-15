#include <stdio.h>
#include "egraph.h"



int main(void) {
  Graph graph = graph_new();
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

  Simulation simulation = simulation_new();
  simulation_add_many_body_force(simulation);
  simulation_add_link_force(simulation, graph);
  simulation_add_center_force(simulation);
  simulation_start(simulation, graph);

  EdgeBundling edge_bundling = edge_bundling_new();
  Line bundling_result = edge_bundling_call(edge_bundling, graph);

  for (int i = 0, n = graph_node_count(graph); i < n; ++i) {
    printf("node[%d] = {x: %f, y: %f}\n", i, graph_get_x(graph, i), graph_get_y(graph, i));
  }


  for (int i = 0, n = graph_edge_count(graph); i < n; ++i) {
    printf("edge[%d] = [\n", i);
    Line line = lines_at(bundling_result, i);
    for (int j = 0, length = line_points_length(line); j < length; ++j) {
      Point point = line_points_at(line, j);
      printf("  [%f, %f],\n", point_x(point), point_y(point));
    }
    printf("]\n");
  }
}
