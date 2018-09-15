#ifndef EGRAPH_H
#define EGRAPH_H

typedef void* Graph;
typedef void* Simulation;
typedef void* EdgeBundling;
typedef void* Line;
typedef void* Point;

Graph graph_new();
unsigned graph_add_node(Graph);
unsigned graph_add_edge(Graph, unsigned, unsigned);
unsigned graph_node_count(Graph);
unsigned graph_edge_count(Graph);
double graph_get_x(Graph, unsigned);
double graph_get_y(Graph, unsigned);
void graph_set_x(Graph, unsigned, double);
void graph_set_y(Graph, unsigned, double);

Simulation simulation_new();
unsigned simulation_add_center_force(Simulation);
unsigned simulation_add_link_force(Simulation, Graph);
unsigned simulation_add_many_body_force(Simulation);
void simulation_start(Simulation, Graph);

EdgeBundling edge_bundling_new();
Line edge_bundling_call(EdgeBundling, Graph);

Line lines_at(Line, unsigned);
Point line_points(Line);
Point line_points_at(Line, unsigned);
unsigned line_points_length(Line);
double point_x(Point);
double point_y(Point);

#endif
