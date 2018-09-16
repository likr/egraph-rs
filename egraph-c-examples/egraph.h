#ifndef EGRAPH_H
#define EGRAPH_H

typedef struct {} Graph;
typedef struct {} Simulation;
typedef struct {} EdgeBundling;
typedef struct {} ForceDirectedGrouping;
typedef struct {} RadialGrouping;
typedef struct {} TreemapGrouping;
typedef struct {} Line;
typedef struct {} Point;
typedef struct {} Group;

Graph* graph_new();
unsigned graph_add_node(Graph*);
unsigned graph_add_edge(Graph*, unsigned, unsigned);
unsigned graph_node_count(Graph*);
unsigned graph_edge_count(Graph*);
double graph_get_x(Graph*, unsigned);
double graph_get_y(Graph*, unsigned);
void graph_set_x(Graph*, unsigned, double);
void graph_set_y(Graph*, unsigned, double);

Simulation* simulation_new();
unsigned simulation_add_center_force(Simulation*);
unsigned simulation_add_link_force(Simulation*, Graph*);
unsigned simulation_add_many_body_force(Simulation*);
unsigned simulation_add_group_center_force(Simulation*, Group*, unsigned, unsigned*, unsigned);
unsigned simulation_add_group_link_force(Simulation*, Graph*, unsigned*, double, double);
unsigned simulation_add_group_many_body_force(Simulation*, Graph*, unsigned, unsigned*, unsigned);
void simulation_start(Simulation*, Graph*);

EdgeBundling* edge_bundling_new();
unsigned edge_bundling_get_cycles(EdgeBundling*);
double edge_bundling_get_s0(EdgeBundling*);
unsigned edge_bundling_get_i0(EdgeBundling*);
double edge_bundling_get_s_step(EdgeBundling*);
double edge_bundling_get_i_step(EdgeBundling*);
void edge_bundling_set_cycles(EdgeBundling*, unsigned);
void edge_bundling_set_s0(EdgeBundling*, double);
void edge_bundling_set_i0(EdgeBundling*, unsigned);
void edge_bundling_set_s_step(EdgeBundling*, double);
void edge_bundling_set_i_step(EdgeBundling*, double);

Line* edge_bundling_call(EdgeBundling*, Graph*);

Line* lines_at(Line*, unsigned);
Point* line_points(Line*);
Point* line_points_at(Line*, unsigned);
unsigned line_points_length(Line*);
double point_x(Point*);
double point_y(Point*);

ForceDirectedGrouping force_directed_grouping_new(Graph*);
Group* force_directed_grouping_call(ForceDirectedGrouping*, double, double, double*, unsigned);
double force_directed_grouping_get_link_length(ForceDirectedGrouping*);
double force_directed_grouping_get_many_body_force_strength(ForceDirectedGrouping*);
double force_directed_grouping_get_many_link_strength(ForceDirectedGrouping*);
double force_directed_grouping_get_many_center_strength(ForceDirectedGrouping*);
void force_directed_grouping_set_link_length(ForceDirectedGrouping*, double);
void force_directed_grouping_set_many_body_force_strength(ForceDirectedGrouping*, double);
void force_directed_grouping_set_many_link_strength(ForceDirectedGrouping*, double);
void force_directed_grouping_set_many_center_strength(ForceDirectedGrouping*, double);

RadialGrouping radial_grouping_new();
Group* radial_grouping_call(RadialGrouping*, double, double, double*, unsigned);

TreemapGrouping treemap_grouping_new();
Group* treemap_grouping_call(TreemapGrouping*, double, double, double*, unsigned);

Group* groups_at(Group*, unsigned);
double group_get_x(Group*);
double group_get_y(Group*);
double group_get_width(Group*);
double group_get_height(Group*);
void group_set_x(Group*, double);
void group_set_y(Group*, double);
void group_set_width(Group*, double);
void group_set_height(Group*, double);

#endif
