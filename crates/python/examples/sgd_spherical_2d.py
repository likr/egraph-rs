import json
import networkx as nx
import egraph as eg


def main():
    nx_graph = nx.les_miserables_graph()
    graph = eg.Graph()
    indices = {}
    for u in nx_graph.nodes:
        indices[u] = graph.add_node(u)
    for u, v in nx_graph.edges:
        graph.add_edge(indices[u], indices[v], (u, v))

    drawing = eg.DrawingSpherical2d.initial_placement(graph)
    rng = eg.Rng.seed_from(0)
    sgd = eg.FullSgd(
        graph,
        lambda _: 0.6,
    )
    scheduler = sgd.scheduler(
        100,
        0.1,
    )

    def step(eta):
        sgd.shuffle(rng)
        sgd.apply(drawing, eta)
    scheduler.run(step)

    for u, i in indices.items():
        nx_graph.nodes[u]['lon'] = drawing.lon(i)
        nx_graph.nodes[u]['lat'] = drawing.lat(i)
    json.dump(nx.node_link_data(nx_graph),
              open('tmp/graph_sphere.json', 'w'),
              ensure_ascii=False)


if __name__ == '__main__':
    main()
