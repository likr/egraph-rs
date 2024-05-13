import networkx as nx
import egraph as eg
import matplotlib.pyplot as plt


def main():
    nx_graph = nx.les_miserables_graph()
    graph = eg.Graph()
    indices = {}
    for u in nx_graph.nodes:
        indices[u] = graph.add_node(u)
    for u, v in nx_graph.edges:
        graph.add_edge(indices[u], indices[v], (u, v))

    size = nx.diameter(nx_graph) * 1.5
    d = eg.all_sources_bfs(
        graph,
        1 / size,  # edge length
    )
    drawing = eg.DrawingTorus2d.initial_placement(graph)
    rng = eg.Rng.seed_from(0)  # random seed
    sgd = eg.FullSgd.new_with_distance_matrix(d)
    scheduler = sgd.scheduler(
        100,  # number of iterations
        0.1,  # eps: eta_min = eps * min d[i, j] ^ 2
    )

    def step(eta):
        print(eg.stress(drawing, d))
        sgd.shuffle(rng)
        sgd.apply(drawing, eta)
    scheduler.run(step)

    pos = {u: (drawing.x(i) * size, drawing.y(i) * size)
           for u, i in indices.items()}
    nx_edge_graph = nx.Graph()
    edge_pos = {}
    for e in graph.edge_indices():
        u, v = graph.edge_endpoints(e)
        segments = drawing.edge_segments(u, v)
        for i, ((x1, y1), (x2, y2)) in enumerate(segments):
            eu = f'{u}:{v}:{i}:0'
            ev = f'{u}:{v}:{i}:1'
            nx_edge_graph.add_edge(eu, ev)
            edge_pos[eu] = (x1 * size, y1 * size)
            edge_pos[ev] = (x2 * size, y2 * size)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    nx.draw_networkx_nodes(nx_graph, pos, ax=ax)
    nx.draw_networkx_edges(nx_edge_graph, edge_pos, ax=ax)
    plt.savefig('tmp/sgd_torus.png')


if __name__ == '__main__':
    main()
