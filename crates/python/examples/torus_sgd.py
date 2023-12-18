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
        1 / size, # edge length
    )
    drawing = eg.DrawingTorus.initial_placement(graph)
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

    pos = {u: (drawing.x(i) , drawing.y(i)) for u, i in indices.items()}
    nx.draw(nx_graph, pos)
    plt.savefig('tmp/torus_sgd.png')


if __name__ == '__main__':
    main()
