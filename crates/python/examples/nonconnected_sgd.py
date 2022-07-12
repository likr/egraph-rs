import networkx as nx
from egraph import Graph, Coordinates, Rng, SparseSgd
import matplotlib.pyplot as plt


def main():
    nx_graph = nx.watts_strogatz_graph(50, 3, 0.3, seed=0)
    components = nx.connected_components(nx_graph)

    graph = Graph()
    indices = {}
    for u in nx_graph.nodes:
        indices[u] = graph.add_node(u)
    for u, v in nx_graph.edges:
        graph.add_edge(indices[u], indices[v], (u, v))
    c = graph.add_node(u)
    for nodes in components:
        u = list(nodes)[0]
        graph.add_edge(c, indices[u], None)

    drawing = Coordinates.initial_placement(graph)
    rng = Rng.seed_from(0)  # random seed
    sgd = SparseSgd(
        graph,
        lambda _: 30,  # edge length
        50,  # number of pivots
        rng,
    )
    scheduler = sgd.scheduler(
        100,  # number of iterations
        0.1,  # eps: eta_min = eps * min d[i, j] ^ 2
    )

    def step(eta):
        sgd.shuffle(rng)
        sgd.apply(drawing, eta)
    scheduler.run(step)

    pos = {u: (drawing.x(i), drawing.y(i)) for u, i in indices.items()}
    nx.draw(nx_graph, pos)
    plt.savefig('tmp/nonconnected_sgd.png')


if __name__ == '__main__':
    main()
