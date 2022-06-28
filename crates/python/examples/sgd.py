import networkx as nx
from egraph import Graph, Coordinates, Rng, SparseSgd
import matplotlib.pyplot as plt


def main():
    nx_graph = nx.les_miserables_graph()
    graph = Graph()
    indices = {}
    for u in nx_graph.nodes:
        indices[u] = graph.add_node(u)
    for u, v in nx_graph.edges:
        graph.add_edge(indices[u], indices[v], (u, v))

    drawing = Coordinates.initial_placement(graph)
    rng = Rng.seed_from(0)
    sgd = SparseSgd(graph, lambda _: 30, 50, rng)
    scheduler = sgd.scheduler(100, 0.1)

    def step(eta):
        sgd.shuffle(rng)
        sgd.apply(drawing, eta)
    scheduler.run(step)

    pos = {u: (drawing.x(i), drawing.y(i)) for u, i in indices.items()}
    nx.draw(nx_graph, pos)
    plt.savefig('tmp/sgd.png')


if __name__ == '__main__':
    main()
