import networkx as nx
from egraph import Graph, Coordinates, StressMajorization
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
    stress_majorization = StressMajorization(graph, drawing, lambda _: 100)
    stress_majorization.run(drawing)

    pos = {u: (drawing.x(i), drawing.y(i)) for u, i in indices.items()}
    nx.draw(nx_graph, pos)
    plt.savefig('tmp/stress_majorization.png')


if __name__ == '__main__':
    main()
