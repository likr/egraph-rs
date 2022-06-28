import networkx as nx
from egraph import Graph, Coordinates, KamadaKawai
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
    kamada_kawai = KamadaKawai(graph, lambda _: 30)
    kamada_kawai.eps = 1e-3
    kamada_kawai.run(drawing)

    pos = {u: (drawing.x(i), drawing.y(i)) for u, i in indices.items()}
    nx.draw(nx_graph, pos)
    plt.savefig('tmp/kamada_kawai.png')


if __name__ == '__main__':
    main()
