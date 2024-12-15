import networkx as nx
import egraph as eg
import matplotlib.pyplot as plt


def plot(nx_graph, indices, drawing, filename):
    pos = {u: (drawing.x(i), drawing.y(i)) for u, i in indices.items()}
    ax = plt.subplot()
    ax.set_aspect("equal")
    nx.draw(nx_graph, pos, ax=ax)
    plt.savefig(filename)
    plt.close()


def main():
    nx_graph = nx.les_miserables_graph()
    graph = eg.Graph()
    indices = {}
    for u in nx_graph.nodes:
        indices[u] = graph.add_node(u)
    for u, v in nx_graph.edges:
        graph.add_edge(indices[u], indices[v], (u, v))

    drawing = eg.Drawing2D.initial_placement(graph)
    d = eg.warshall_floyd(graph, lambda _: 100)
    s0 = eg.stress(drawing, d)
    stress_majorization = eg.StressMajorization.with_distance_matrix(drawing, d)
    stress_majorization.run(drawing)
    s = eg.stress(drawing, d)
    print(f"stress {s0:.2f} -> {s:.2f}")

    plot(nx_graph, indices, drawing, "tmp/stress_majorization.png")


if __name__ == "__main__":
    main()
