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

    drawing = eg.DrawingEuclidean2d.initial_placement(graph)
    rng = eg.Rng.seed_from(0)
    sgd = eg.FullSgd(graph, lambda _: 100)
    scheduler = sgd.scheduler(100, 0.1)
    overwrap_removal = eg.OverwrapRemoval(graph, lambda _: 25)

    def step(eta):
        sgd.shuffle(rng)
        sgd.apply(drawing, eta)
        overwrap_removal.apply(drawing)
    scheduler.run(step)
    drawing.centralize()

    pos = {u: (drawing.x(i), drawing.y(i)) for u, i in indices.items()}
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-400, 400)
    ax.set_ylim(-400, 400)
    nx.draw(nx_graph, pos, ax=ax, node_size=(72 / 100 * 20) ** 2)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig('tmp/overwrap_removal.png')


if __name__ == '__main__':
    main()
