import math
import networkx as nx
import egraph as eg
import matplotlib.pyplot as plt


class Scheduler:
    def __init__(self, eta_max, eta_min, t_max):
        self.a = eta_max
        self.b = math.log(eta_min / eta_max) / (t_max - 1)

    def __call__(self, t):
        return self.a * math.exp(self.b * t)


def optimize(sgd, drawing, rng, etas, size):
    for eta in etas:
        sgd.shuffle(rng)
        sgd.apply(drawing, eta / size ** 2)


def main():
    nx_graph = nx.complete_graph(5)
    graph = eg.Graph()
    indices = {}
    for u in nx_graph.nodes:
        indices[u] = graph.add_node(u)
    for u, v in nx_graph.edges:
        graph.add_edge(indices[u], indices[v], (u, v))

    n = graph.node_count()
    diameter = nx.diameter(nx_graph)
    rng = eg.Rng.seed_from(0)
    gss_iterations = 20
    sgd_iterations = 5
    eps = 0.1

    distance = eg.all_sources_bfs(graph, 1)
    t_max = gss_iterations * sgd_iterations
    w_min = 1 / min(distance.get(i, j)
                    for i in range(n) for j in range(n)
                    if distance.get(i, j) != 0) ** 2
    w_max = 1 / max(distance.get(i, j)
                    for i in range(n) for j in range(n)
                    if distance.get(i, j) != 0) ** 2
    scheduler = Scheduler(w_min, eps / w_max, t_max)
    eta = [scheduler(t) for t in range(t_max)]

    low = 0
    high = 5
    lr_diff = high - low
    x = (3 - math.sqrt(5)) / 2 * lr_diff

    m1 = x
    low_drawing = eg.DrawingTorus2d.initial_placement(graph)
    low_distance = eg.DistanceMatrix(graph)
    for i in range(n):
        for j in range(n):
            low_distance.set(i, j, distance.get(i, j) / (diameter * m1))
    low_sgd = eg.FullSgd.new_with_distance_matrix(low_distance)
    optimize(low_sgd, low_drawing, rng,
             eta[:sgd_iterations], diameter * m1)

    m2 = high - x
    high_drawing = eg.DrawingTorus2d.initial_placement(graph)
    high_distance = eg.DistanceMatrix(graph)
    for i in range(n):
        for j in range(n):
            high_distance.set(i, j, distance.get(i, j) / (diameter * m2))
    high_sgd = eg.FullSgd.new_with_distance_matrix(high_distance)
    optimize(high_sgd, high_drawing, rng,
             eta[:sgd_iterations], diameter * m2)

    for i in range(1, gss_iterations):
        print(m1, m2)
        if eg.stress(low_drawing, low_distance) > eg.stress(high_drawing, high_distance):
            low = m1
            m1 = m2
            m2 = high - (lr_diff - 2 * x)

            # low <- high
            for i in range(n):
                low_drawing.set_x(i, high_drawing.x(i))
                low_drawing.set_y(i, high_drawing.y(i))
                for j in range(n):
                    low_distance.set(i, j, high_distance.get(i, j))
            low_sgd = high_sgd
            optimize(low_sgd, low_drawing, rng,
                     eta[i * sgd_iterations:(i + 1) * sgd_iterations], diameter * m2)

            # high <- high - (lr_diff - 2 * x)
            for i in range(n):
                for j in range(n):
                    high_distance.set(i, j, high_distance.get(i, j) * m1 / m2)
            high_sgd = eg.FullSgd.new_with_distance_matrix(high_distance)
            optimize(high_sgd, high_drawing, rng,
                     eta[i * sgd_iterations:(i + 1) * sgd_iterations], diameter * m2)
        else:
            high = m2
            m2 = m1
            m1 = low + lr_diff - 2 * x

            # high <- low
            for i in range(n):
                high_drawing.set_x(i, low_drawing.x(i))
                high_drawing.set_y(i, low_drawing.y(i))
                for j in range(n):
                    high_distance.set(i, j, low_distance.get(i, j))
            high_sgd = low_sgd
            optimize(low_sgd, low_drawing, rng,
                     eta[i * sgd_iterations:(i + 1) * sgd_iterations], diameter * m2)

            # low <- low + (lr_diff - 2 * x)
            for i in range(n):
                for j in range(n):
                    low_distance.set(i, j, low_distance.get(i, j) * m2 / m1)
            low_sgd = eg.FullSgd.new_with_distance_matrix(low_distance)
            optimize(high_sgd, high_drawing, rng,
                     eta[i * sgd_iterations:(i + 1) * sgd_iterations], diameter * m2)

        lr_diff = high - low
        x = (3 - math.sqrt(5)) / 2 * lr_diff

    if eg.stress(low_drawing, low_distance) > eg.stress(high_drawing, high_distance):
        drawing = high_drawing
        size = m2
    else:
        drawing = low_drawing
        size = m1

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
    plt.savefig('tmp/sgd_torus_cell_optimization.png')


if __name__ == '__main__':
    main()
