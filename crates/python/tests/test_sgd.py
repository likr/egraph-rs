import math
import unittest
import networkx as nx
import egraph as eg


def draw(nx_graph):
    graph = eg.Graph()
    indices = {}
    for u in nx_graph.nodes:
        indices[u] = graph.add_node(u)
    for u, v in nx_graph.edges:
        graph.add_edge(indices[u], indices[v], (u, v))
    return graph


def check_drawing_2d(graph, drawing):
    for u in graph.node_indices():
        assert math.isfinite(drawing.x(u))
        assert math.isfinite(drawing.y(u))


def check_drawing_3d(graph, drawing):
    for u in graph.node_indices():
        assert math.isfinite(drawing.get(u, 0))
        assert math.isfinite(drawing.get(u, 1))
        assert math.isfinite(drawing.get(u, 2))


class TestSgd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._graphs = [draw(nx.les_miserables_graph())]
        cls._schedulers = [
            eg.SchedulerConstant,
            eg.SchedulerLinear,
            eg.SchedulerQuadratic,
            eg.SchedulerExponential,
            eg.SchedulerReciprocal,
        ]

    def test_full_sgd(self):
        for graph in self._graphs:
            for Scheduler in self._schedulers:
                drawing = eg.DrawingEuclidean2d.initial_placement(graph)
                rng = eg.Rng.seed_from(0)
                sgd = eg.FullSgd().build(graph, lambda _: 30)
                scheduler = Scheduler(15)

                def step(eta):
                    sgd.shuffle(rng)
                    sgd.apply(drawing, eta)

                scheduler.run(step)
                check_drawing_2d(graph, drawing)

    def test_sparse_sgd(self):
        for graph in self._graphs:
            for Scheduler in self._schedulers:
                drawing = eg.DrawingEuclidean2d.initial_placement(graph)
                rng = eg.Rng.seed_from(0)
                sgd = eg.SparseSgd().build(graph, lambda _: 30, rng)
                scheduler = Scheduler(15)

                def step(eta):
                    sgd.shuffle(rng)
                    sgd.apply(drawing, eta)

                scheduler.run(step)
                check_drawing_2d(graph, drawing)

    def test_full_sgd_3d(self):
        for graph in self._graphs:
            for Scheduler in self._schedulers:
                drawing = eg.ClassicalMds(graph, lambda _: 30).run(3)
                rng = eg.Rng.seed_from(0)
                sgd = eg.FullSgd().build(graph, lambda _: 30)
                scheduler = Scheduler(15)

                def step(eta):
                    sgd.shuffle(rng)
                    sgd.apply(drawing, eta)

                scheduler.run(step)
                check_drawing_3d(graph, drawing)

    def test_sparse_sgd_3d(self):
        for graph in self._graphs:
            for Scheduler in self._schedulers:
                drawing = eg.ClassicalMds(graph, lambda _: 30).run(3)
                rng = eg.Rng.seed_from(0)
                sgd = eg.SparseSgd().build(graph, lambda _: 30, rng)
                scheduler = Scheduler(15)

                def step(eta):
                    sgd.shuffle(rng)
                    sgd.apply(drawing, eta)

                scheduler.run(step)
                check_drawing_3d(graph, drawing)

if __name__ == "__main__":
    unittest.main()
