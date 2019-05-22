import egraph


def main():
    graph = egraph.Graph()
    for _ in range(10):
        graph.add_node()
    for i in range(10):
        for j in range(i + 1, 10):
            graph.add_edge(i, j)
    for i in graph.node_indices():
        for j in graph.neighbors(i):
            print(i, j)

    fm3 = egraph.FM3()
    fm3(graph)


if __name__ == '__main__':
    main()
