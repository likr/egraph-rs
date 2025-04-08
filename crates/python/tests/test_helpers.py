import math
import unittest
import networkx as nx
import numpy as np
import egraph as eg


def draw_networkx(nx_graph):
    """
    Convert a NetworkX graph to an egraph Graph

    Parameters:
        nx_graph: A NetworkX graph

    Returns:
        An egraph Graph
    """
    graph = eg.Graph()
    indices = {}
    for u in nx_graph.nodes:
        indices[u] = graph.add_node(u)
    for u, v in nx_graph.edges:
        graph.add_edge(indices[u], indices[v], (u, v))
    return graph


def create_line_graph(size=3):
    """
    Create a line graph with the specified number of nodes

    Parameters:
        size: Number of nodes

    Returns:
        A tuple (graph, nodes) where graph is an egraph Graph and nodes is a list of node indices
    """
    graph = eg.Graph()
    nodes = []
    for i in range(size):
        nodes.append(graph.add_node(i))
    for i in range(size - 1):
        graph.add_edge(nodes[i], nodes[i + 1], (i, i + 1))
    return graph, nodes


def create_cycle_graph(size=3):
    """
    Create a cycle graph with the specified number of nodes

    Parameters:
        size: Number of nodes

    Returns:
        A tuple (graph, nodes) where graph is an egraph Graph and nodes is a list of node indices
    """
    graph = eg.Graph()
    nodes = []
    for i in range(size):
        nodes.append(graph.add_node(i))
    for i in range(size):
        graph.add_edge(nodes[i], nodes[(i + 1) % size], (i, (i + 1) % size))
    return graph, nodes


def create_complete_graph(size=3):
    """
    Create a complete graph with the specified number of nodes

    Parameters:
        size: Number of nodes

    Returns:
        A tuple (graph, nodes) where graph is an egraph Graph and nodes is a list of node indices
    """
    graph = eg.Graph()
    nodes = []
    for i in range(size):
        nodes.append(graph.add_node(i))
    for i in range(size):
        for j in range(i + 1, size):
            graph.add_edge(nodes[i], nodes[j], (i, j))
    return graph, nodes


def create_star_graph(size=5):
    """
    Create a star graph with a central node connected to all other nodes

    Parameters:
        size: Number of nodes (including the central node)

    Returns:
        A tuple (graph, nodes) where graph is an egraph Graph and nodes is a list of node indices
    """
    graph = eg.Graph()
    nodes = []
    for i in range(size):
        nodes.append(graph.add_node(i))
    for i in range(1, size):
        graph.add_edge(nodes[0], nodes[i], (0, i))
    return graph, nodes


def create_grid_graph(width=3, height=3):
    """
    Create a grid graph with the specified width and height

    Parameters:
        width: Number of nodes in the horizontal direction
        height: Number of nodes in the vertical direction

    Returns:
        A tuple (graph, nodes) where graph is an egraph Graph and nodes is a 2D list of node indices
    """
    graph = eg.Graph()
    nodes = []

    # Create nodes
    for y in range(height):
        row = []
        for x in range(width):
            row.append(graph.add_node((y, x)))
        nodes.append(row)

    # Create horizontal edges
    for y in range(height):
        for x in range(width - 1):
            graph.add_edge(nodes[y][x], nodes[y][x + 1], ((y, x), (y, x + 1)))

    # Create vertical edges
    for y in range(height - 1):
        for x in range(width):
            graph.add_edge(nodes[y][x], nodes[y + 1][x], ((y, x), (y + 1, x)))

    return graph, nodes


def check_drawing_2d(graph, drawing):
    """
    Verify that all coordinates in a 2D drawing are finite

    Parameters:
        graph: An egraph Graph
        drawing: A DrawingEuclidean2d instance
    """
    for u in graph.node_indices():
        assert math.isfinite(drawing.x(u))
        assert math.isfinite(drawing.y(u))


def check_drawing_spherical(graph, drawing):
    """
    Verify that all coordinates in a spherical drawing are finite

    Parameters:
        graph: An egraph Graph
        drawing: A DrawingSpherical2d instance
    """
    for u in graph.node_indices():
        assert math.isfinite(drawing.lon(u))
        assert math.isfinite(drawing.lat(u))


def check_drawing_nd(graph, drawing, dimensions):
    """
    Verify that all coordinates in an n-dimensional drawing are finite

    Parameters:
        graph: An egraph Graph
        drawing: A DrawingEuclidean instance
        dimensions: Number of dimensions
    """
    for u in graph.node_indices():
        for d in range(dimensions):
            assert math.isfinite(drawing.get(u, d))


def record_positions_2d(drawing, graph):
    """
    Record the positions of all nodes in a 2D drawing

    Parameters:
        drawing: A DrawingEuclidean2d instance
        graph: An egraph Graph

    Returns:
        A dictionary mapping node indices to (x, y) tuples
    """
    positions = {}
    for u in graph.node_indices():
        positions[u] = (drawing.x(u), drawing.y(u))
    return positions


def record_positions_spherical(drawing, graph):
    """
    Record the positions of all nodes in a spherical drawing

    Parameters:
        drawing: A DrawingSpherical2d instance
        graph: An egraph Graph

    Returns:
        A dictionary mapping node indices to (lon, lat) tuples
    """
    positions = {}
    for u in graph.node_indices():
        positions[u] = (drawing.lon(u), drawing.lat(u))
    return positions


def record_positions_nd(drawing, graph, dimensions):
    """
    Record the positions of all nodes in an n-dimensional drawing

    Parameters:
        drawing: A DrawingEuclidean instance
        graph: An egraph Graph
        dimensions: Number of dimensions

    Returns:
        A dictionary mapping node indices to position tuples
    """
    positions = {}
    for u in graph.node_indices():
        positions[u] = tuple(drawing.get(u, d) for d in range(dimensions))
    return positions


def positions_changed_2d(drawing, graph, initial_positions):
    """
    Check if any node positions have changed in a 2D drawing

    Parameters:
        drawing: A DrawingEuclidean2d instance
        graph: An egraph Graph
        initial_positions: Dictionary mapping node indices to (x, y) tuples

    Returns:
        True if any position has changed, False otherwise
    """
    for u in graph.node_indices():
        if (drawing.x(u), drawing.y(u)) != initial_positions[u]:
            return True
    return False


def positions_changed_spherical(drawing, graph, initial_positions):
    """
    Check if any node positions have changed in a spherical drawing

    Parameters:
        drawing: A DrawingSpherical2d instance
        graph: An egraph Graph
        initial_positions: Dictionary mapping node indices to (lon, lat) tuples

    Returns:
        True if any position has changed, False otherwise
    """
    for u in graph.node_indices():
        if (drawing.lon(u), drawing.lat(u)) != initial_positions[u]:
            return True
    return False


def positions_changed_nd(drawing, graph, initial_positions, dimensions):
    """
    Check if any node positions have changed in an n-dimensional drawing

    Parameters:
        drawing: A DrawingEuclidean instance
        graph: An egraph Graph
        initial_positions: Dictionary mapping node indices to position tuples
        dimensions: Number of dimensions

    Returns:
        True if any position has changed, False otherwise
    """
    for u in graph.node_indices():
        current_pos = tuple(drawing.get(u, d) for d in range(dimensions))
        if current_pos != initial_positions[u]:
            return True
    return False


def calculate_energy(graph, drawing, distance_func=None):
    """
    Calculate the energy of a drawing according to the Kamada-Kawai model

    Parameters:
        graph: An egraph Graph
        drawing: A DrawingEuclidean2d instance
        distance_func: A function that takes an edge weight and returns a distance

    Returns:
        The energy value
    """
    if distance_func is None:
        def distance_func(e): return 1.0

    energy = 0.0
    for i in graph.node_indices():
        for j in graph.node_indices():
            if i < j:  # Only consider each pair once
                # Calculate Euclidean distance in the drawing
                dx = drawing.x(i) - drawing.x(j)
                dy = drawing.y(i) - drawing.y(j)
                actual_distance = math.sqrt(dx * dx + dy * dy)

                # Calculate ideal distance (graph-theoretic)
                # For simplicity, we'll use 1.0 for all edges
                ideal_distance = 1.0

                # Calculate spring constant (typically 1/d^2)
                spring_constant = 1.0 / (ideal_distance * ideal_distance)

                # Add to energy
                diff = actual_distance - ideal_distance
                energy += spring_constant * diff * diff

    return energy


def verify_node_positions(drawing, expected_positions, tolerance=1e-8):
    """
    Verify that node positions match expected positions within tolerance

    Parameters:
        drawing: A DrawingEuclidean2d instance
        expected_positions: Dictionary mapping node indices to position dictionaries
        tolerance: Tolerance for floating-point comparison
    """
    for node_index, position in expected_positions.items():
        if hasattr(drawing, 'x') and hasattr(drawing, 'y'):
            # 2D Euclidean, Torus, or Hyperbolic drawing
            if 'x' in position:
                assert abs(drawing.x(node_index) - position['x']) < tolerance, \
                    f"Node {node_index} X coordinate should match expected value"
            if 'y' in position:
                assert abs(drawing.y(node_index) - position['y']) < tolerance, \
                    f"Node {node_index} Y coordinate should match expected value"
        elif hasattr(drawing, 'lon') and hasattr(drawing, 'lat'):
            # Spherical drawing
            if 'lon' in position:
                assert abs(drawing.lon(node_index) - position['lon']) < tolerance, \
                    f"Node {node_index} longitude should match expected value"
            if 'lat' in position:
                assert abs(drawing.lat(node_index) - position['lat']) < tolerance, \
                    f"Node {node_index} latitude should match expected value"
        elif hasattr(drawing, 'get'):
            # N-dimensional Euclidean drawing
            for d, value in position.items():
                if isinstance(d, int):
                    assert abs(drawing.get(node_index, d) - value) < tolerance, \
                        f"Node {node_index} coordinate at dimension {d} should match expected value"


def verify_connected_nodes_closer(graph, drawing):
    """
    Verify that connected nodes are positioned closer to each other than the average distance

    Parameters:
        graph: An egraph Graph
        drawing: A DrawingEuclidean2d instance
    """
    connected_pairs_count = 0
    connected_pairs_distance = 0
    all_pairs_count = 0
    all_pairs_distance = 0

    # Calculate average distance between connected nodes
    for e in graph.edge_indices():
        # Get the endpoints of the edge
        u, v = graph.edge_endpoints(e)

        dx = drawing.x(u) - drawing.x(v)
        dy = drawing.y(u) - drawing.y(v)
        distance = math.sqrt(dx * dx + dy * dy)
        connected_pairs_distance += distance
        connected_pairs_count += 1

    # Calculate average distance between all node pairs
    node_indices = list(graph.node_indices())
    for i in range(len(node_indices)):
        for j in range(i + 1, len(node_indices)):
            u = node_indices[i]
            v = node_indices[j]
            dx = drawing.x(u) - drawing.x(v)
            dy = drawing.y(u) - drawing.y(v)
            distance = math.sqrt(dx * dx + dy * dy)
            all_pairs_distance += distance
            all_pairs_count += 1

    avg_connected_distance = connected_pairs_distance / connected_pairs_count
    avg_all_distance = all_pairs_distance / all_pairs_count

    assert avg_connected_distance < avg_all_distance, \
        "Connected nodes should be positioned closer to each other than the average distance between all nodes"


def verify_layout_quality(graph, drawing, options=None):
    """
    Verify layout quality using various metrics

    Parameters:
        graph: An egraph Graph
        drawing: A drawing instance
        options: Dictionary of options for verification

    Returns:
        Dictionary of quality metrics
    """
    if options is None:
        options = {}

    # Verify that all coordinates are finite numbers
    if hasattr(drawing, 'x') and hasattr(drawing, 'y'):
        # 2D Euclidean, Torus, or Hyperbolic drawing
        check_drawing_2d(graph, drawing)
    elif hasattr(drawing, 'lon') and hasattr(drawing, 'lat'):
        # Spherical drawing
        check_drawing_spherical(graph, drawing)
    elif hasattr(drawing, 'get'):
        # N-dimensional Euclidean drawing
        dimensions = options.get('dimensions', 3)
        check_drawing_nd(graph, drawing, dimensions)

    # Verify that connected nodes are positioned closer together
    if hasattr(drawing, 'x') and hasattr(drawing, 'y') and options.get('verify_connected_nodes_closer', True):
        verify_connected_nodes_closer(graph, drawing)

    # Calculate stress if requested
    if options.get('calculate_stress', False):
        stress = eg.stress(graph, drawing)
        assert math.isfinite(stress), "Stress should be a finite number"
        return {'stress': stress}

    return {}


def verify_layout_improvement(graph, before_drawing, after_drawing, metric='stress'):
    """
    Verify that layout quality has improved

    Parameters:
        graph: An egraph Graph
        before_drawing: Drawing before layout application
        after_drawing: Drawing after layout application
        metric: Metric to use for comparison: 'stress', 'crossing_number', 'neighborhood_preservation'

    Returns:
        Dictionary containing before and after metric values
    """
    before_value = None
    after_value = None

    if metric.lower() == 'stress':
        before_value = eg.stress(graph, before_drawing)
        after_value = eg.stress(graph, after_drawing)
        assert after_value <= before_value, \
            f"Stress should be reduced or equal after layout (before: {before_value}, after: {after_value})"
    elif metric.lower() == 'crossing_number':
        before_value = eg.crossing_number(graph, before_drawing)
        after_value = eg.crossing_number(graph, after_drawing)
        # We don't assert improvement here as some layouts might increase crossings
        # while optimizing for other metrics
    elif metric.lower() == 'neighborhood_preservation':
        before_value = eg.neighborhood_preservation(graph, before_drawing)
        after_value = eg.neighborhood_preservation(graph, after_drawing)
        assert after_value >= before_value, \
            f"Neighborhood preservation should be improved or equal after layout (before: {before_value}, after: {after_value})"
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return {'before_value': before_value, 'after_value': after_value}
