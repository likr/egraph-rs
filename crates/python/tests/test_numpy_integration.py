"""Test numpy integration for Array1, Array2, and DrawingEuclidean2d.from_array2"""

import unittest
import numpy as np
import egraph as eg


class TestNumpyIntegration(unittest.TestCase):
    """Test numpy integration for Array1, Array2, and DrawingEuclidean2d.from_array2"""

    def test_array1_numpy_integration(self):
        """Test Array1 numpy integration"""
        # Create numpy array
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

        # Test constructor with numpy array
        arr1 = eg.Array1(data)
        self.assertEqual(len(arr1), 4)
        self.assertEqual(arr1[0], 1.0)
        self.assertEqual(arr1[3], 4.0)

        # Test from_numpy classmethod
        arr2 = eg.Array1.from_numpy(data)
        self.assertEqual(len(arr2), 4)
        self.assertEqual(arr2[1], 2.0)

        # Test to_numpy method
        numpy_result = arr1.to_numpy()
        np.testing.assert_array_equal(numpy_result, data)

        # Test empty constructor
        empty_arr = eg.Array1()
        self.assertEqual(len(empty_arr), 0)

    def test_array2_numpy_integration(self):
        """Test Array2 numpy integration"""
        # Create 2D numpy array
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)

        # Test constructor with numpy array
        arr2d = eg.Array2(data)
        self.assertEqual(arr2d.shape, (3, 2))
        self.assertEqual(arr2d.nrows, 3)
        self.assertEqual(arr2d.ncols, 2)
        self.assertEqual(arr2d[0, 0], 1.0)
        self.assertEqual(arr2d[2, 1], 6.0)

        # Test from_numpy classmethod
        arr2d_from_numpy = eg.Array2.from_numpy(data)
        self.assertEqual(arr2d_from_numpy.shape, (3, 2))
        self.assertEqual(arr2d_from_numpy[1, 0], 3.0)

        # Test to_numpy method
        numpy_result = arr2d.to_numpy()
        np.testing.assert_array_equal(numpy_result, data)

        # Test row and column access
        row = arr2d.row(0)
        self.assertEqual(len(row), 2)
        self.assertEqual(row[0], 1.0)
        self.assertEqual(row[1], 2.0)

        col = arr2d.column(1)
        self.assertEqual(len(col), 3)
        self.assertEqual(col[0], 2.0)
        self.assertEqual(col[1], 4.0)
        self.assertEqual(col[2], 6.0)

        # Test empty constructor
        empty_arr2d = eg.Array2()
        self.assertEqual(empty_arr2d.shape, (0, 0))

    def test_drawing_euclidean_2d_from_array2(self):
        """Test DrawingEuclidean2d.from_array2 method"""
        # Create a simple graph
        graph = eg.Graph()
        n1 = graph.add_node("node1")
        n2 = graph.add_node("node2")
        n3 = graph.add_node("node3")
        graph.add_edge(n1, n2, "edge1")
        graph.add_edge(n2, n3, "edge2")

        # Create coordinates array (3 nodes, 2 coordinates each)
        coordinates = np.array([
            [0.0, 0.0],  # node 0 at origin
            [1.0, 0.0],  # node 1 at (1,0)
            [0.5, 1.0]   # node 2 at (0.5,1)
        ], dtype=np.float64)

        # Create Array2 from numpy array
        coord_array = eg.Array2(coordinates)

        # Create drawing from array
        drawing = eg.DrawingEuclidean2d.from_array2(graph, coord_array)

        # Verify coordinates are set correctly
        self.assertEqual(drawing.x(0), 0.0)
        self.assertEqual(drawing.y(0), 0.0)
        self.assertEqual(drawing.x(1), 1.0)
        self.assertEqual(drawing.y(1), 0.0)
        self.assertEqual(drawing.x(2), 0.5)
        self.assertEqual(drawing.y(2), 1.0)

        # Test with wrong shape - should raise ValueError
        with self.assertRaisesRegex(ValueError, "exactly 2 columns"):
            wrong_shape = np.array(
                [[1.0, 2.0, 3.0]], dtype=np.float64)  # 3 columns
            wrong_array = eg.Array2(wrong_shape)
            eg.DrawingEuclidean2d.from_array2(graph, wrong_array)

        # Test with wrong number of rows - should raise ValueError
        with self.assertRaisesRegex(ValueError, "rows to match graph node count"):
            wrong_rows = np.array([[1.0, 2.0], [3.0, 4.0]],
                                  dtype=np.float64)  # 2 rows for 3 nodes
            wrong_array = eg.Array2(wrong_rows)
            eg.DrawingEuclidean2d.from_array2(graph, wrong_array)

    def test_array1_indexing_and_iteration(self):
        """Test Array1 indexing and iteration functionality"""
        data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
        arr = eg.Array1(data)

        # Test positive indexing
        self.assertEqual(arr[0], 10.0)
        self.assertEqual(arr[1], 20.0)

        # Test negative indexing
        self.assertEqual(arr[-1], 40.0)
        self.assertEqual(arr[-2], 30.0)

        # Test iteration
        values = list(arr)
        self.assertEqual(values, [10.0, 20.0, 30.0, 40.0])

        # Test to_list
        list_result = arr.to_list()
        self.assertEqual(list_result, [10.0, 20.0, 30.0, 40.0])

    def test_array2_indexing_and_methods(self):
        """Test Array2 indexing and method functionality"""
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        arr = eg.Array2(data)

        # Test positive indexing
        self.assertEqual(arr[0, 0], 1.0)
        self.assertEqual(arr[1, 2], 6.0)

        # Test negative indexing
        self.assertEqual(arr[-1, -1], 6.0)
        self.assertEqual(arr[-2, 0], 1.0)

        # Test to_list
        list_result = arr.to_list()
        expected = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        self.assertEqual(list_result, expected)

    def test_array1_error_handling(self):
        """Test Array1 error handling for invalid indices"""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        arr = eg.Array1(data)

        # Test out of bounds access
        with self.assertRaises(IndexError):
            _ = arr[5]

        with self.assertRaises(IndexError):
            _ = arr[-5]

        # Test out of bounds assignment
        with self.assertRaises(IndexError):
            arr[5] = 10.0

    def test_array2_error_handling(self):
        """Test Array2 error handling for invalid indices"""
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        arr = eg.Array2(data)

        # Test out of bounds access
        with self.assertRaises(IndexError):
            _ = arr[5, 0]

        with self.assertRaises(IndexError):
            _ = arr[0, 5]

        # Test out of bounds row/column access
        with self.assertRaises(IndexError):
            _ = arr.row(5)

        with self.assertRaises(IndexError):
            _ = arr.column(5)


if __name__ == "__main__":
    unittest.main()
