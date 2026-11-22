PRISM Overlap Removal (``PrismOverlapRemoval``)
===============================================

This example demonstrates $O(N \log N)$ Delaunay PRISM overlap removal using :class:`egraph.PrismOverlapRemoval` and distortion evaluation via :func:`egraph.prism_distortion`.

Removing Circular & Rectangular Overlaps
----------------------------------------

.. testcode:: python

    import numpy as np
    import egraph as eg

    # Create intentionally overlapping initial coordinates
    coords = np.array(
        [
            [0.0, 0.0],
            [0.05, 0.02],
            [0.02, 0.06],
            [0.08, 0.04],
        ],
        dtype=np.float64,
    )
    original = eg.Layout2d.from_coords(coords)
    layout = eg.Layout2d.from_coords(coords)

    radii = np.full(4, 0.25, dtype=np.float64)
    prism = eg.PrismOverlapRemoval.from_radii(
        radii,
        max_iterations=50,
        padding=0.05,
        seed=42,
    )
    prism.run(layout)

    sigma_dist, sigma_disp = eg.prism_distortion(original, layout)
    assert layout.is_all_finite()
    assert prism.last_overlap_count == 0
    print("Remaining overlaps:", prism.last_overlap_count)

.. testoutput:: python

    Remaining overlaps: 0

Rectangular Bounding Boxes
--------------------------

You can also specify per-node half-widths and half-heights for labels or non-square nodes:

.. testcode:: python

    import numpy as np
    import egraph as eg

    layout = eg.Layout2d.from_coords([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
    half_widths = np.array([0.4, 0.4, 0.4], dtype=np.float64)
    half_heights = np.array([0.2, 0.2, 0.2], dtype=np.float64)

    prism = eg.PrismOverlapRemoval(half_widths, half_heights, padding=0.02, seed=42)
    prism.run(layout)
    assert prism.last_overlap_count == 0
    print("Box overlaps:", prism.last_overlap_count)

.. testoutput:: python

    Box overlaps: 0
