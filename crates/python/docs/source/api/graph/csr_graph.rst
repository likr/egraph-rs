CsrGraph
========

Compact ``u32``-indexed Compressed Sparse Row (CSR) graph storing topology and optional numeric edge weights (``"float64"`` or ``"float32"`` via ``weight_dtype``) without host object payloads, with helpers ``__len__``, ``edges()``, ``has_edge()``, ``edge_weight()``, and ``is_connected()``.

.. autoclass:: egraph.CsrGraph
   :members:
   :undoc-members:
   :show-inheritance:
