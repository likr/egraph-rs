Installation
============

This guide will help you install egraph and its dependencies.

Basic Installation
------------------

The easiest way to install egraph is using pip:

.. code-block:: bash

    pip install --pre "egraph==6.0.0a5"

To include optional interoperability and visualization dependencies (SciPy, NetworkX, igraph, matplotlib):

.. code-block:: bash

    pip install --pre "egraph[interop,viz]==6.0.0a5"

Requirements
------------

egraph requires:

* **Python**: 3.10 or later (``abi3-py310`` wheel)
* **NumPy**: ``>= 1.26`` (automatically installed as a required dependency)

egraph is implemented in strictly single-threaded pure CPU Rust with deterministic seeded execution (no ``rayon``, ``std::thread``, or C/Fortran FFI dependencies).

Optional Dependencies
---------------------

Optional extras defined in ``pyproject.toml``:

* ``interop``: ``scipy>=1.11``, ``networkx>=3.0``, ``igraph>=0.11``
* ``viz``: ``matplotlib>=3.7``, ``networkx>=3.0``
* ``docs``: ``sphinx>=8.0``, ``sphinx-rtd-theme>=2.0``

Development Installation
------------------------

If you want to contribute to egraph or build from source:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/likr/egraph-rs.git
    cd egraph-rs/crates/python

    # Install in development mode
    pip install "maturin>=1.7,<2.0" "numpy>=1.26"
    maturin develop --release

This requires:

* Rust toolchain (install from https://rustup.rs/)
* ``maturin>=1.7,<2.0`` (Python package for building Rust extensions)

Verifying Installation
----------------------

To verify that egraph is installed correctly, run:

.. testcode:: python

    import egraph as eg

    g = eg.CsrGraph.from_edges(3, [(0, 1), (1, 2), (2, 0)])
    assert g.num_nodes == 3 and g.num_edges == 3
    print("egraph ready:", g.num_nodes, g.num_edges)

.. testoutput:: python

    egraph ready: 3 3

Next Steps
----------

Now that you have egraph installed, proceed to:

* :doc:`quickstart` - Create your first v6 graph layout
* :doc:`overview` - Explore v6 capabilities
* :doc:`migration_v6` - Migrate from v5 / legacy alpha APIs to v6
