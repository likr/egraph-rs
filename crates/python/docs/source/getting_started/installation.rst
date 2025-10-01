Installation
============

This guide will help you install egraph and its dependencies.

Basic Installation
------------------

The easiest way to install egraph is using pip:

.. code-block:: bash

    pip install egraph

This will install the latest stable version of egraph from PyPI.

Requirements
------------

egraph requires:

* Python 3.6 or later
* NumPy (automatically installed as a dependency)

Optional Dependencies
---------------------

For visualization and working with examples, you may want to install:

.. code-block:: bash

    pip install matplotlib networkx

* **matplotlib**: For creating visualizations of graph layouts
* **networkx**: For creating and manipulating graphs, and for using built-in graph datasets

Development Installation
------------------------

If you want to contribute to egraph or use the latest development version, you can install from source:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/likr/egraph-rs.git
    cd egraph-rs/crates/python

    # Install in development mode
    pip install maturin
    maturin develop

This requires:

* Rust toolchain (install from https://rustup.rs/)
* maturin (Python package for building Rust extensions)

Verifying Installation
----------------------

To verify that egraph is installed correctly, run:

.. code-block:: python

    import egraph as eg
    print(eg.__version__)

You should see the version number printed without any errors.

Next Steps
----------

Now that you have egraph installed, proceed to the :doc:`quickstart` guide to learn how to use it.
