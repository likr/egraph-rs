Exceptions (``EgraphError`` Hierarchy)
======================================

Structured exceptions raised across ``egraph`` bindings, rooted at :class:`egraph.EgraphError` (a subclass of ``ValueError``). :class:`egraph.IndexOutOfBoundsError` inherits from both ``EgraphError`` and ``IndexError``.

.. autoexception:: egraph.EgraphError
   :members:
   :show-inheritance:

.. autoexception:: egraph.InvalidInputError
   :members:
   :show-inheritance:

.. autoexception:: egraph.DimensionMismatchError
   :members:
   :show-inheritance:

.. autoexception:: egraph.IndexOutOfBoundsError
   :members:
   :show-inheritance:

.. autoexception:: egraph.InvalidCsrError
   :members:
   :show-inheritance:

.. autoexception:: egraph.NonFiniteValueError
   :members:
   :show-inheritance:

.. autoexception:: egraph.UnsupportedOperationError
   :members:
   :show-inheritance:

.. autoexception:: egraph.NumericalBreakdownError
   :members:
   :show-inheritance:

.. autoexception:: egraph.BackendUnavailableError
   :members:
   :show-inheritance:

.. autoexception:: egraph.DeviceLimitExceededError
   :members:
   :show-inheritance:
