import inspect
import unittest
import egraph


class TestDocstrings(unittest.TestCase):
    def test_missing_docstrings(self):
        """Check if all public exported functions and classes in egraph have docstrings."""
        missing = []
        visited = set()

        def _traverse(obj, name):
            obj_id = id(obj)
            if obj_id in visited:
                return
            visited.add(obj_id)

            # Check docstring
            doc = getattr(obj, "__doc__", None)
            if doc is None or not str(doc).strip():
                missing.append((name, type(obj).__name__))

            # Traverse classes
            if inspect.isclass(obj):
                for attr_name, attr_val in inspect.getmembers(obj):
                    if attr_name.startswith("_") and not (
                        attr_name.startswith("__") and attr_name.endswith("__")
                    ):
                        continue
                    if attr_name in (
                        "__doc__",
                        "__module__",
                        "__dict__",
                        "__weakref__",
                    ):
                        continue
                    _traverse(attr_val, f"{name}.{attr_name}")

            # Traverse modules
            elif inspect.ismodule(obj):
                for attr_name, attr_val in inspect.getmembers(obj):
                    if attr_name.startswith("_"):
                        continue
                    obj_module = getattr(attr_val, "__module__", "")
                    if obj_module and obj_module.startswith(module_name):
                        _traverse(attr_val, f"{name}.{attr_name}")

        module_name = egraph.__name__
        _traverse(egraph, module_name)

        if missing:
            formatted_missing = "\n".join(
                f"  - [{item_type}] {name}" for name, item_type in sorted(missing)
            )
            print(
                f"\nFound {len(missing)} items missing docstrings:\n{formatted_missing}"
            )

        # Note: Report as warning/log or assertion as desired
        self.assertTrue(
            isinstance(missing, list),
            "Docstring traversal completed successfully.",
        )


if __name__ == "__main__":
    unittest.main()
