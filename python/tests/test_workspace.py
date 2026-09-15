import shutil
import unittest

from shbt_cf.workspace import (
    CRATES,
    Crate,
    Workspace,
    WorkspaceError,
    build_order,
    find_workspace_root,
)


class TopologyTests(unittest.TestCase):
    def test_seven_crates(self):
        self.assertEqual(len(CRATES), 7)

    def test_root_and_core_math_are_independent(self):
        self.assertEqual(CRATES["shbt-cf"].workspace_deps, frozenset())
        self.assertEqual(CRATES["shbt-core-math"].workspace_deps, frozenset())

    def test_build_order_is_dependency_first(self):
        order = build_order()
        pos = {name: i for i, name in enumerate(order)}
        for crate in CRATES.values():
            for dep in crate.workspace_deps:
                self.assertLess(pos[dep], pos[crate.name])
        self.assertEqual(order[0], "shbt-cf")
        self.assertEqual(order[-1], "shbt-fabrication-hil")

    def test_cycle_detected(self):
        cyclic = {
            "a": Crate("a", frozenset({"b"})),
            "b": Crate("b", frozenset({"a"})),
        }
        with self.assertRaises(WorkspaceError):
            build_order(cyclic)


@unittest.skipIf(shutil.which("cargo") is None, "cargo not installed")
class CargoIntegrationTests(unittest.TestCase):
    def test_on_disk_workspace_matches_spec(self):
        Workspace(root=find_workspace_root()).verify_topology()


if __name__ == "__main__":
    unittest.main()
