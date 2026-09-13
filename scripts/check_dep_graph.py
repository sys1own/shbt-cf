#!/usr/bin/env python3
"""Validate the workspace crate dependency graph against the Stage 1 topology.

Checks, using `cargo metadata`:
  1. exactly the six expected crates are workspace members;
  2. the workspace-internal dependency graph is acyclic (Kahn's algorithm);
  3. each crate's workspace dependencies match the required topology exactly
     (unidirectional: core-math <- domain crates <- fabrication-hil).

Exit status is non-zero on any violation so the script can gate CI.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import deque
from pathlib import Path

EXPECTED_TOPOLOGY: dict[str, frozenset[str]] = {
    "shbt-core-math": frozenset(),
    "shbt-dielectric-floquet": frozenset({"shbt-core-math"}),
    "shbt-rcwa-optics": frozenset({"shbt-core-math"}),
    "shbt-fea-structural": frozenset({"shbt-core-math"}),
    "shbt-metrology-gum": frozenset({"shbt-core-math"}),
    "shbt-fabrication-hil": frozenset(
        {
            "shbt-core-math",
            "shbt-dielectric-floquet",
            "shbt-rcwa-optics",
            "shbt-fea-structural",
            "shbt-metrology-gum",
        }
    ),
}


def load_metadata(manifest_dir: Path) -> dict:
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=manifest_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(out)


def workspace_graph(metadata: dict) -> dict[str, set[str]]:
    """Map each workspace crate to the set of workspace crates it depends on."""
    members = {pkg["id"]: pkg for pkg in metadata["packages"]}
    member_names = {pkg["name"] for pkg in members.values()}
    graph: dict[str, set[str]] = {}
    for pkg in members.values():
        deps = {
            dep["name"]
            for dep in pkg["dependencies"]
            if dep["name"] in member_names and dep["kind"] in (None, "normal", "build")
        }
        graph[pkg["name"]] = deps
    return graph


def find_cycle(graph: dict[str, set[str]]) -> list[str]:
    """Return the crates participating in a cycle, or [] if the graph is a DAG."""
    indegree = {node: 0 for node in graph}
    for deps in graph.values():
        for dep in deps:
            indegree[dep] += 1
    queue = deque(node for node, deg in indegree.items() if deg == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for dep in graph[node]:
            indegree[dep] -= 1
            if indegree[dep] == 0:
                queue.append(dep)
    return sorted(node for node, deg in indegree.items() if deg > 0) if visited != len(graph) else []


def check(graph: dict[str, set[str]]) -> list[str]:
    errors: list[str] = []

    expected_names = set(EXPECTED_TOPOLOGY)
    actual_names = set(graph)
    if missing := expected_names - actual_names:
        errors.append(f"missing workspace crates: {sorted(missing)}")
    if extra := actual_names - expected_names:
        errors.append(f"unexpected workspace crates: {sorted(extra)}")

    if cycle := find_cycle(graph):
        errors.append(f"circular dependency among: {cycle}")

    for name, expected in EXPECTED_TOPOLOGY.items():
        actual = graph.get(name)
        if actual is None or actual == expected:
            continue
        if forbidden := actual - expected:
            errors.append(f"{name} must not depend on {sorted(forbidden)}")
        if absent := expected - actual:
            errors.append(f"{name} must depend on {sorted(absent)}")

    return errors


def main(argv: list[str]) -> int:
    manifest_dir = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent.parent
    graph = workspace_graph(load_metadata(manifest_dir))

    for name in sorted(graph):
        deps = ", ".join(sorted(graph[name])) or "(none)"
        print(f"{name:<26} -> {deps}")

    errors = check(graph)
    if errors:
        print("\nDEPENDENCY GRAPH VIOLATIONS:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print("\nOK: workspace dependency graph is acyclic and matches the required topology.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
