"""Workspace model and `cargo` orchestration."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

# Strict IEEE 754 flags from .cargo/config.toml, re-asserted on every cargo
# invocation so a caller's environment cannot silently re-enable fast-math.
STRICT_FP_RUSTFLAGS: tuple[str, ...] = (
    "-C",
    "llvm-args=-enable-no-nans-fp-math=false",
    "-C",
    "llvm-args=-enable-no-signed-zeros-fp-math=false",
)


class WorkspaceError(RuntimeError):
    """Raised when the Rust workspace cannot be located or a cargo command fails."""


@dataclass(frozen=True)
class Crate:
    """A workspace member and its workspace-internal dependencies."""

    name: str
    workspace_deps: frozenset[str] = field(default_factory=frozenset)

    @property
    def path(self) -> str:
        return f"crates/{self.name}"


CRATES: dict[str, Crate] = {
    c.name: c
    for c in (
        Crate("shbt-cf"),
        Crate("shbt-core-math"),
        Crate("shbt-dielectric-floquet", frozenset({"shbt-core-math"})),
        Crate("shbt-rcwa-optics", frozenset({"shbt-core-math"})),
        Crate("shbt-fea-structural", frozenset({"shbt-core-math"})),
        Crate("shbt-metrology-gum", frozenset({"shbt-core-math"})),
        Crate(
            "shbt-fabrication-hil",
            frozenset(
                {
                    "shbt-core-math",
                    "shbt-dielectric-floquet",
                    "shbt-rcwa-optics",
                    "shbt-fea-structural",
                    "shbt-metrology-gum",
                }
            ),
        ),
        Crate("shbt-rcwa"),
        Crate("shbt-contact"),
        Crate("shbt-py-bindings", frozenset({"shbt-rcwa"})),
    )
}


def find_workspace_root(start: Path | None = None) -> Path:
    """Walk upward from `start` (default: this file) to the directory holding
    the workspace `Cargo.toml`."""
    here = (start or Path(__file__)).resolve()
    for candidate in (here, *here.parents):
        manifest = candidate / "Cargo.toml"
        if manifest.is_file() and "[workspace]" in manifest.read_text(encoding="utf-8"):
            return candidate
    raise WorkspaceError(f"no Cargo workspace found above {here}")


def build_order(crates: dict[str, Crate] = CRATES) -> list[str]:
    """Topologically sort crates (dependencies first). Raises on a cycle."""
    remaining = {name: set(c.workspace_deps) for name, c in crates.items()}
    order: list[str] = []
    while remaining:
        ready = sorted(name for name, deps in remaining.items() if not deps)
        if not ready:
            raise WorkspaceError(f"circular dependency among {sorted(remaining)}")
        order.extend(ready)
        for name in ready:
            del remaining[name]
        for deps in remaining.values():
            deps.difference_update(ready)
    return order


@dataclass
class Workspace:
    """Handle on the Rust workspace used to drive cargo deterministically."""

    root: Path = field(default_factory=find_workspace_root)
    cargo: str = "cargo"

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        extra = " ".join(STRICT_FP_RUSTFLAGS)
        existing = env.get("CARGO_BUILD_RUSTFLAGS", "")
        env["CARGO_BUILD_RUSTFLAGS"] = (
            f"{existing} {extra}".strip() if existing else extra
        )
        return env

    def run(
        self, *args: str, capture: bool = False
    ) -> subprocess.CompletedProcess[str]:
        """Run `cargo <args>` at the workspace root with strict-FP flags asserted."""
        cmd = [self.cargo, *args]
        try:
            return subprocess.run(
                cmd,
                cwd=self.root,
                env=self._env(),
                check=True,
                text=True,
                capture_output=capture,
            )
        except FileNotFoundError as exc:
            raise WorkspaceError(f"cargo executable not found: {self.cargo}") from exc
        except subprocess.CalledProcessError as exc:
            raise WorkspaceError(
                f"{' '.join(cmd)} failed with exit code {exc.returncode}"
            ) from exc

    def check(self, crates: Sequence[str] = ()) -> None:
        self.run("check", "--workspace", *_package_args(crates))

    def build(self, crates: Sequence[str] = (), release: bool = True) -> None:
        self.run(
            "build",
            "--workspace",
            *(("--release",) if release else ()),
            *_package_args(crates),
        )

    def test(self, crates: Sequence[str] = ()) -> None:
        self.run("test", "--workspace", *_package_args(crates))

    def clippy(self) -> None:
        self.run("clippy", "--workspace", "--all-targets", "--", "-D", "warnings")

    def metadata(self) -> dict:
        proc = self.run("metadata", "--format-version", "1", "--no-deps", capture=True)
        return json.loads(proc.stdout)

    def actual_graph(self) -> dict[str, frozenset[str]]:
        """Workspace-internal dependency graph as reported by cargo."""
        meta = self.metadata()
        names = {pkg["name"] for pkg in meta["packages"]}
        return {
            pkg["name"]: frozenset(
                d["name"] for d in pkg["dependencies"] if d["name"] in names
            )
            for pkg in meta["packages"]
        }

    def verify_topology(self) -> None:
        """Assert the on-disk workspace matches `CRATES` exactly."""
        actual = self.actual_graph()
        expected = {name: c.workspace_deps for name, c in CRATES.items()}
        if actual != expected:
            diff = {
                k: (sorted(expected.get(k, ())), sorted(actual.get(k, ())))
                for k in expected | actual
                if expected.get(k) != actual.get(k)
            }
            raise WorkspaceError(
                f"workspace topology mismatch (expected, actual): {diff}"
            )


def _package_args(crates: Sequence[str]) -> list[str]:
    unknown = [c for c in crates if c not in CRATES]
    if unknown:
        raise WorkspaceError(f"unknown workspace crate(s): {unknown}")
    args: list[str] = []
    for crate in crates:
        args += ["-p", crate]
    return args
