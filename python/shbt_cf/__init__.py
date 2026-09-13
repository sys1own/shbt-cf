"""Python orchestration layer for the SHBT zero-drift simulator workspace.

Python owns parameter sweeps, optimisation loops and operator interfaces; all
numerics run in the Rust crates. This package never enters high-frequency
iteration loops itself — it drives `cargo` and, from Stage 5 onward, the PyO3
native modules exported by `shbt-fabrication-hil`.
"""

from shbt_cf.workspace import (
    CRATES,
    Crate,
    Workspace,
    WorkspaceError,
    find_workspace_root,
)

__all__ = ["CRATES", "Crate", "Workspace", "WorkspaceError", "find_workspace_root"]
__version__ = "0.1.0"
