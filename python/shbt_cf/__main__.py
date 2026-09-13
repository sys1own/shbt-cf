"""Command-line entry point: `python -m shbt_cf <command>`."""

from __future__ import annotations

import argparse
import sys

from shbt_cf.workspace import CRATES, Workspace, WorkspaceError, build_order


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="shbt-cf", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("topology", help="print crate dependency topology and build order")
    sub.add_parser("verify", help="verify on-disk workspace matches the required topology")
    for name in ("check", "build", "test"):
        p = sub.add_parser(name, help=f"run cargo {name} across the workspace")
        p.add_argument("crates", nargs="*", help="restrict to these crates")
    sub.add_parser("clippy", help="run cargo clippy with -D warnings")

    args = parser.parse_args(argv)

    try:
        if args.command == "topology":
            for name in build_order():
                deps = ", ".join(sorted(CRATES[name].workspace_deps)) or "(none)"
                print(f"{name:<26} -> {deps}")
            return 0
        ws = Workspace()
        if args.command == "verify":
            ws.verify_topology()
            print("OK: workspace topology matches specification")
        elif args.command == "check":
            ws.check(args.crates)
        elif args.command == "build":
            ws.build(args.crates)
        elif args.command == "test":
            ws.test(args.crates)
        elif args.command == "clippy":
            ws.clippy()
    except WorkspaceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
