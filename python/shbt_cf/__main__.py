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

    p = sub.add_parser("mc", help="GUM Monte-Carlo PDF (power-net or coffin-manson)")
    p.add_argument("model", choices=["power-net", "coffin-manson"])
    p.add_argument("-n", type=int, default=200_000, help="MC draws")
    p = sub.add_parser("sweep", help="parameter sweep over a solver function")
    p.add_argument("--target", choices=["disc-force"], default="disc-force")
    p.add_argument("--s-um", type=float, nargs="+", required=True,
                   help="deflection values in µm")
    p = sub.add_parser("optimize", help="NSGA-III demo over the Belleville "
                                        "force/weight trade-off")
    p.add_argument("--generations", type=int, default=30)
    sub.add_parser("hud", help="print one fabrication-HUD tick (synthetic inputs)")
    p = sub.add_parser("hud-web", help="launch the Streamlit fabrication HUD")
    sub.add_parser("verify-zerocopy",
                   help="pointer-identity gate: NumPy view vs Rust slice")

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

    from shbt_cf.workbench import (
        BellevillePreload,
        GratingAlignment,
        HipimsStress,
        TorqueWrench,
        coffin_manson_mc,
        optimize,
        hud_tick,
        power_net_mc,
        sweep,
    )

    if args.command == "mc":
        pdf = (power_net_mc((45.0, 13.25), [[6.25, -2.025], [-2.025, 0.81]], n=args.n)
               if args.model == "power-net" else coffin_manson_mc(n=args.n))
        print(f"mean={pdf.mean:.4f} std={pdf.std:.4f} "
              f"95%=[{pdf.coverage_95[0]:.4f}, {pdf.coverage_95[1]:.4f}]")
    elif args.command == "sweep":
        preload = BellevillePreload(0.03175, 0.01626, 0.00150, 0.00178, 206e9, 0.30)
        rows = sweep(lambda s_um: preload.stack_force_n(s_um * 1e-6),
                     {"s_um": tuple(args.s_um)})
        for r in rows:
            print(f"s={r['s_um']:8.1f} µm -> F={r['value']:12.2f} N")
    elif args.command == "optimize":
        import numpy as np

        def objectives(x):
            de, di, t = x[:, 0] * 1e-3, x[:, 1] * 1e-3, x[:, 2] * 1e-3
            h0 = 1.2 * t
            s = 0.5 * h0
            delta = de / di
            ln = np.log(delta)
            k1 = (((delta - 1) / delta) ** 2
                  / (np.pi * ((delta + 1) / (delta - 1) - 2 / ln)))
            x, h = s / t, h0 / t
            force = (4 * 206e9 / (1 - 0.3**2) * t**4 / (k1 * de**2)
                     * x * ((h - x) * (h - 0.5 * x) + 1))
            mass = 7.85e3 * np.pi * (de**2 - di**2) / 4 * t
            return np.c_[-force / 1e4, mass]

        res = optimize(objectives, (np.array([20.0, 8.0, 0.5]),
                                    np.array([60.0, 30.0, 3.0])),
                       2, generations=args.generations)
        f = res.objectives[res.front]
        print(f"first front: {len(f)} designs "
              f"(F range {(-f[:, 0]).min():.3f}..{(-f[:, 0]).max():.3f} ×10⁴ N)")
    elif args.command == "hud":
        hud = hud_tick(
            TorqueWrench(18.0, 0.5),
            BellevillePreload(0.03175, 0.01626, 0.00150, 0.00178, 206e9, 0.30),
            HipimsStress(2.0),
            GratingAlignment(0.4, -0.3, 0.1),
            torque_nm=18.2, deflection_m=800e-6, stress_gpa=-1.4)
        print(f"torque      : {hud.torque_nm:.2f} N·m  [{hud.torque_status}]")
        print(f"preload     : {hud.stack_force_n:,.1f} N @ "
              f"{hud.stack_deflection_m * 1e6:.0f} µm")
        print(f"hipims      : {hud.hipims_stress_gpa:+.2f} GPa [{hud.hipims_status}]")
        print(f"alignment   : dx={hud.alignment.dx_um:+.2f}µm "
              f"dy={hud.alignment.dy_um:+.2f}µm "
              f"θ={hud.alignment.theta_tilt_mrad:+.2f}mrad "
              f"[{hud.alignment_status}]")
    elif args.command == "hud-web":
        import subprocess
        from pathlib import Path

        script = Path(__file__).with_name("hud_streamlit.py")
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", str(script)])
        except FileNotFoundError:
            print("streamlit not installed: pip3 install --user streamlit",
                  file=sys.stderr)
            return 1
    elif args.command == "verify-zerocopy":
        import importlib.util

        from pathlib import Path

        script = Path(__file__).resolve().parents[2] / "scripts" / "verify_zerocopy.py"
        spec = importlib.util.spec_from_file_location("verify_zerocopy", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
