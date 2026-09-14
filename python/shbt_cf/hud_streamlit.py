"""Streamlit fabrication-workbench HUD.

Run:  streamlit run python/shbt_cf/hud_streamlit.py

Mode selector in the sidebar switches between Offline Theoretical Closure
(parameter sweeps / NSGA-III / GUM PDFs) and the Physical Fabrication
Workbench (cleanroom assembly HUD). Uses simulated operator inputs driven
by sliders so the page runs on machines without live cleanroom hardware.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from shbt_cf.native import HAVE_NATIVE
from shbt_cf.workbench import (
    BellevillePreload,
    GratingAlignment,
    HipimsStress,
    TorqueWrench,
    hud_tick,
    power_net_mc,
    ring_residual_sample,
)


def _status_badge(status: str) -> str:
    return {"ok": "🟢 ok", "warn": "🟡 warn"}.get(status, f"🔴 {status}")


def fabrication_page() -> None:
    st.header("Physical Fabrication Workbench — cleanroom HUD")
    col1, col2, col3 = st.columns(3)

    torque = TorqueWrench(setpoint_nm=18.0, tolerance_nm=0.5)
    preload = BellevillePreload(de_m=0.03175, di_m=0.01626, t_m=0.00150,
                                h0_m=0.00178, e_pa=206e9, nu=0.30)
    hipims = HipimsStress(limit_gpa=2.0)

    with st.sidebar:
        torque_nm = st.slider("Torque wrench reading (N·m)", 14.0, 22.0, 18.0, 0.05)
        defl_um = st.slider("Stack deflection (µm)", 0.0, 1500.0, 800.0, 5.0)
        stress_gpa = st.slider("HiPIMS residual stress (GPa)", -3.0, 3.0, -1.4, 0.05)
        dx = st.slider("Δx alignment (µm)", -5.0, 5.0, 0.4, 0.05)
        dy = st.slider("Δy alignment (µm)", -5.0, 5.0, -0.3, 0.05)
        tilt = st.slider("θ_tilt (mrad)", -1.5, 1.5, 0.1, 0.01)

    align = GratingAlignment(dx_um=dx, dy_um=dy, theta_tilt_mrad=tilt)
    residuals = {}
    if HAVE_NATIVE:
        residuals = ring_residual_sample(frames=200)

    hud = hud_tick(torque, preload, hipims, align, torque_nm,
                   defl_um * 1e-6, stress_gpa, residuals)

    col1.metric("Torque", f"{hud.torque_nm:.2f} N·m", _status_badge(hud.torque_status))
    col2.metric("Stack preload", f"{hud.stack_force_n:,.0f} N",
                f"deflection {defl_um:.0f} µm")
    col3.metric("HiPIMS stress", f"{hud.hipims_stress_gpa:+.2f} GPa",
                _status_badge(hud.hipims_status))

    st.subheader("RCWA grating alignment")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Δx", f"{align.dx_um:+.2f} µm")
    a2.metric("Δy", f"{align.dy_um:+.2f} µm")
    a3.metric("θ_tilt", f"{align.theta_tilt_mrad:+.2f} mrad")
    a4.metric("Status", _status_badge(align.status()))

    if residuals:
        st.subheader("Telemetry χ² residuals (vs reduced-order surrogate)")
        rcols = st.columns(len(residuals))
        for c, (name, v) in zip(rcols, residuals.items()):
            c.metric(name, f"χ̄² = {v:.3f}")
    elif not HAVE_NATIVE:
        st.info("Native extension not built — telemetry residuals disabled.")


def offline_page() -> None:
    st.header("Offline Theoretical Closure")
    st.subheader("GUM Monte-Carlo PDF — P_net")
    with st.sidebar:
        p_teg = st.number_input("P_teg mean (W)", value=45.0)
        p_sup = st.number_input("P_support mean (W)", value=13.25)
        rho = st.slider("input correlation ρ", -0.9, 0.9, -0.30, 0.05)
        n_mc = st.select_slider("MC draws", [100_000, 500_000, 1_000_000], 500_000)
    if not HAVE_NATIVE:
        st.warning("Build `shbt_cf_native` to run the native MC engine.")
        return
    cov = np.array([[2.5**2, rho * 2.5 * 0.9], [rho * 2.5 * 0.9, 0.9**2]])
    pdf = power_net_mc((p_teg, p_sup), cov, n=n_mc)
    st.write(f"mean {pdf.mean:.3f} W · σ {pdf.std:.3f} W · "
             f"95% coverage [{pdf.coverage_95[0]:.3f}, {pdf.coverage_95[1]:.3f}] W")
    st.bar_chart(pdf.density)


def main() -> None:
    st.set_page_config(page_title="SHBT Workbench", layout="wide")
    st.title("SHBT zero-drift simulator — engineering workbench")
    mode = st.sidebar.radio("Mode", ["Fabrication Workbench", "Offline Theoretical Closure"])
    if mode == "Fabrication Workbench":
        fabrication_page()
    else:
        offline_page()


if __name__ == "__main__":
    main()
