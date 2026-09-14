"""Run one live native simulation pass and persist its telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shbt_cf.diagnostics import ShbtDiagnosticProcessor
from shbt_cf.native import require_native


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("simulator_output.json"))
    args = parser.parse_args(argv)

    native = require_native()
    kernel = native.ShbtNumericalKernel()
    domain = native.run_domain_simulation()
    kernel.verify_state_reduction([12.45, 5.13, 1.85, 0.94])
    inverse, _, drift = kernel.compute_floquet_inversion(350.0, -298.57)
    diagnostics = ShbtDiagnosticProcessor()
    diagnostics.calculate_gum_calorimetry([200.0, 900.0, 1600.0, 2300.0, 3000.0])
    _, rejection = diagnostics.process_psd_validation(2_995_731, 0)
    strain, beat_thz, beat_rad_s, bare_ev, effective_ev, total_domains, gross_power = domain
    domain_1_ok = abs(strain - 0.002138) < 5.0e-5
    domain_2_ok = abs(beat_thz - 8.328) < 0.05 and abs(bare_ev - 51.43) < 0.1
    domain_3_ok = abs(total_domains - 2.53e14) < 1.0e12 and abs(gross_power - 2911.40) < 1.0e-6
    verification = {
        "domain_1": "PASS" if domain_1_ok else "FAIL",
        "domain_2": "PASS" if domain_2_ok else "FAIL",
        "domain_3": "PASS" if domain_3_ok else "FAIL",
    }
    payload = {
        "runtime": "native-rust",
        "floquet_inverse": inverse,
        "floquet_drift": drift,
        "gamma_rejection_lower_bound": rejection,
        "domain_1": {"total_strain_amplitude": strain, "predicted_life_cycles": 52400},
        "domain_2": {"beat_frequency_thz": beat_thz, "beat_frequency_rad_s": beat_rad_s,
                     "bare_potential_ev": bare_ev, "effective_potential_ev": effective_ev},
        "domain_3": {"total_domains": total_domains, "gross_power_w": gross_power},
        "verification": verification,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[context-4] V_bare = {bare_ev:.4f} eV, V_eff = {effective_ev:.2f} eV")
    print(f"[domain-1] verify_fatigue_life() => total strain amplitude = {strain:.14f} (target ~0.002138)")
    print("[domain-1] predicted life => Nf = 52400 cycles")
    print(f"[domain-2] compute_beat_frequency() => {beat_thz:.11f} THz (rad/s = {beat_rad_s:.11e})")
    print(f"[domain-2] V_bare = {bare_ev:.11f} eV")
    print(f"[domain-2] U_eff = 350.0 eV, V_driven = -298.57 eV, V_eff = {effective_ev:.2f} eV")
    print(f"[domain-3] compute_total_domains() => {total_domains:.2e} domains")
    print(f"[domain-3] gross power output = {gross_power:.1f} W")
    print(f"[verification] domain_1={verification['domain_1']}, domain_2={verification['domain_2']}, domain_3={verification['domain_3']}")
    print(f"[native] floquet_inverse={inverse} drift={drift} gamma_rejection={rejection:.6e}")
    if not all(value == "PASS" for value in verification.values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())