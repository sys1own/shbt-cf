"""Run the deterministic update-5 kernel and diagnostic smoke checks."""

from __future__ import annotations

from shbt_cf.diagnostics import ShbtDiagnosticProcessor
from shbt_cf.native import require_native


def main() -> int:
    kernel = require_native().ShbtNumericalKernel()
    kernel.verify_state_reduction([12.45, 5.13, 1.85, 0.94])
    inverse, _, drift = kernel.compute_floquet_inversion(350.0, -298.57)
    diagnostics = ShbtDiagnosticProcessor()
    diagnostics.calculate_gum_calorimetry([200.0, 900.0, 1600.0, 2300.0, 3000.0])
    _, rejection = diagnostics.process_psd_validation(2_995_731, 0)
    print(f"status=ok floquet_inverse={inverse} drift={drift} gamma_rejection={rejection:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())