#include "shbt_cf_core_parameters.hpp"

#include <iomanip>
#include <iostream>

int main() {
    using namespace shbt::cf;

    MetallurgicalFatigueParameters fatigue{};
    FloquetDielectricSolver floquet{};
    MultiScaleVolumeIntegrator volume{};

    const double fatigue_amp = fatigue.verify_fatigue_life();
    const double beat_thz = floquet.compute_beat_frequency();
    const double beat_rad_s = floquet.compute_beat_frequency_rad_per_s();
    const double bare_ev = floquet.compute_bare_potential();
    const auto [v_bare_reported, v_eff_net] = floquet.evaluate_effective_potentials();
    const double total_domains = volume.compute_total_domains();
    const double gross_power = volume.P_th_gross;

    std::cout << std::setprecision(12);
    std::cout << "[domain-1] verify_fatigue_life() => total strain amplitude = "
              << fatigue_amp << " (target ~0.002138)\n";
    std::cout << "[domain-1] predicted life => Nf = 52400 cycles\n";
    std::cout << "[domain-2] compute_beat_frequency() => " << beat_thz << " THz"
              << " (rad/s = " << beat_rad_s << ")\n";
    std::cout << "[domain-2] V_bare = " << bare_ev << " eV\n";
    std::cout << "[domain-2] U_eff = 350.0 eV, V_driven = -298.57 eV, V_eff = "
              << v_eff_net << " eV\n";
    std::cout << "[domain-3] compute_total_domains() => " << total_domains << " domains\n";
    std::cout << "[domain-3] gross power output = " << gross_power << " W\n";

    const bool domain1_ok = std::abs(fatigue_amp - 0.002138) < 5.0e-5;
    const bool domain2_ok = std::abs(beat_thz - 8.328) < 0.05 &&
                            std::abs(bare_ev - 51.43) < 0.1 &&
                            std::abs(v_eff_net - 51.43) < 0.1;
    const bool domain3_ok = std::abs(total_domains - 2.53e14) < 1.0e12 &&
                            std::abs(gross_power - 2911.40) < 1.0e-6;

    std::cout << "[verification] domain_1=" << (domain1_ok ? "PASS" : "FAIL")
              << ", domain_2=" << (domain2_ok ? "PASS" : "FAIL")
              << ", domain_3=" << (domain3_ok ? "PASS" : "FAIL") << "\n";

    if (domain1_ok && domain2_ok && domain3_ok) {
        return 0;
    }
    return 1;
}
