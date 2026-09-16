#ifndef SHBT_CF_CORE_PARAMETERS_HPP
#define SHBT_CF_CORE_PARAMETERS_HPP

#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <utility>

namespace shbt::cf {

constexpr double PI = 3.14159265358979323846;
constexpr double C_LIGHT = 299792458.0;             // m/s
constexpr double HBAR_EV = 6.582119569e-16;         // eV s
constexpr double ELEMENTARY_CHARGE = 1.602176634e-19; // C
constexpr double EPSILON_0 = 8.8541878128e-12;       // F/m

// update-11.1 Task 1 — Pd_{0.9132}Ir_{0.0868}D_x many-body / CFT parameters.
constexpr double PD_IR_IRIDIUM_FRACTION = 0.0868;
constexpr double PD_IR_DOS_STATES_EV_ATOM = 1.4606;
constexpr double PD_IR_THOMAS_FERMI_LENGTH_ANGSTROM = 0.4137;
constexpr double PD_IR_THOMAS_FERMI_LENGTH_M = 0.4137e-10;
constexpr double SHBT_COHERENCE_FACTOR_C_COH = 0.8821;
constexpr double SHBT_BOUNDARY_G_FACTOR_CHI_CFT = 0.9412;
constexpr double SHBT_HOLOGRAPHIC_RESONANCE_FREQ_HZ = 6.92e8;
constexpr double SHBT_SCREENING_SHIFT_UE_EV = 350.0034;
constexpr double SHBT_HOLOGRAPHIC_NOISE_FLOOR_EV = 1e-35;

// update-11.1 Task 2 — microchannel cold plate, TEG/BOP ledger, loading caps.
constexpr double MHS_COLD_PLATE_WIDTH_M = 0.120;
constexpr double MHS_COLD_PLATE_LENGTH_M = 0.120;
constexpr double MHS_CHANNEL_WIDTH_M = 250.0e-6;
constexpr double MHS_FIN_WIDTH_M = 250.0e-6;
constexpr double MHS_CHANNEL_HEIGHT_M = 1500.0e-6;
constexpr int MHS_CHANNEL_COUNT = 240;
constexpr double MHS_COOLANT_FLOW_RATE_M3_S = 3.0e-5; // 1.8 L/min
constexpr double MHS_WASTE_HEAT_Q_COLD_W = 2047.86;
constexpr double MHS_PRESSURE_DROP_PA = 6848.77;
constexpr double MHS_PUMP_POWER_W = 0.3161;
constexpr double MHS_TOTAL_THERMAL_RESISTANCE_KW = 0.027702;
constexpr double TEG_HOT_SIDE_TEMP_K = 600.0;
constexpr double TEG_COLD_SIDE_TEMP_K = 357.47;
constexpr double TEG_MAX_BASE_TEMP_LIMIT_K = 358.0;
constexpr double TEG_CONVERSION_EFFICIENCY = 0.1079;
constexpr double BOP_PARASITIC_TOTAL_W = 200.3161;
constexpr double BOP_NET_EXPORT_POWER_W = 47.4039;
constexpr double D_PD_MAX_PHYSICAL_LOADING_CAP = 0.904;
constexpr double D_PD_SIMULATION_UPPER_BOUND = 0.81;

struct MetallurgicalFatigueParameters {
    double bondline_thickness_tlp = 3.5e-6;
    double bondline_thickness_graded = 15.0e-6;
    double T_min_K = 293.15;
    double T_max_K = 550.0;
    double delta_T_K = 256.85;

    double E_modulus = 128.5e9;
    double yield_strength = 380.0e6;
    double sigma_f_prime = 545.0e6;
    double b_fatigue_exponent = -0.082;
    double epsilon_f_prime = 0.320;
    double c_fatigue_exponent = -0.560;

    double C1_chaboche = 42.5e9;
    double gamma1_chaboche = 450.0;
    double C2_chaboche = 12.8e9;
    double gamma2_chaboche = 85.0;

    double delta_epsilon_p_half = 0.0004940236;
    double delta_epsilon_e_half = 0.0016437048;

    [[nodiscard]] double calculate_total_strain_amplitude(double Nf) const {
        const double reversals = 2.0 * Nf;
        const double elastic_part =
            (sigma_f_prime / E_modulus) * std::pow(reversals, b_fatigue_exponent);
        const double plastic_part = epsilon_f_prime * std::pow(reversals, c_fatigue_exponent);
        return elastic_part + plastic_part;
    }

    [[nodiscard]] double verify_fatigue_life() const {
        constexpr double nf_target = 52400.0;
        return calculate_total_strain_amplitude(nf_target);
    }
};

struct FloquetDielectricSolver {
    double lambda_1 = 785.0e-9;
    double lambda_2 = 802.5e-9;
    double r_d_interaction = 0.280e-10;
    double F_z_enhancement = 21.45;
    int max_floquet_order = 2;
    double carrier_density_ne = 4.85e28;

    [[nodiscard]] double compute_beat_frequency() const {
        const double omega1 = 2.0 * PI * C_LIGHT / lambda_1;
        const double omega2 = 2.0 * PI * C_LIGHT / lambda_2;
        return std::abs(omega1 - omega2) / (2.0 * PI * 1.0e12);
    }

    [[nodiscard]] double compute_beat_frequency_rad_per_s() const {
        const double omega1 = 2.0 * PI * C_LIGHT / lambda_1;
        const double omega2 = 2.0 * PI * C_LIGHT / lambda_2;
        return std::abs(omega1 - omega2);
    }

    [[nodiscard]] double compute_bare_potential() const {
        const double r_angstrom = r_d_interaction * 1.0e10;
        return 14.3996 / r_angstrom;
    }

    [[nodiscard]] std::pair<double, double> evaluate_effective_potentials() const {
        const double U_eff_eV = 350.0;
        const double V_driven_eV = -298.57;
        const double V_eff_net = U_eff_eV + V_driven_eV;
        const double V_bare_eV = compute_bare_potential();
        std::cout << "[context-4] V_bare = " << V_bare_eV << " eV, V_eff = " << V_eff_net
                  << " eV\n";
        return {V_bare_eV, V_eff_net};
    }
};

struct MultiScaleVolumeIntegrator {
    double P_th_gross = 2911.40;
    double V_metal = 2.875e-12;
    double V_active = 2.530e-10;
    double d_domain = 10.0e-9;
    double k_extinction = 4.862;

    [[nodiscard]] double compute_optical_skin_depth() const {
        const double lambda_avg = 0.5 * (785.0e-9 + 802.5e-9);
        return lambda_avg / (4.0 * PI * k_extinction);
    }

    [[nodiscard]] double compute_domain_volume() const {
        return std::pow(d_domain, 3.0);
    }

    [[nodiscard]] double compute_domain_density() const {
        return 1.0 / compute_domain_volume();
    }

    [[nodiscard]] double compute_total_domains() const {
        return compute_domain_density() * V_active;
    }

    [[nodiscard]] double compute_coverage_factor() const {
        return V_metal / V_active;
    }

    [[nodiscard]] double compute_q_active_volumetric() const {
        return P_th_gross / V_active;
    }

    [[nodiscard]] double compute_q_metal_volumetric() const {
        return P_th_gross / V_metal;
    }
};

} // namespace shbt::cf

#endif // SHBT_CF_CORE_PARAMETERS_HPP
