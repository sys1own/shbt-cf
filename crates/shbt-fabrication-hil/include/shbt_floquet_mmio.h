#ifndef SHBT_FLOQUET_MMIO_H
#define SHBT_FLOQUET_MMIO_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FLOQUET_MATRIX_DIM 15625u
#define MMIO_BASE_ADDR     0x7F0000000000ULL
#define GPU_ALIGNMENT      64u

/* 64-byte aligned complex element for the Floquet dielectric matrix. */
typedef struct __attribute__((aligned(64))) {
    double real;
    double imag;
} complex64_t;

/* Zero-copy C-ABI hardware MMIO control map, widened to a 128-byte dual
 * cacheline record (cf3 spec §6). Binary layout is fixed across host C11
 * drivers, Rust orchestration wrappers, and CUDA/ROCm kernels. */
typedef struct __attribute__((packed, aligned(64))) {
    volatile uint32_t ctrl_status;        /* 0x0000: Control/Status Bits */
    uint32_t          reserved0;
    volatile double   thermal_surge_w;    /* 0x0008: Inputs P_thermal (W) */
    volatile double   x_deuterium_avg;    /* 0x0010: Deuterium loading x(r,t) */
    volatile double   u_screen_eff_ev;    /* 0x0018: Screening potential U_eff (eV) */
    volatile double   b_lat_fraction;     /* 0x0020: Branching fraction B_lat */
    volatile double   plastic_strain_max; /* 0x0028: Peak Chaboche plastic strain */
    volatile double   dose_surface_usv;   /* 0x0030: OpenMC surface dose rate (uSv/h) */
    volatile uint64_t gpu_frame_count;    /* 0x0038: HIL frame counter */
    uint32_t          matrix_dim;         /* 0x0040: Dimension = 15625 */
    uint32_t          reserved1;
    volatile uint64_t d_floquet_ptr;      /* 0x0048: GPUDirect device pointer */
    uint8_t           pad[16];            /* 0x0050: Pad to 128-byte dual cacheline */
} shbt_mmio_control_t;

/*
 * Executes one real-time HIL frame against the MMIO control block.
 * Returns 0 on success; -1 dimension/alignment mismatch, -2 accelerator
 * offline, -3 boundary threshold breach, -4 physics gate failure.
 */
int shbt_floquet_mmio_execute_frame(shbt_mmio_control_t* mmio, complex64_t* h_matrix);

#ifdef __cplusplus
}
#endif

#endif /* SHBT_FLOQUET_MMIO_H */
