#ifndef SHBT_RCWA_FLOQUET_H
#define SHBT_RCWA_FLOQUET_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Solves the complex system A * x = b for Floquet RCWA.
 *
 * Uses an LU decomposition with partial pivoting (shbt-rcwa `floquet_lu`).
 *
 * @param a_re Real parts of matrix A, n*n entries in row-major order.
 * @param a_im Imaginary parts of matrix A, n*n entries in row-major order.
 * @param b_re Real parts of the right-hand side vector b, n entries.
 * @param b_im Imaginary parts of the right-hand side vector b, n entries.
 * @param n    Dimension of the square matrix.
 * @param x_re Output buffer for the real parts of x, n entries.
 * @param x_im Output buffer for the imaginary parts of x, n entries.
 * @return 0 on success;
 *         -1 if any pointer is NULL;
 *         -2 if A is singular (pivot magnitude below 1e-16);
 *         -3 on a solve-stage buffer error.
 */
int32_t shbt_rcwa_solve_floquet(
    const double* a_re,
    const double* a_im,
    const double* b_re,
    const double* b_im,
    size_t n,
    double* x_re,
    double* x_im
);

#ifdef __cplusplus
}
#endif

#endif /* SHBT_RCWA_FLOQUET_H */
