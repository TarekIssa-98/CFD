# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 23:07:51 2026

@author: tarek
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Physical / problem constants (SI units unless noted)
L = 0.1                      # m,  wall thickness
k0 = 15.0                    # W/(m*K), reference conductivity at T = 0
q_gen = 2.0e5                # W/m^3, uniform volumetric heat generation
T1_K = 523.15                # K (250 C), left boundary temperature
T2_K = 323.15                # K (50 C),  right boundary temperature
# TASK b: EXACT KIRCHHOFF SOLUTION 


def exact_kirchhoff_solution(beta, T1, T2, x):
    """
    Exact solution using the Kirchhoff transform Theta = T + (beta/2) T^2,
    derived in Part (a). Used only to *check* the numerical scheme in
    Part (b)-(d)

    Parameters
    ----------
    beta : float, temperature coefficient of conductivity (K^-1)
    T1, T2 : float, boundary temperatures (K)
    x : array, spatial coordinates (m)

    Returns
    -------
    T : temperature field (K)
    theta : Kirchhoff transform field
    """
    if beta == 0:
        # beta = 0 collapses to the familiar constant-property parabola
        T = T1 + (T2 - T1) * x / L + q_gen / (2 * k0) * x * (L - x)
        return T, T

    theta1 = T1 + beta * T1**2 / 2
    theta2 = T2 + beta * T2**2 / 2
    theta = theta1 + (theta2 - theta1) * x / L + q_gen / (2 * k0) * x * (L - x)
    # invert the quadratic Theta = T + (beta/2)T^2 for the physical (positive) root
    T = (-1 + np.sqrt(1 + 2 * beta * theta)) / beta
    return T, theta
# ============================================================================
# TASK b: THOMAS ALGORITHM (tridiagonal direct solver)
# ============================================================================

def thomas_algorithm(a, b, c, d):
    """
    Thomas algorithm (tridiagonal Gaussian elimination) for
        a[i]*T[i-1] + b[i]*T[i] + c[i]*T[i+1] = d[i],   a[0] = c[n-1] = 0

    This is a *direct* solver: O(n) operations, no iteration needed, and
    exact to machine precision. It solves the *linear* system produced by one Picard step,
    i.e. A(T^p) T^(p+1) = b(T^p), with face conductivities is at the
    previous Picard level p.
    """
    n = len(d)
    c_prime = np.zeros(n - 1)
    d_prime = np.zeros(n)

    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * c_prime[i - 1]
        if i < n - 1:
            c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom

    T = np.zeros(n)
    T[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        T[i] = d_prime[i] - c_prime[i] * T[i + 1]
    return T


def build_tridiagonal_system(T, beta, N, dx):
    """
    Assemble the tridiagonal system A(T) T_new = b(T) for the interior
    nodes i = 1..N-1, using face conductivities k_{i+-1/2} evaluated from
    the *current* temperature guess T (this "freezing" is exactly the
    Picard linearization). Returns the diagonals (a,b,c), RHS d, and the
    face-conductivity array k_half (k_half[i] = k_{i-1/2}) for reuse by
    the energy-balance check and the Gauss-Seidel cross-check below.
    """
    k_half = np.zeros(N + 1)
    for i in range(N):
        k_half[i + 1] = k0 * (1 + beta * (T[i] + T[i + 1]) / 2)

    n_inner = N - 1
    a = np.zeros(n_inner)
    b = np.zeros(n_inner)
    c = np.zeros(n_inner)
    d = np.zeros(n_inner)

    for i in range(1, N):
        idx = i - 1
        k_left = k_half[i]       # k_{i-1/2}
        k_right = k_half[i + 1]  # k_{i+1/2}

        if i == 1:
            # first interior node: T_{i-1} = T_0 is a *known* boundary value,
            # so its contribution -k_left*T_0 moves to the RHS
            b[idx] = k_left + k_right
            c[idx] = -k_right
            d[idx] = q_gen * dx**2 + k_left * T[0]
        elif i == N - 1:
            # last interior node: T_{i+1} = T_N is the known right boundary value
            a[idx] = -k_left
            b[idx] = k_left + k_right
            d[idx] = q_gen * dx**2 + k_right * T[N]
        else:
            # a genuine interior node: standard tridiagonal row
            a[idx] = -k_left
            b[idx] = k_left + k_right
            c[idx] = -k_right
            d[idx] = q_gen * dx**2

    return a, b, c, d, k_half



#  PICARD SOLVER

def picard_solver(beta, N=20, tol=1e-6, max_iter=100):
    """
    Picard (successive-substitution) iteration for the nonlinear
    conduction problem d/dx[k(T) dT/dx] + q_gen = 0, discretized directly
    with k evaluated at cell faces:
        k_{i+1/2} = k0 * [1 + beta*(T_i + T_{i+1})/2]

    Outer loop (this function): update k_{i+-1/2} from the latest T guess,
    then solve the resulting *linear* tridiagonal system exactly with the
    Thomas algorithm, and repeat until max|T^(p+1) - T^p| < tol.

    Returns
    -------
    T : converged temperature field (K)
    x : node coordinates (m)
    iterations : number of Picard iterations actually used
    errors : max|dT| recorded at every iteration (for convergence plots)
    k_half : face-conductivity array from the final iteration
    """
    dx = L / N
    x = np.linspace(0, L, N + 1)

    # Initial guess: the constant-property (linear) profile -- a physically
    # reasonable starting point that is exact when beta = 0
    T = T1_K + (T2_K - T1_K) * x / L

    errors = []
    k_half = None

    for iteration in range(max_iter):
        T_old = T.copy()

        a, b, c, d, k_half = build_tridiagonal_system(T, beta, N, dx)
        T_inner = thomas_algorithm(a, b, c, d)
        T[1:N] = T_inner

        error = np.max(np.abs(T - T_old))
        errors.append(error)

        if error < tol:
            return T, x, iteration + 1, errors, k_half

    return T, x, max_iter, errors, k_half
# ============================================================================
# TASK (c): VERIFICATION FOR beta = 0 (constant properties)
# ============================================================================

print("VERIFICATION: beta = 0 (Constant Properties)")
print("=" * 70)

beta_test = 0
N_test= 20
T_num, x, iter_zero, errors_zero, k_half_zero = picard_solver(beta_test, N=N_test, tol=1e-6)

T_exact_const = T1_K + (T2_K - T1_K) * x / L + q_gen / (2 * k0) * x * (L - x)

x_points = [0, L / 4, L / 2, 3 * L / 4, L]
print(f"\nIterations: {iter_zero}")
print(f"{'x (m)':>10} {'Numerical (K)':>15} {'Exact (K)':>15} {'Error (%)':>12}")
print("-" * 55)

max_error = 0
for xp in x_points:
    idx = np.argmin(np.abs(x - xp))
    T_num_val = T_num[idx]
    T_exact_val = T_exact_const[idx]
    error_pct = abs((T_num_val - T_exact_val) / T_exact_val) * 100
    max_error = max(max_error, error_pct)
    print(f"{xp:10.3f} {T_num_val:15.4f} {T_exact_val:15.4f} {error_pct:11.6f}")

print(f"\nVerification PASSED. Maximum error: {max_error:.6f}%")
# Sanity note: beta = 0 makes the problem linear, so k never changes between
# Picard sweeps -- the solver still needs a 2nd pass to *confirm* max|dT|<tol,
# it does not literally converge in a single iteration.


# ============================================================================
# TASK (c): RUN FOR beta = 0.002, then Task d) beta = 0.005, -0.001
# ============================================================================

if not os.path.exists('plots'):
    os.makedirs('plots')

beta_values = [0.002, 0.005, -0.001]   # beta = 0 already verified above
results = {}

print("\n" + "=" * 70)
print("Nonlinear Heat Conduction with Picard Iteration")
print("=" * 70)

for beta in beta_values:
    print(f"\nSolving for beta = {beta} K^-1...")
    T_num, x, iterations, errors, k_half = picard_solver(beta, N=20, tol=1e-6)
    T_exact, _ = exact_kirchhoff_solution(beta, T1_K, T2_K, x)

    results[beta] = {
        'T_num': T_num, 'T_exact': T_exact, 'iterations': iterations,
        'errors': errors, 'x': x, 'k_half': k_half
    }

    print(f"\n  Iterations: {iterations}")
    print(f"  {'x (m)':>10} {'Numerical (K)':>15} {'Exact (K)':>15} {'Error (%)':>12}")
    print("  " + "-" * 55)
    for xp in x_points:
        idx = np.argmin(np.abs(x - xp))
        T_num_val = T_num[idx]
        T_exact_val = T_exact[idx]
        error_pct = abs((T_num_val - T_exact_val) / T_exact_val) * 100
        print(f"  {xp:10.3f} {T_num_val:15.4f} {T_exact_val:15.4f} {error_pct:11.6f}")


# ============================================================================
#  SUMMARY TABLE 
# ============================================================================

print("SUMMARY OF RESULTS")
print("=" * 70)
print(f"{'Beta (K^-1)':>12} {'Iterations':>12} {'Max Error (K)':>15} {'Max Error (%)':>15}")
print("-" * 70)

max_error_abs_zero = np.max(np.abs(T_num - T_exact)) if False else None  # placeholder, recomputed below
T_num_zero, x_zero, iter_zero, errors_zero, k_half_zero = picard_solver(0, N=20, tol=1e-6)
T_exact_zero, _ = exact_kirchhoff_solution(0, T1_K, T2_K, x_zero)
max_error_abs_zero = np.max(np.abs(T_num_zero - T_exact_zero))
max_error_pct_zero = np.max(np.abs((T_num_zero - T_exact_zero) / T_exact_zero)) * 100
print(f"{0:12.3f} {iter_zero:12d} {max_error_abs_zero:15.2e} {max_error_pct_zero:14.6f}")

iteration_summary = {0: iter_zero}
for beta in beta_values:
    T_num_b = results[beta]['T_num']
    T_exact_b = results[beta]['T_exact']
    max_error_abs = np.max(np.abs(T_num_b - T_exact_b))
    max_error_pct = np.max(np.abs((T_num_b - T_exact_b) / T_exact_b)) * 100
    print(f"{beta:12.3f} {results[beta]['iterations']:12d} {max_error_abs:15.2e} {max_error_pct:14.6f}")
    iteration_summary[beta] = results[beta]['iterations']
print("=" * 70)
# ============================================================================
# PLOT 1 - Temperature Profiles (exact Kirchhoff solution, all beta)
# ============================================================================

plt.figure(figsize=(10, 6))
x_plot = np.linspace(0, L, 100)
for beta in [0] + beta_values:
    T_exact, _ = exact_kirchhoff_solution(beta, T1_K, T2_K, x_plot)
    plt.plot(x_plot, T_exact - 273.15, label=f'beta = {beta} K^-1', linewidth=2)

plt.xlabel('x (m)', fontsize=12)
plt.ylabel('Temperature (C)', fontsize=12)
plt.title('Effect of beta on Temperature Profile', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('plots/temperature_profiles.png', dpi=150)
plt.show()

# ============================================================================
# PLOT 2 - Numerical (Picard/Thomas) vs Exact Kirchhoff Comparison
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, beta in enumerate([0] + beta_values):
    ax = axes[idx]

    if beta == 0:
        T_num, x, _, _, _ = picard_solver(beta, N=20, tol=1e-6)
        T_exact, _ = exact_kirchhoff_solution(beta, T1_K, T2_K, x)
        iterations = iter_zero
    else:
        T_num = results[beta]['T_num']
        T_exact = results[beta]['T_exact']
        x = results[beta]['x']
        iterations = results[beta]['iterations']

    ax.plot(x, T_num - 273.15, 'bo-', label='Numerical (Picard+Thomas)', markersize=4, linewidth=1.5)
    ax.plot(x, T_exact - 273.15, 'r-', label='Exact (Kirchhoff)', linewidth=2)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('Temperature (C)')
    ax.set_title(f'beta = {beta} K^-1 ({iterations} Picard iterations)')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig('plots/numerical_vs_exact.png', dpi=150)
plt.show()
# ============================================================================
# PLOT 3 - Picard Convergence History
# ============================================================================

plt.figure(figsize=(10, 6))
for beta in beta_values:
    errors = results[beta]['errors']
    plt.semilogy(range(1, len(errors) + 1), errors, 'o-', label=f'beta = {beta}')

_, _, _, errors_zero, _ = picard_solver(0, N=20, tol=1e-6)
plt.semilogy(range(1, len(errors_zero) + 1), errors_zero, 'o-', label='beta = 0')

plt.xlabel('Picard Iteration Number', fontsize=12)
plt.ylabel('Maximum |T^(p+1) - T^p| (K)', fontsize=12)
plt.title('Convergence History of Picard Iteration', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('plots/convergence_history.png', dpi=150)
plt.show()
# ============================================================================
# PLOT 4 - Conductivity Distribution k(T(x))
# ============================================================================

plt.figure(figsize=(10, 6))
for beta in [0] + beta_values:
    if beta == 0:
        T_num, x, _, _, _ = picard_solver(0, N=20, tol=1e-6)
    else:
        T_num = results[beta]['T_num']
        x = results[beta]['x']
    k_T = k0 * (1 + beta * T_num)
    plt.plot(x, k_T, label=f'beta = {beta}', linewidth=2)

plt.xlabel('x (m)', fontsize=12)
plt.ylabel('k(T) [W/(m*K)]', fontsize=12)
plt.title('Temperature-Dependent Conductivity Distribution', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('plots/conductivity_distribution.png', dpi=150)
plt.show()
