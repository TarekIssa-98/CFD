# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 21:24:55 2026

@author: tarek
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.optimize import brentq
from IPython.display import display
from scipy.linalg import lu_factor, lu_solve
import time

# ---------------- Physical data (as given in the problem statement) ----------------
R     = 0.02          # m,          cylinder radius
alpha = 1.0e-5         # m^2/s,      thermal diffusivity
k     = 20.0           # W/(m K),    thermal conductivity
h     = 200.0          # W/(m^2 K),  convective coefficient
Ti    = 300.0          # deg C,      initial temperature
Tinf  = 25.0           # deg C,      ambient temperature

Bi = h*R/k
theta_i = Ti - Tinf

print(f"Bi      = {Bi:.4f}")
print(f"theta_i = {theta_i:.1f} K")

# Build the system:
def build_A(N):
    '''Tridiagonal system matrix for d(theta)/dFo_grid = A theta, on N+1 nodes r_i = i*dr.'''
    dr = R/N                  # Physical distance between nodes
    delta = dr/R              # Dimensionless grid spacing (= 1/N)
    A = np.zeros((N+1, N+1))  # Initialize (N+1)x(N+1) matrix with zeros
    
    # centre node (i = 0): theta_0' = -4 theta_0 + 4 theta_1
    A[0, 0] = -4.0            # Coefficient for theta_0 at the center
    A[0, 1] =  4.0            # Coefficient for theta_1 at the center
    
    # interior nodes
    for i in range(1, N):
        A[i, i-1] = 1.0 - 1.0/(2*i)  # Coeff for theta_{i-1}
        A[i, i]   = -2.0             # Coeff for theta_i
        A[i, i+1] = 1.0 + 1.0/(2*i)  # Coeff for theta_{i+1}
        
    # outer boundary node (i = N), convective BC eliminated via ghost node
    A[N, N-1] = 2.0                                # Coeff for theta_{N-1} at boundary
    A[N, N]   = -(2.0 + 2.0*Bi*delta + Bi*delta**2) # Coeff for theta_N at boundary
    
    return A, dr, delta

N = 50                      # Number of spatial intervals -> N+1 nodes
A, dr, delta = build_A(N)   # Build the system matrix and get grid spacing
r_nodes = np.linspace(0, R, N+1) # Array of physical node coordinates

Fo_grid_max = 0.25           # Explicit stability limit derived in Task (a)
dt_max = Fo_grid_max * dr**2 / alpha # Max allowable time step (dt = Fo * dr^2 / alpha)
print(f"N = {N},  dr = {dr:.3e} m,  explicit dt_max = {dt_max:.4e} s") # Print grid info

theta0 = theta_i * np.ones(N+1)   # Initial condition: theta(r,0) = theta_i everywhere

def dFo_grid(dt):
    '''Grid Fourier number for a real time step dt (seconds).'''
    return alpha*dt/dr**2         # Calculates Fo = alpha * dt / dr^2 for the given dt

#Numerical schemes:
    
def run_explicit(t_target, dt):

    steps = int(np.ceil(t_target/dt))   # Calculate minimum integer steps to reach t_target
    dt_actual = t_target/steps          # Adjust dt slightly so steps * dt_actual = t_target exactly
    M = np.eye(N+1) + dFo_grid(dt_actual)*A  # Build explicit update matrix: (I + Fo*A)
    th = theta0.copy()                  # Copy initial condition to avoid modifying the original
    for _ in range(steps):              # Time-stepping loop
        th = M @ th                     # Apply explicit update: theta^{n+1} = M * theta^n
    return th, steps                    # Return final temperature profile and step count

def run_implicit(t_target, dt):
    '''(I - dFo*A) theta^{n+1} = theta^n.  Matrix is constant for fixed dt,
    so it is LU-factorised once and reused every step.'''
    steps = int(np.ceil(t_target/dt))   # Calculate minimum integer steps to reach t_target
    dt_actual = t_target/steps          # Adjust dt slightly so steps * dt_actual = t_target exactly
    Mmat = np.eye(N+1) - dFo_grid(dt_actual)*A  # Build implicit LHS matrix: (I - Fo*A)
    lu, piv = lu_factor(Mmat)           # LU-factorize once (constant matrix)
    th = theta0.copy()                  # Copy initial condition
    for _ in range(steps):              # Time-stepping loop
        th = lu_solve((lu, piv), th)    # Solve linear system each step using stored LU factors
    return th, steps                    # Return final temperature profile and step count

def run_CN(t_target, dt):
    '''(I - dFo/2 A) theta^{n+1} = (I + dFo/2 A) theta^n, same LU-reuse idea.'''
    steps = int(np.ceil(t_target/dt))   # Calculate minimum integer steps to reach t_target
    dt_actual = t_target/steps          # Adjust dt slightly so steps * dt_actual = t_target exactly
    dFo = dFo_grid(dt_actual)           # Calculate grid Fourier number for the adjusted dt
    Mmat = np.eye(N+1) - 0.5*dFo*A      # Build Crank-Nicolson LHS matrix: (I - Fo/2 * A)
    Nmat = np.eye(N+1) + 0.5*dFo*A      # Build Crank-Nicolson RHS matrix: (I + Fo/2 * A)
    lu, piv = lu_factor(Mmat)           # LU-factorize LHS once (constant matrix)
    th = theta0.copy()                  # Copy initial condition
    for _ in range(steps):              # Time-stepping loop
        th = lu_solve((lu, piv), Nmat @ th)  # Solve: LHS * theta^{n+1} = RHS * theta^n
    return th, steps                    # Return final temperature profile and step count

# Finding Lamda_coeff:
    
def f_char(lam):
    # Characteristic equation for the convective boundary condition:
    # lambda * J1(lambda) - Bi * J0(lambda) = 0
    return lam*jv(1, lam) - Bi*jv(0, lam)

def find_roots(n_roots, search_max=250.0, n_scan=40000):
    # Scan the characteristic function over a dense grid to find sign changes
    lam_scan = np.linspace(1e-8, search_max, n_scan)  # Avoid lambda=0 (singularity)
    f_scan = f_char(lam_scan)                          # Evaluate f at all scan points
    roots = []                                         # Store found eigenvalues
    for i in range(len(lam_scan)-1):
        if f_scan[i]*f_scan[i+1] < 0:                  # Sign change -> root in this interval
            # Refine the root using Brent's method (high precision)
            root = brentq(f_char, lam_scan[i], lam_scan[i+1], xtol=1e-13, rtol=1e-14)
            roots.append(root)
            if len(roots) >= n_roots:                  # Stop once we have enough roots
                break
    return np.array(roots)                             # Return eigenvalues as an array

N_TERMS = 30                                           # Number of terms in the series solution
lambdas = find_roots(N_TERMS)                          # Compute the first 30 eigenvalues
# Compute the series coefficients C_n from the analytical solution
Cn = 2.0/lambdas * jv(1, lambdas) / (jv(0, lambdas)**2 + jv(1, lambdas)**2)

roots_table = pd.DataFrame({
    'n': np.arange(1, 7),                              
    'lambda_n': lambdas[:6],                           
    'C_n': Cn[:6]                                      
})

roots_table.style.hide(axis='index').format({
    'lambda_n': '{:.6f}',
    'C_n': '{:.6f}'
}).set_caption(f"First 6 eigenvalues and series coefficients (Bi = {Bi:g})")

# Convergence check :
def theta_ratio_analytical(r, Fo, n_terms=None):
    '''Bessel-series solution'''

    nt = len(lambdas) if n_terms is None else n_terms  # Use all roots if n_terms not specified
    lam = lambdas[:nt]                                 # Select first nt eigenvalues
    c = Cn[:nt]                                        # Select corresponding coefficients
    # Sum the series: theta/theta_i = sum( C_n * exp(-lambda_n^2 * Fo) * J0(lambda_n * r/R) )
    return np.sum(c * np.exp(-lam**2 * Fo) * jv(0, lam*r/R))

def convergence_check(Fo, r, terms_list=(5, 10, 15, 20, 25, 30)):
    values, changes = [], []                           # Store series values and step-to-step changes
    prev = None                                        # Track previous value for change computation

    for nt in terms_list:
        val = theta_ratio_analytical(r=r, Fo=Fo, n_terms=nt)  # Evaluate series with nt terms
        values.append(val)
        # Compute change from previous iteration (NaN for the first term)
        changes.append(np.nan if prev is None else val - prev)
        prev = val                                     # Update previous value

    table = pd.DataFrame({
        'Number of terms': list(terms_list),
        'theta/theta_i': values,
        'Change from previous': changes,
    })

    return table.style.hide(axis='index').format({
        'theta/theta_i': '{:.8f}',
        'Change from previous': lambda x: '' if pd.isna(x) else f'{x:+.2e}'
    }).set_caption(f"Convergence of the series at Fo = {Fo}, r = {r:g} m")

# Worst case for convergence: centre, smallest Fo
display(convergence_check(Fo=0.1, r=0.0))              # Test convergence at r=0 (centre)
# A second check at the outer surface, same Fo:
display(convergence_check(Fo=0.1, r=R))                # Test convergence at r=R (surface)

# Comparison b/w schemes:
    
dt_common = 0.9*dt_max
Fo_targets = [0.1, 0.3, 0.6]
eval_labels = ["r=0", "r=R/2", "r=R"]
eval_idx = [0, N//2, N]

rows = []
max_err = {"Explicit": 0.0, "Implicit": 0.0, "Crank-Nicolson": 0.0}

for Fo in Fo_targets:
    t_target = Fo*R**2/alpha
    th_e, steps_e = run_explicit(t_target, dt_common)
    th_i, steps_i = run_implicit(t_target, dt_common)
    th_c, steps_c = run_CN(t_target, dt_common)

    for label, idx in zip(eval_labels, eval_idx):
        r = r_nodes[idx]
        th_a = theta_ratio_analytical(r, Fo) * theta_i
        row = {
            "Fo": Fo, "location": label,
            "Analytical T (C)": th_a + Tinf,
            "Explicit T (C)": th_e[idx] + Tinf,
            "Implicit T (C)": th_i[idx] + Tinf,
            "CN T (C)": th_c[idx] + Tinf,
            "|err| Explicit": abs(th_e[idx]-th_a),
            "|err| Implicit": abs(th_i[idx]-th_a),
            "|err| CN": abs(th_c[idx]-th_a),
        }
        rows.append(row)
        max_err["Explicit"] = max(max_err["Explicit"], row["|err| Explicit"])
        max_err["Implicit"] = max(max_err["Implicit"], row["|err| Implicit"])
        max_err["Crank-Nicolson"] = max(max_err["Crank-Nicolson"], row["|err| CN"])

comparison_df = pd.DataFrame(rows)
n_steps_06 = int(np.ceil(0.6*R**2/alpha/dt_common))

styled_comparison = (
    comparison_df.style
    .hide(axis='index')
    .format({
        "Fo": '{:.1f}',
        "Analytical T (C)": '{:.4f}', "Explicit T (C)": '{:.4f}',
        "Implicit T (C)": '{:.4f}', "CN T (C)": '{:.4f}',
        "|err| Explicit": '{:.2e}', "|err| Implicit": '{:.2e}', "|err| CN": '{:.2e}',
    })
    .set_caption(f"Scheme vs. analytical series, dt = {dt_common:.4e} s "
                 f"({n_steps_06} steps to reach Fo = 0.6)")
)
display(styled_comparison)

max_err_table = pd.DataFrame({
    "Scheme": list(max_err.keys()),
    "Max |error| over all (Fo, location) [K]": list(max_err.values()),
})
display(max_err_table.style.hide(axis='index').format({
    "Max |error| over all (Fo, location) [K]": '{:.5f}'
}).set_caption("Maximum error per scheme, Task (c)"))

#plots:

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
r_dense = np.linspace(0, R, 200)

for ax, Fo in zip(axes, Fo_targets):
    t_target = Fo*R**2/alpha
    T_analytical_dense = np.array([theta_ratio_analytical(r, Fo) for r in r_dense])*theta_i + Tinf
    th_e, _ = run_explicit(t_target, dt_common)
    th_i, _ = run_implicit(t_target, dt_common)
    th_c, _ = run_CN(t_target, dt_common)

    ax.plot(r_dense*1000, T_analytical_dense, 'k-', lw=2, label='Analytical (series)')
    ax.plot(r_nodes*1000, th_e+Tinf, 'o', ms=4, label='Explicit')
    ax.plot(r_nodes*1000, th_i+Tinf, 's', ms=4, label='Implicit')
    ax.plot(r_nodes*1000, th_c+Tinf, '^', ms=4, label='Crank-Nicolson')
    ax.set_xlabel('r (mm)')
    ax.set_title(f'Fo = {Fo}')
    ax.grid(alpha=0.3)

axes[0].set_ylabel('T (deg C)')
axes[0].legend()
fig.suptitle('Temperature profile: numerical schemes vs. analytical series')
plt.tight_layout()
plt.show()

# Cpu time comparison:

t_target = 0.6*R**2/alpha
th_a_full = np.array([theta_ratio_analytical(r, 0.6) for r in r_nodes]) * theta_i

tol = 0.01   # K -- fixed engineering accuracy target (comfortably above the
             # baseline errors found for all three schemes in Task (c))

def worst_err(run_func, dt):
    th, steps = run_func(t_target, dt)
    return np.max(np.abs(th - th_a_full)), steps

def max_dt_for_tol(run_func, tol, lo_factor=1.0, hi_factor=2000.0, iters=30):
    lo, hi = lo_factor, hi_factor
    e_lo, _ = worst_err(run_func, dt_common*lo)
    assert e_lo <= tol, "baseline dt already exceeds tolerance"
    for _ in range(iters):
        mid = 0.5*(lo+hi)
        e_mid, _ = worst_err(run_func, dt_common*mid)
        if e_mid <= tol:
            lo = mid
        else:
            hi = mid
    return dt_common*lo

dt_impl_opt = max_dt_for_tol(run_implicit, tol)
dt_cn_opt   = max_dt_for_tol(run_CN, tol)

steps_expl = int(np.ceil(t_target/dt_common))
steps_impl = int(np.ceil(t_target/dt_impl_opt))
steps_cn   = int(np.ceil(t_target/dt_cn_opt))

def time_it(run_func, dt, repeats=5):
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        run_func(t_target, dt)
        t1 = time.perf_counter()
        best = min(best, t1-t0)
    return best

cpu_expl = time_it(run_explicit, dt_common)
cpu_impl = time_it(run_implicit, dt_impl_opt)
cpu_cn   = time_it(run_CN, dt_cn_opt)

summary = pd.DataFrame({
    "scheme": ["Explicit", "Implicit", "Crank-Nicolson"],
    "dt used (s)": [dt_common, dt_impl_opt, dt_cn_opt],
    "steps to Fo=0.6": [steps_expl, steps_impl, steps_cn],
    "CPU time (s)": [cpu_expl, cpu_impl, cpu_cn],
    "max error (K)": [worst_err(run_explicit, dt_common)[0], worst_err(run_implicit, dt_impl_opt)[0], worst_err(run_CN, dt_cn_opt)[0]],
})

display(summary.style.hide(axis='index').format({
    "dt used (s)": '{:.4e}',
    "steps to Fo=0.6": '{:d}',
    "CPU time (s)": '{:.4e}',
    "max error (K)": '{:.5f}',
}).set_caption(f"Steps and CPU time for a fixed accuracy target (max error <= {tol:g} K)"))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
colors = ['#d62728', '#1f77b4', '#2ca02c']

axes[0].bar(summary["scheme"], summary["steps to Fo=0.6"], color=colors)
axes[0].set_ylabel('time steps to reach Fo = 0.6')
axes[0].set_yscale('log')
axes[0].set_title(f'Steps needed for max error <= {tol:g} K')
for i, v in enumerate(summary["steps to Fo=0.6"]):
    axes[0].text(i, v*1.15, str(v), ha='center')
axes[0].grid(alpha=0.3, axis='y')

axes[1].bar(summary["scheme"], summary["CPU time (s)"], color=colors)
axes[1].set_ylabel('CPU time (s)')
axes[1].set_yscale('log')
axes[1].set_title(f'Wall-clock time for max error <= {tol:g} K')
for i, v in enumerate(summary["CPU time (s)"]):
    axes[1].text(i, v*1.15, f'{v:.1e}', ha='center')
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()