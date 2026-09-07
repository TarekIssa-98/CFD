# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 00:58:52 2026

@author: tarek
"""

import numpy as np
import time
import tracemalloc
from scipy.sparse import lil_matrix, csc_matrix, csr_matrix
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

# Domain & Physical Parameters
R = 0.05          # m
k = 40.0          # W/(m*K)
h = 25.0           # W/(m^2*K)
T_inf = 25.0       # degC

# Flux Fourier coefficients
q0 = 2000.0        # W/m^2  (uniform component)
q1 = 1200.0        # W/m^2  (cos theta component)
q2 = 800.0         # W/m^2  (sin theta component)
q3 = 500.0         # W/m^2  (cos 3*theta component)

def q_flux(theta):
    """Prescribed heat flux at angle theta."""
    return q0 + q1*np.cos(theta) + q2*np.sin(theta) + q3*np.cos(3*theta)

def analytical_solution(r, theta):
    """Closed-form analytical solution (for validation only)."""
    if r == 0:
        return T_inf + q0/h
    r_R = r / R
    A0 = q0/h
    A1 = q1/(h + k/R)
    B1 = q2/(h + k/R)
    A3 = q3/(h + 3*k/R)
    return T_inf + A0 + r_R*(A1*np.cos(theta) + B1*np.sin(theta)) + r_R**3*A3*np.cos(3*theta)

def analytical_surface(theta):
    """Analytical solution evaluated at r = R."""
    return analytical_solution(R, theta)

print(f"A0={q0/h:.4f}, A1={q1/(h+k/R):.4f}, B1={q2/(h+k/R):.4f}, A3={q3/(h+3*k/R):.4f}")
#----------------------------------------------
# Grid Parameters
Nr = 30        # radial divisions (r = 1..Nr, i=0 is the pole)
Ntheta = 60    # angular divisions (0..2*pi, periodic)

dr = R / Nr
dtheta = 2*np.pi / Ntheta
N = Nr * Ntheta          # total unknowns

TOL = 1e-6
MAX_ITER = 20000

print(f"Nr={Nr}, Ntheta={Ntheta}  ->  N={N} unknowns")
print(f"dr={dr:.6f} m, dtheta={np.rad2deg(dtheta):.3f} deg")
#=======================================================
# Matrix System:
# Discretization and boundary conditions 
def idx(i, j, Ntheta):
    """Map (i,j) grid indices (i=1..Nr, j=0..Ntheta-1) to a flat equation index."""
    return (i-1)*Ntheta + j

def build_system(Nr, Ntheta):
    dr = R / Nr
    dtheta = 2*np.pi / Ntheta
    N = Nr * Ntheta
    A = lil_matrix((N, N))
    b = np.zeros(N)

    counts = {'interior_eqns': 0, 'pole_coupled_eqns': 0, 'surface_eqns': 0}

    # (a) + (b): interior points, i = 1..Nr-1
    for i in range(1, Nr):
        r_i = i * dr
        for j in range(Ntheta):
            p = idx(i, j, Ntheta)
            A[p, p] = -2.0/dr**2 - 2.0/(r_i**2 * dtheta**2)
            A[p, idx(i+1, j, Ntheta)] = 1.0/dr**2 + 1.0/(2*r_i*dr)   # outward neighbor

            if i == 1:
                # (b) pole condition: couples to the WHOLE i=1 ring
                coeff = (1.0/dr**2 - 1.0/(2*r_i*dr)) * (1.0/Ntheta)
                for kk in range(Ntheta):
                    A[p, idx(1, kk, Ntheta)] += coeff
                counts['pole_coupled_eqns'] += 1
            else:
                A[p, idx(i-1, j, Ntheta)] = 1.0/dr**2 - 1.0/(2*r_i*dr)  # inward neighbor

            # (c) periodic angular neighbors
            #If j=N_theta −1, j=N_theta −1 (last node) , then (j+1) % Ntheta = 0 (wraps to first node)
            #if j = 0 (first node),  then (j-1) % Ntheta = N_\theta - 1 (wraps to last node)
            j_plus, j_minus = (j+1) % Ntheta, (j-1) % Ntheta
            A[p, idx(i, j_plus, Ntheta)] += 1.0/(r_i**2 * dtheta**2)
            A[p, idx(i, j_minus, Ntheta)] += 1.0/(r_i**2 * dtheta**2)

            b[p] = 0.0
            counts['interior_eqns'] += 1

    # (d) outer Robin BC, i = Nr
    for j in range(Ntheta):
        i = Nr
        p = idx(i, j, Ntheta)
        theta_j = j * dtheta
        A[p, p] = k/dr + h
        A[p, idx(i-1, j, Ntheta)] = -k/dr
        b[p] = h*T_inf + q_flux(theta_j)
        counts['surface_eqns'] += 1

    return A.tocsc(), b, N, counts

A, b, N, bc_counts = build_system(Nr, Ntheta)
print(bc_counts)
#=========================================================
# Matrix information
nnz = A.nnz
mem_kb = (A.data.nbytes + A.indices.nbytes + A.indptr.nbytes) / 1024

matrix_summary = pd.DataFrame([{
    'Unknowns (N)': N,
    'Matrix shape': f'{A.shape[0]} x {A.shape[1]}',
    'Non-zeros': nnz,
    'Sparsity (% non-zero)': round(nnz / N**2 * 100, 4),
    'Memory (KB)': round(mem_kb, 2),
    'Interior eqns': bc_counts['interior_eqns'],
    'Pole-coupled eqns': bc_counts['pole_coupled_eqns'],
    'Surface (Robin) eqns': bc_counts['surface_eqns'],
}])
display(matrix_summary)
#===============================================
# Coefficent of Matrix check
print("coefficient assembly check")

def check_row(label, p, expected):
    row = A[p, :].toarray().flatten()
    ok = True
    for col, exp_val in expected.items():
        actual = row[col]
        match = abs(actual - exp_val) < 1e-9
        ok &= match
        print(f"  {label} col {col}: actual={actual:.6f}, expected={exp_val:.6f}  {'OK' if match else 'MISMATCH'}")
    return ok

# Interior stencil at a sample point (i=2, j=1)
i_t, j_t = 2, 1
r_i = i_t * dr
p_t = idx(i_t, j_t, Ntheta)
expected_int = {
    p_t:                                    -2/dr**2 - 2/(r_i**2*dtheta**2),
    idx(i_t+1, j_t, Ntheta):                  1/dr**2 + 1/(2*r_i*dr),
    idx(i_t-1, j_t, Ntheta):                  1/dr**2 - 1/(2*r_i*dr),
    idx(i_t, (j_t+1) % Ntheta, Ntheta):       1/(r_i**2*dtheta**2),
    idx(i_t, (j_t-1) % Ntheta, Ntheta):       1/(r_i**2*dtheta**2),
}
check_row("Interior", p_t, expected_int)

# Pole regularity row (i=1, j=0) - diagonal entry + one coupling entry
r_p = 1 * dr
coeff_pole = (1/dr**2 - 1/(2*r_p*dr)) / Ntheta
p_pole = idx(1, 0, Ntheta)
expected_pole = {
    p_pole:            -2/dr**2 - 2/(r_p**2*dtheta**2) + coeff_pole,
    idx(1, 5, Ntheta):  coeff_pole,   # any other angular index should carry the same coupling
}
check_row("Pole", p_pole, expected_pole)

# Outer Robin BC row (i=Nr, j=0)
p_surf = idx(Nr, 0, Ntheta)
expected_surf = {
    p_surf:                    k/dr + h,
    idx(Nr-1, 0, Ntheta):     -k/dr,
}
check_row("Surface", p_surf, expected_surf)
expected_b = h*T_inf + q_flux(0.0)
print(f"  Surface RHS: actual={b[p_surf]:.6f}, expected={expected_b:.6f}  {'OK' if abs(b[p_surf]-expected_b)<1e-9 else 'MISMATCH'}")
#======================================================
# Measuring wall-clock time, CPU time, and peak memory (via tracemalloc) around each solver call so every method reports on the same basis.
def measure(fn, *args, **kwargs):
    tracemalloc.start()  # Start tracking memory
    t_wall0, t_cpu0 = time.perf_counter(), time.process_time() # Start wall clock time & Start CPU time
    result = fn(*args, **kwargs)
    t_cpu1, t_wall1 = time.process_time(), time.perf_counter() # End  wall clock time & End CPU time
    _, peak = tracemalloc.get_traced_memory() ## Get peak memory
    tracemalloc.stop()
    meta = {'wall_s': t_wall1 - t_wall0, 'cpu_s': t_cpu1 - t_cpu0, 'peak_kb': peak/1024}
    return result, meta
#=========================================================
# Direct solver (LU Factorization):
def solve_direct(A, b):
    lu = splu(A)
    return lu.solve(b)

T_direct, meta_direct = measure(solve_direct, A, b)
print(meta_direct)
#============================================================
#Jacobi solver
def solve_jacobi(A, b, N, TOL=1e-6, MAX_ITER=20000):
    """
    Solve linear system A*T = b using Jacobi iterative method.
    
    Parameters:
    -----------
    A : sparse matrix 
        Coefficient matrix from finite difference discretization
    b : numpy array
        Right-hand side vector (boundary conditions)
    N : int
        Number of unknowns
    TOL : float
        Convergence tolerance (maximum allowed change between iterations)
    MAX_ITER : int
        Maximum number of iterations allowed
    
    Returns:
    --------
    T : numpy array
        Solution vector (temperature field)
    it : int
        Number of iterations actually performed
    residuals : list
        History of maximum changes between iterations (for convergence analysis)
    """
    
    # Convert sparse matrix to compressed sparse row format for efficient matrix-vector multiplication
    A_csr = A.tocsr()
    
    # Extract diagonal elements and compute their inverses
    # D_inv = 1/diag(A) for Jacobi preconditioning
    D_inv = 1.0 / A_csr.diagonal()
    
    # Initialize solution
    # Using surface temperature estimate: T_inf + q0/h (from Robin BC at uniform flux)
    # This provides a better initial guess than all zeros,
    T = np.ones(N) * (T_inf + q0/h)
    
    # Store history of residuals for convergence monitoring
    residuals = []
    
    # Iteration counter (starts at 0, will be updated)
    it = 0
    
    # Main iteration loop
    for c in range(MAX_ITER):
        # Jacobi iteration: T_new = T - D^{-1} * (A*T - b)
        # 
        # Derivation: A*T = b
        #             (D + L + U)*T = b  (split into diagonal, lower, upper)
        #             D*T = b - (L+U)*T
        #             T_new = D^{-1} * [b - (L+U)*T]
        #        or,  T_new = T - D^{-1} * (A*T - b)
        #
        # A_csr @ T computes the residual: r = A*T - b
        # D_inv * r scales the residual by inverse diagonal
        # T - D_inv * r gives the new solution
        T_new = T - D_inv * (A_csr @ T - b)
        
        # Compute convergence metric: maximum absolute change in any node
        diff = np.max(np.abs(T_new - T))
        
        # Store residual history for plotting/convergence analysis
        residuals.append(diff)
        
        # Update iteration counter (1-based indexing for output)
        it = c + 1
        
        # Update solution for next iteration
        T = T_new
        
        # Check convergence: if maximum change is below tolerance, stop
        if diff < TOL:
            break
    
    # Return solution, iteration count, and residual history
    return T, it, residuals


# ============================================================================
# PERFORMANCE MEASUREMENT AND VALIDATION
# ============================================================================

# Run the Jacobi solver with performance measurement
# measure() wraps the function and returns: (result_tuple, performance_metadata)
(T_jac, iter_jac, res_jac), meta_jac = measure(
    solve_jacobi,           # Function to benchmark
    A, b, N,                # Positional arguments
    TOL, MAX_ITER           # Keyword arguments 
)

# Print results:
# 1. Number of iterations taken to converge
print(f"Iterations: {iter_jac}")

# 2. Performance metrics
#    - wall_s: Wall-clock time (real elapsed time)
#    - cpu_s: CPU time (actual computation time)
#    - peak_kb: Peak memory usage during execution
print(f"Performance: {meta_jac}")

# 3. Accuracy check against direct solver (assuming T_direct is available)
#    This validates the iterative solution against a direct method (sparse LU)
print(f"Max difference vs direct solver: {np.max(np.abs(T_jac - T_direct))}")

#==================================================================

#G-S Method

def solve_gauss_seidel(A, b, N, TOL=1e-6, MAX_ITER=20000):
    # Convert to CSR format for efficient row-based operations
    # CSR allows fast access to each row's non-zero elements
    A_csr = csr_matrix(A)
    
    # Extract diagonal elements
    # In Gauss-Seidel: T_new[i] = (b[i] - sum_{j≠i} A[i,j]*T[j]) / A[i,i]
    diag = A_csr.diagonal()
    
    # Extract CSR matrix components for fast iteration
    # indptr: start/end indices for each row in the data/indices arrays
    # indices: column indices of non-zero elements
    # data: actual matrix values
    indptr, indices, data = A_csr.indptr, A_csr.indices, A_csr.data
    
    # Initialize solution 
    T = np.ones(N) * (T_inf + q0/h)
    
    # Store history of residuals for convergence monitoring
    residuals = []
    
    # Iteration counter
    it = 0
    
    # Main iteration loop
    for c in range(MAX_ITER):
        # Store full old solution before updating
        # This is needed to compute the maximum change after all nodes are updated
        T_old_full = T.copy()
        
        # Loop over all unknowns (nodes)
        # This is a sequential update: each new value is immediately used
        for p in range(N):
            # Get the start and end indices for row p in the CSR arrays
            row_start, row_end = indptr[p], indptr[p+1]
            
            # Compute sum of A[p, j] * T[j] for all j ≠ p
            # This is the matrix-vector product but excluding the diagonal term
            s = 0.0
            for kk in range(row_start, row_end):
                col = indices[kk]  # Column index of this non-zero element
                if col != p:       # Skip the diagonal element
                    s += data[kk] * T[col]  # Add contribution to sum
            
            # Gauss-Seidel update formula:
            # T_new[p] = (b[p] - sum_{j≠p} A[p,j] * T[j]) / A[p,p]
            # in this iteration (for col < p) OR old values (for col > p)
            T[p] = (b[p] - s) / diag[p]
        
        # Compute convergence metric: maximum absolute change in any node
        diff = np.max(np.abs(T - T_old_full))
        
        # Store residual history for analysis
        residuals.append(diff)
        
        # Update iteration counter
        it = c + 1
        
        # Check convergence
        if diff < TOL:
            break
    
    return T, it, residuals


# ============================================================================
# PERFORMANCE MEASUREMENT AND VALIDATION
# ============================================================================

# Run the Gauss-Seidel solver with performance measurement
# measure() returns: (result_tuple, performance_metadata)
(T_gs, iter_gs, res_gs), meta_gs = measure(
    solve_gauss_seidel,     # Function to benchmark
    A, b, N,               # Positional arguments
    TOL, MAX_ITER          # Keyword arguments (positional here)
)

# Print results:
# 1. Number of iterations taken to converge
print(f"Gauss-Seidel iterations: {iter_gs}")

# 2. Performance metrics
#    - wall_s: Wall-clock time (real elapsed time)
#    - cpu_s: CPU time (actual computation time)  
#    - peak_kb: Peak memory usage during execution
print(f"Performance: {meta_gs}")

# 3. Accuracy check against direct solver
#    This validates the iterative solution against a direct method
print(f"Max difference vs direct solver: {np.max(np.abs(T_gs - T_direct))}")

# == ===========================================================
# SOR Method
def _sor_vectorized(Nr, Ntheta, omega, TOL=1e-6, MAX_ITER=20000):
    """
    Solve heat equation using SOR (Successive Over-Relaxation) with vectorized operations.

    """
    
    # ============================================================================
    # 1. PRE-COMPUTE CONSTANTS AND COEFFICIENTS (Vectorized)
    # ============================================================================
    
    # Grid spacings
    dr = R / Nr
    dtheta = 2*np.pi / Ntheta
    
    # Radial positions for interior nodes (i = 1 to Nr-1)
    # Shape: (Nr-1, 1) for broadcasting with angular dimension
    r_int = (np.arange(1, Nr) * dr)[:, None]  # Shape: (Nr-1, 1)
    
    # Radial discretization coefficients 
    # These come from: d²T/dr² + (1/r)*dT/dr
    c_out = 1.0/dr**2 + 1.0/(2*r_int*dr)   # Coefficient for T[i+1, j] (outward)
    c_in  = 1.0/dr**2 - 1.0/(2*r_int*dr)   # Coefficient for T[i-1, j] (inward)
    
    # Angular discretization coefficient: (1/r²) * d²T/dθ²
    c_ang = 1.0/(r_int**2 * dtheta**2)      # Coefficient for T[i, j±1]
    
    # Diagonal coefficient (central point)
    c_diag = -2.0/dr**2 - 2.0/(r_int**2*dtheta**2)
    
    # Angular positions for heat flux evaluation
    theta = np.arange(Ntheta) * dtheta
    q_theta = q_flux(theta)  # Spatially varying heat flux at surface
    
    # ============================================================================
    # 2. INITIALIZE SOLUTION
    # ============================================================================
    
    # Initial guess: surface temperature estimate 
    T = np.ones((Nr, Ntheta)) * (T_inf + q0/h)
    
    # ============================================================================
    # 3. SET UP RED-BLACK ORDERING
    # ============================================================================
    
    # Create meshgrid for all interior points (i = 0 to Nr-2, j = 0 to Ntheta-1)
    #We exclude the surface (i = Nr-1) for the interior sweep
    ii, jj = np.meshgrid(np.arange(Nr-1), np.arange(Ntheta), indexing='ij')
    
    # Red-black ordering: chessboard pattern
    # Points where (i+j) is even are "red", odd are "black"
    # This allows parallel updates within each color while maintaining Gauss-Seidel properties
    red_mask = ((ii+jj) % 2 == 0)    # Even sum -> red
    black_mask = ((ii+jj) % 2 != 0)  # Odd sum -> black
    
    # ============================================================================
    # 4. DEFINE SWEEP FUNCTION (One color update)
    # ============================================================================
    
    def sweep(mask, pole_avg):
        """
        Perform one SOR sweep on points of a given color (red or black).
        
        Parameters:
        -----------
        mask : 2D boolean array
            Mask for points to update (red_mask or black_mask)
        pole_avg : float
            Average temperature at the pole (for regularity condition)
        """
        
        # Get interior points (all except surface)
        T_int = T[:Nr-1]  # Shape: (Nr-1, Ntheta)
        
        # Radial neighbors (using roll for vectorized shifting) 
        
        # Outward neighbor: T[i+1, j]
        # Roll up (-1 axis) to shift indices: T_int[k] -> T_int[k+1]
        Tp1 = np.roll(T_int, -1, axis=0)
        # At the last interior point (i = Nr-2), the outward neighbor is the surface
        # So we need to set Tp1[-1] = T[Nr-1] (surface temperature)
        Tp1[-1] = T[Nr-1]
        
        # Inward neighbor: T[i-1, j]
        # Roll down (+1 axis) to shift indices: T_int[k] -> T_int[k-1]
        Tm1 = np.roll(T_int, 1, axis=0)
        # At the first interior point (i = 0), the inward neighbor is the pole
        # Pole regularity: T at r=0 is the average of the first ring
        Tm1[0] = pole_avg  # T[0, j] = pole_avg (axisymmetric at pole)
        
        # Angular neighbors
        
        # Angular +1: T[i, j+1] with periodic wrap-around
        Tang_p = np.roll(T_int, -1, axis=1)  # Roll left
        # Angular -1: T[i, j-1] with periodic wrap-around
        Tang_m = np.roll(T_int, 1, axis=1)   # Roll right
        
        #Compute Gauss-Seidel update (vectorized for all points in mask) 
        
        # Right-hand side: sum of all neighbor contributions
        # This is: - (A_off_diagonal * T_neighbors)
        rhs = c_out*Tp1 + c_in*Tm1 + c_ang*Tang_p + c_ang*Tang_m
        
        # Gauss-Seidel update: T_new = -rhs / c_diag
        # Because: c_diag * T + rhs = 0  =>  T = -rhs / c_diag
        T_gs = -rhs / c_diag
        
        # --- Apply SOR relaxation ---
        # SOR: T_new = (1-omega)*T_old + omega*T_gs
        # omega = 1: Gauss-Seidel
        # 1 < omega < 2: Over-relaxation (faster convergence)
        # 0 < omega < 1: Under-relaxation (for difficult problems)
        T_int[mask] = (1-omega)*T_int[mask] + omega*T_gs[mask]
    
    # ============================================================================
    # 5. MAIN ITERATION LOOP
    # ============================================================================
    
    residuals = []
    it = 0
    
    for c in range(MAX_ITER):
        # Store old solution for convergence check
        T_old = T.copy()
        
        # Compute average temperature at the pole (first ring)
        # This is needed for the pole regularity condition
        pole_avg = np.mean(T[0])  # Average over angular direction at i=0
        
        # Perform red-black sweeps
        # Red then black ordering is common
        sweep(red_mask, pole_avg)    # Update all red points
        sweep(black_mask, pole_avg)  # Update all black points
        
        # ======================================================================
        # 6. APPLY ROBIN BOUNDARY CONDITION AT SURFACE (i = Nr-1)
        # ======================================================================
        
        # Robin BC: (k/dr + h)*T[Nr-1] - (k/dr)*T[Nr-2] = h*T_inf + q_theta
        # Solve for T[Nr-1]:
        T[Nr-1] = (h*T_inf + q_theta + (k/dr)*T[Nr-2]) / (k/dr + h)
        
        # Check convergence: maximum absolute change
        diff = np.max(np.abs(T - T_old))
        residuals.append(diff)
        it = c + 1
        
        if diff < TOL:
            break
    
    # Return flattened solution 
    return T.flatten(), it, residuals


def solve_sor(Nr, Ntheta, omega=1.7, TOL=1e-6, MAX_ITER=20000):
    """
    Wrapper function for SOR solver with standard interface.
    """
    return _sor_vectorized(Nr, Ntheta, omega, TOL, MAX_ITER)


# ============================================================================
# PERFORMANCE MEASUREMENT AND VALIDATION
# ============================================================================

# Run SOR solver with performance measurement
# omega = 1.7 is typically optimal for 2D Poisson problems
(T_sor, iter_sor, res_sor), meta_sor = measure(
    solve_sor,              # Function to benchmark
    Nr, Ntheta,             # Positional arguments
    1.7,                    # omega (relaxation parameter)
    TOL,                    # Convergence tolerance
    MAX_ITER                # Maximum iterations
)

# Print results
print(f"SOR (omega=1.7) iterations: {iter_sor}")
print(f"Performance: {meta_sor}")
print(f"Max difference vs direct solver: {np.max(np.abs(T_sor - T_direct))}")

# ============================================================================
# OPTIMAL OMEGA SWEEP FOR SOR (Successive Over-Relaxation)
# ============================================================================

# Generate a range of relaxation parameters to test
# SOR works for: 0 < omega < 2
# Over-relaxation: 1 < omega < 2 (faster convergence)
omegas = np.linspace(1.0, 1.95, 20)  # 20 points from 1.0 to 1.95

# Store number of iterations for each omega value
iters_sweep = []

print("Testing SOR relaxation parameters...")
print("-" * 50)

# Loop through each omega value and test convergence
for om in omegas:
    # Run SOR solver with current omega
    # Returns: (T_solution, iterations_taken, residuals_history)
    _, it_, _ = solve_sor(Nr, Ntheta, om, TOL=1e-6, MAX_ITER=3000)
    
    # Store iteration count
    iters_sweep.append(it_)
    
    # Print progress (optional)
    print(f"ω = {om:.3f}: {it_:4d} iterations")

# Find the omega that gave the minimum iterations
# np.argmin() returns the index of the minimum value
min_iter_index = np.argmin(iters_sweep)
omega_opt = omegas[min_iter_index]
min_iters = min(iters_sweep)

print("-" * 50)
print(f"✓ Optimal omega = {omega_opt:.3f} (only {min_iters} iterations!)")

# ============================================================================
# SURFACE TEMPERATURE VALIDATION (Task C)
# ============================================================================

# Define target angles for comparison (in degrees)
theta_targets_deg = [0, 60, 120, 180, 270]

# ============================================================================
# 1. EXTRACT NUMERICAL SOLUTION AT THE SURFACE
# ============================================================================

# The flat solution array is ordered as:
# [T(1,0), T(1,1), ..., T(1,Ntheta-1),    <- r = dr (first interior)
#  T(2,0), T(2,1), ..., T(2,Ntheta-1),    <- r = 2dr
#  ...
#  T(Nr,0), T(Nr,1), ..., T(Nr,Ntheta-1)] <- r = R (surface)

# Extract the last Ntheta entries = surface ring (r = R, i = Nr)
# -Ntheta means "start Ntheta elements from the end"
T_surface_num = T_direct[-Ntheta:] 

# ============================================================================
# 2. COMPARE AT SPECIFIED ANGULAR POSITIONS
# ============================================================================

rows_c = []  # Store comparison results

for th_deg in theta_targets_deg:
    # Convert degrees to radians for calculation
    theta = np.deg2rad(th_deg)
    
    # Find the nearest angular grid point
    # j = round(theta / dtheta) gives the index closest to the target angle
    # % Ntheta ensures wrap-around (safety, though not needed here)
    j = int(round(theta / dtheta)) % Ntheta
    
    # Get numerical temperature at this angular position
    T_num = T_surface_num[j]
    
    # Get analytical solution at this angle
    # analytical_surface() should implement the exact solution from Task B
    T_ana = analytical_surface(theta)
    
    # Store results
    rows_c.append([
        th_deg,           # Target angle in degrees
        T_ana,            # Analytical solution
        T_num,            # Numerical solution
        abs(T_ana - T_num) # Absolute error
    ])

# ============================================================================
# 3. CREATE AND DISPLAY RESULTS TABLE
# ============================================================================

# Create pandas DataFrame for nice tabular display
task_c_df = pd.DataFrame(rows_c, columns=[
    'theta (deg)',
    'Analytical T(R,theta) (C)',
    'Numerical T(R,theta) (C)',
    'Abs error (C)'
])

# Display with formatted numbers (6 decimal places, scientific notation for errors)
display(task_c_df.style.format({
    'Analytical T(R,theta) (C)': '{:.6f}',
    'Numerical T(R,theta) (C)': '{:.6f}',
    'Abs error (C)': '{:.3e}'  # Scientific notation for small errors
}))

# ============================================================================
# 4. REPORT MAXIMUM ERROR
# ============================================================================

max_err_c = task_c_df['Abs error (C)'].max()
print(f"\nMaximum error at the 5 requested angles: {max_err_c:.3e} °C")


# ============================================================================


# ENERGY CONSERVATION VERIFICATION (Task E)
# Verify that heat in = heat out for steady-state solution
# ============================================================================

# ============================================================================
# 1. ANALYTICAL ENERGY BALANCE
# ============================================================================

# Heat input from applied flux:
# q_in = ∫ q''(theta) * R * dtheta ( integration limits theta -- 2pi)
#
# q''(theta) = q0 + q1*cos(theta) + q2*sin(theta) + q3*cos(3theta)
# Only the constant term q0 survives integration over a full period
# integral [ cos(ntheta) dtheta] = 0 for n = 1, 2, 3, ... ( integration limits theta -- 2pi)
# integral [sin(ntheta) dtheta] = 0 for n = 1, 2, 3, ... ( integration limits theta -- 2pi)

q_in_ana = 2 * np.pi * R * q0
print(f"Analytical heat in: {q_in_ana:.6f} W/m")

# Heat out via convection: q_out = ∫₀²π h*(T(R,theta) - T_inf) * R * dtheta
# The analytical surface temperature is:
# T(R,theta) = A0 + A1*cos(theta) + B1*sin(theta) + A3*cos(3theta)
# T(R,theta) - T_inf = (q0/h) + A1*cos(theta) + B1*sin(theta) + A3*cos(3theta)
#
# Again, only the constant term survives integration:
# integral [ (q0/h) * R * h * dtheta] = 2πR*q0 ( integration limits theta -- 2pi)

q_out_ana = 2 * np.pi * R * h * (q0 / h) 
print(f"Analytical heat out: {q_out_ana:.6f} W/m")

# Net heat should be exactly zero (steady-state)
net_ana = q_in_ana - q_out_ana
print(f"Analytical net: {net_ana:.2e} W/m")

# ============================================================================
# 2. NUMERICAL ENERGY BALANCE 
# ============================================================================

# Discretize the angular domain into Ntheta nodes
theta_nodes = np.arange(Ntheta) * dtheta  # θ = 0, Δθ, 2Δθ, ..., (Ntheta-1)Δθ

# ----------------------------------------------------------------------------
# 2a. Numerical heat input
# ----------------------------------------------------------------------------

# Heat in: q_in = ∫π q''(θ) * R * dθ
# q_in ≈ Σ [q''(θ_j) * R * Δθ]

q_in_num = np.sum(q_flux(theta_nodes) * R * dtheta)

print(f"\nNumerical heat in: {q_in_num:.6f} W/m")

# ----------------------------------------------------------------------------
# 2b. Numerical heat output
# ----------------------------------------------------------------------------

# Heat out: q_out = ∫₀²π h*(T(R,θ) - T_inf) * R * dθ
# Using periodic trapezodial:
# q_out ≈ Σ [h*(T_surface_num[j] - T_inf) * R * Δθ]

q_out_num = np.sum(h * (T_surface_num - T_inf) * R * dtheta)

print(f"Numerical heat out: {q_out_num:.6f} W/m")

# ----------------------------------------------------------------------------
# 2c. Numerical net heat
# ----------------------------------------------------------------------------

net_num = q_in_num - q_out_num
print(f"Numerical net: {net_num:.6e} W/m")

# ============================================================================
# 3. CONVERGENCE CHECK
# ============================================================================

print("\n" + "=" * 60)
print("ENERGY CONSERVATION VERIFICATION")
print("=" * 60)

if abs(net_num) < 1e-6:
    print("✓ Energy conservation confirmed (net heat ~ 0).")
    print(f"  Net heat: {net_num:.2e} W/m (within tolerance)")
elif abs(net_num) < 1e-3:
    print("Energy conservation acceptable (net heat small).")
    print(f"  Net heat: {net_num:.2e} W/m")
else:
    print("ENERGY CONSERVATION FAILED!")
    print(f"  Net heat: {net_num:.2e} W/m (too large)")
    print("  Check:")
    print("    - Grid resolution (increase Nr, Ntheta)")
    print("    - Boundary condition discretization")
    print("    - Numerical integration method")
    
#==========================================
# Plots and Reports
#==========================================

def extract_solution(T_flat, Nr, Ntheta):
   
# Convert the flat numerical solution into a polar field.
# Row 0 corresponds to r = 0 (the pole).
#Rows 1...Nr correspond to the actual numerical rings.
#The pole temperature is reconstructed from the first ring.


    field = np.zeros((Nr + 1, Ntheta))

    # Numerical solution:
    # flat index corresponds to i = 1...Nr
    field[1:, :] = T_flat.reshape((Nr, Ntheta))

    # At r = 0 the temperature is single-valued.
    # Use the average of the first ring.
    field[0, :] = np.mean(field[1, :])

    return field

fields = {
    "Direct": extract_solution(T_direct, Nr, Ntheta),
    "Jacobi": extract_solution(T_jac, Nr, Ntheta),
    "GS": extract_solution(T_gs, Nr, Ntheta),
    "SOR": extract_solution(T_sor, Nr, Ntheta)
}


results = {
    "A": A,
    "b": b,

    "direct": {
        "T": T_direct,
        "time": meta_direct["wall_s"],
        "iterations": 0,
        "res": []
    },

    "jacobi": {
        "T": T_jac,
        "time": meta_jac["wall_s"],
        "iterations": iter_jac,
        "res": res_jac
    },

    "gs": {
        "T": T_gs,
        "time": meta_gs["wall_s"],
        "iterations": iter_gs,
        "res": res_gs
    },

    "sor": {
        "T": T_sor,
        "time": meta_sor["wall_s"],
        "iterations": iter_sor,
        "res": res_sor,
        "omega": 1.7
    }
}

# 1. SELECTED NODAL TEMPERATURE TABLE

def nodal_temperature_table(fields):

    dr = R / Nr
    dtheta = 2*np.pi / Ntheta

    radii = [0.25*R, 0.50*R, 0.75*R]
    angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]

    rows = []

    for r_target in radii:

        i = int(round(r_target / dr))
        i = min(max(i, 0), Nr)

        r_actual = i * dr

        for theta_deg in angles_deg:

            theta = np.deg2rad(theta_deg)

            j = int(round(theta / dtheta)) % Ntheta

            T_ana = analytical_solution(r_actual, theta)

            rows.append([
                r_actual,
                r_actual / R,
                theta_deg,
                T_ana,
                fields["Direct"][i, j],
                fields["Jacobi"][i, j],
                fields["GS"][i, j],
                fields["SOR"][i, j]
            ])

    df = pd.DataFrame(
        rows,
        columns=[
            "r (m)",
            "r/R",
            "theta (deg)",
            "Analytical (°C)",
            "Direct (°C)",
            "Jacobi (°C)",
            "GS (°C)",
            "SOR (°C)"
        ]
    )
    
    print("SELECTED NODAL TEMPERATURES")

    display(
        df.style.format({
            "r (m)": "{:.5f}",
            "r/R": "{:.2f}",
            "theta (deg)": "{:.0f}",
            "Analytical (°C)": "{:.6f}",
            "Direct (°C)": "{:.6f}",
            "Jacobi (°C)": "{:.6f}",
            "GS (°C)": "{:.6f}",
            "SOR (°C)": "{:.6f}"
        })
    )

    return df

# 2. METHOD PERFORMANCE TABLE
def method_performance_table(results):

    T_direct = results["direct"]["T"]

    rows = []

    for name, key in [
        ("Direct LU", "direct"),
        ("Jacobi", "jacobi"),
        ("GS", "gs"),
        ("SOR", "sor")
    ]:

        item = results[key]
        T = item["T"]

        # Relative residual
        residual = (
            np.linalg.norm(results["A"] @ T - results["b"], ord=np.inf)
            / max(np.linalg.norm(results["b"], ord=np.inf), 1e-30)
        )

        diff_direct = np.max(np.abs(T - T_direct))

        if key == "direct":
            iterations = "-"
        else:
            iterations = item["iterations"]

        rows.append([
            name,
            iterations,
            item["time"],
            residual,
            diff_direct
        ])

    df = pd.DataFrame(
        rows,
        columns=[
            "Method",
            "Iterations",
            "Time (s)",
            "Relative residual",
            "Max |T - T_direct| (°C)"
        ]
    )

    print("METHOD PERFORMANCE COMPARISON")

    display(
        df.style.format({
            "Time (s)": "{:.6f}",
            "Relative residual": "{:.3e}",
            "Max |T - T_direct| (°C)": "{:.3e}"
        })
    )

    print(f"\nSOR relaxation factor: omega = {results['sor']['omega']:.3f}")

    return df


# ============================================================
# 3. SURFACE TEMPERATURE COMPARISON
# ============================================================

def plot_surface_comparison(fields):

    theta_fine = np.linspace(0, 2*np.pi, 1000)
    theta_num = np.arange(Ntheta) * (2*np.pi/Ntheta)

    plt.figure(figsize=(9, 6))

    plt.plot(
        np.rad2deg(theta_fine),
        analytical_surface(theta_fine),
        label="Analytical",
        linewidth=2
    )

    markers = {
        "Direct": "o",
        "Jacobi": "s",
        "GS": "^",
        "SOR": "d"
    }

    for name, marker in markers.items():

        plt.plot(
            np.rad2deg(theta_num),
            fields[name][Nr, :],
            marker=marker,
            linestyle="--",
            markersize=4,
            linewidth=1,
            label=name
        )

    plt.xlabel(r"$\theta$ (degrees)")
    plt.ylabel("Surface temperature (°C)")
    plt.title("Surface Temperature: Numerical Methods vs Analytical Solution")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "surface_temperature_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

# 4. RADIAL TEMPERATURE COMPARISON

def plot_radial_comparison(fields):

    theta_list = [0, 90, 180, 270]

    dr = R / Nr

    r_num = np.arange(Nr + 1) * dr
    r_R = r_num / R

    for theta_deg in theta_list:

        theta = np.deg2rad(theta_deg)

        j = int(round(theta / (2*np.pi/Ntheta))) % Ntheta

        # Analytical radial solution
        A0 = q0 / h
        A1 = q1 / (h + k/R)
        B1 = q2 / (h + k/R)
        A3 = q3 / (h + 3*k/R)

        T_ana = (
            T_inf
            + A0
            + r_R * (
                A1*np.cos(theta)
                + B1*np.sin(theta)
            )
            + r_R**3 * A3*np.cos(3*theta)
        )

        plt.figure(figsize=(8, 5.5))

        plt.plot(
            r_num,
            T_ana,
            label="Analytical",
            linewidth=2
        )

        for name, marker in [
            ("Direct", "o"),
            ("Jacobi", "s"),
            ("GS", "^"),
            ("SOR", "d")
        ]:

            plt.plot(
                r_num,
                fields[name][:, j],
                marker=marker,
                linestyle="--",
                markersize=3.5,
                linewidth=1,
                label=name
            )

        plt.xlabel("r (m)")
        plt.ylabel("Temperature (°C)")
        plt.title(
            f"Radial Temperature Comparison at "
            f"theta = {theta_deg}°"
        )

        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            f"radial_temperature_theta_{theta_deg}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

# 5. ITERATIVE CONVERGENCE HISTORY

def plot_iterative_convergence(results):

    plt.figure(figsize=(8, 6))

    plt.semilogy(
        np.arange(1, len(results["jacobi"]["res"]) + 1),
        results["jacobi"]["res"],
        label="Jacobi",
        linewidth=2
    )

    plt.semilogy(
        np.arange(1, len(results["gs"]["res"]) + 1),
        results["gs"]["res"],
        label="GS",
        linewidth=2
    )

    plt.semilogy(
        np.arange(1, len(results["sor"]["res"]) + 1),
        results["sor"]["res"],
        label=f"SOR (omega={results['sor']['omega']:.3f})",
        linewidth=2
    )

    plt.xlabel("Iteration")
    plt.ylabel(
        r"$\max|T^{(m+1)}-T^{(m)}|$ (°C)"
    )

    plt.title("Iterative Convergence History")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "iterative_convergence.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# 6. GRID REFINEMENT STUDY

def grid_refinement_report():

    grids = [
        (10, 20, "Coarse"),
        (20, 40, "Medium"),
        (40, 80, "Fine"),
        (80, 120, "Very Fine")
    ]

    radii = [0.25*R, 0.50*R, 0.75*R]
    angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]

    rows = []
    errors = []
    drs = []

    for Nr_g, Nt_g, label in grids:

        A_g, b_g, N_g, counts_g = build_system(Nr_g, Nt_g)

        start = time.perf_counter()

        T_g = solve_direct(A_g, b_g)

        elapsed = time.perf_counter() - start

        dr_g = R / Nr_g
        dtheta_g = 2*np.pi / Nt_g

        field_g = extract_solution(T_g, Nr_g, Nt_g)

        validation_errors = []

        for r_target in radii:

            i = int(round(r_target / dr_g))
            i = min(max(i, 0), Nr_g)

            r_actual = i * dr_g

            for theta_deg in angles_deg:

                theta = np.deg2rad(theta_deg)

                j = int(round(theta / dtheta_g)) % Nt_g

                T_ana = analytical_solution(
                    r_actual,
                    theta
                )

                T_num = field_g[i, j]

                validation_errors.append(
                    abs(T_num - T_ana)
                )

        max_err = max(validation_errors)

        rows.append([
            label,
            Nr_g,
            Nt_g,
            N_g,
            dr_g,
            max_err,
            elapsed
        ])

        errors.append(max_err)
        drs.append(dr_g)

    df = pd.DataFrame(
        rows,
        columns=[
            "Grid",
            "Nr",
            "Ntheta",
            "Unknowns",
            "dr (m)",
            "Maximum absolute error (°C)",
            "Direct time (s)"
        ]
    )

    print("GRID REFINEMENT STUDY")

    display(
        df.style.format({
            "dr (m)": "{:.6e}",
            "Maximum absolute error (°C)": "{:.3e}",
            "Direct time (s)": "{:.6f}"
        })
    )


# 7. POLAR MESH

def plot_mesh(Nr, Ntheta, R):

    dr = R / Nr
    dtheta = 2*np.pi / Ntheta

    plt.figure(figsize=(7, 7))

    # Radial lines
    r = np.linspace(0, R, Nr + 1)

    for j in range(Ntheta):

        theta = j * dtheta

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        plt.plot(x, y, 'k', linewidth=0.7)

    # Circular lines
    theta_circle = np.linspace(0, 2*np.pi, 300)

    for i in range(1, Nr + 1):

        x = i * dr * np.cos(theta_circle)
        y = i * dr * np.sin(theta_circle)

        plt.plot(x, y, 'k', linewidth=0.7)

    plt.axis("equal")

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    plt.title(
        f"Polar Mesh: Nr = {Nr}, Ntheta = {Ntheta}"
    )

    plt.grid(False)
    plt.tight_layout()

    plt.savefig(
        "polar_mesh.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# 8. TEMPERATURE FIELD

def plot_grid_and_temperature(fields):

    T = fields["Direct"]

    dr = R / Nr
    dtheta = 2*np.pi / Ntheta

    r = np.arange(Nr + 1) * dr
    theta = np.arange(Ntheta) * dtheta

    # Add theta = 2*pi for plotting only
    theta_plot = np.append(theta, 2*np.pi)

    # Repeat theta = 0 temperature at theta = 2*pi
    T_plot = np.column_stack((T, T[:, 0]))

    Rg, Tg = np.meshgrid(
        r,
        theta_plot,
        indexing="ij"
    )

    X = Rg * np.cos(Tg)
    Y = Rg * np.sin(Tg)

    plt.figure(figsize=(7, 7))

    plt.contourf(
        X,
        Y,
        T_plot,
        levels=30,
        cmap = 'viridis'
    )

    plt.colorbar(
        label="Temperature (°C)"
    )

    plt.plot(
        X.ravel(),
        Y.ravel(),
        ".",
        markersize=1,
        alpha=0.25
    )

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    plt.title(
        "Computational Grid and Temperature Field"
    )

    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(
        "grid_temperature_field.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
# 9. HOTTEST / COLDEST POINTS

def plot_hot_cold_points(fields):

    theta_fine = np.linspace(0, 2*np.pi, 2000)
    T_ana_fine = analytical_surface(theta_fine)

    i_hot_ana = np.argmax(T_ana_fine)
    i_cold_ana = np.argmin(T_ana_fine)

    theta_hot_ana = theta_fine[i_hot_ana]
    theta_cold_ana = theta_fine[i_cold_ana]

    theta_num = np.arange(Ntheta) * (2*np.pi/Ntheta)
    T_surface_num = fields["Direct"][Nr, :]

    i_hot_num = np.argmax(T_surface_num)
    i_cold_num = np.argmin(T_surface_num)

    theta_hot_num = theta_num[i_hot_num]
    theta_cold_num = theta_num[i_cold_num]

    plt.figure(figsize=(9, 6))

    plt.plot(np.rad2deg(theta_fine), T_ana_fine, label="Analytical", linewidth=2)
    plt.plot(np.rad2deg(theta_num), T_surface_num, "o--", markersize=4,
             linewidth=1, label="Direct (numerical)")

    plt.plot(np.rad2deg(theta_hot_ana), T_ana_fine[i_hot_ana], "r^",
             markersize=12, label="Hottest (analytical)")
    plt.plot(np.rad2deg(theta_cold_ana), T_ana_fine[i_cold_ana], "bv",
             markersize=12, label="Coldest (analytical)")
    plt.plot(np.rad2deg(theta_hot_num), T_surface_num[i_hot_num], "r^",
             markersize=9, markeredgecolor="k", label="Hottest (numerical)")
    plt.plot(np.rad2deg(theta_cold_num), T_surface_num[i_cold_num], "bv",
             markersize=9, markeredgecolor="k", label="Coldest (numerical)")

    plt.annotate(f"{np.rad2deg(theta_hot_ana):.1f}°, {T_ana_fine[i_hot_ana]:.2f}°C",
                 (np.rad2deg(theta_hot_ana), T_ana_fine[i_hot_ana]),
                 textcoords="offset points", xytext=(8, 8))
    plt.annotate(f"{np.rad2deg(theta_cold_ana):.1f}°, {T_ana_fine[i_cold_ana]:.2f}°C",
                 (np.rad2deg(theta_cold_ana), T_ana_fine[i_cold_ana]),
                 textcoords="offset points", xytext=(8, -14))

    plt.xlabel(r"$\theta$ (degrees)")
    plt.ylabel("Surface temperature (°C)")
    plt.title("Hottest / Coldest Surface Points: Analytical vs Numerical")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("hot_cold_points.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("HOTTEST / COLDEST POINTS")
    print(f"  Analytical hottest: theta = {np.rad2deg(theta_hot_ana):7.2f} deg,  T = {T_ana_fine[i_hot_ana]:.4f} C")
    print(f"  Analytical coldest: theta = {np.rad2deg(theta_cold_ana):7.2f} deg,  T = {T_ana_fine[i_cold_ana]:.4f} C")
    print(f"  Numerical  hottest: theta = {np.rad2deg(theta_hot_num):7.2f} deg,  T = {T_surface_num[i_hot_num]:.4f} C")
    print(f"  Numerical  coldest: theta = {np.rad2deg(theta_cold_num):7.2f} deg,  T = {T_surface_num[i_cold_num]:.4f} C")

    return {
        "theta_hot_ana": theta_hot_ana, "T_hot_ana": T_ana_fine[i_hot_ana],
        "theta_cold_ana": theta_cold_ana, "T_cold_ana": T_ana_fine[i_cold_ana],
        "theta_hot_num": theta_hot_num, "T_hot_num": T_surface_num[i_hot_num],
        "theta_cold_num": theta_cold_num, "T_cold_num": T_surface_num[i_cold_num],
    }

# MAIN REPORT

if __name__ == "__main__":


    nodal_df = nodal_temperature_table(fields)

    performance_df = method_performance_table(results)

    plot_mesh(Nr, Ntheta, R)

    plot_grid_and_temperature(fields)

    plot_hot_cold_points(fields)

    plot_surface_comparison(fields)

    plot_radial_comparison(fields)

    plot_iterative_convergence(results)

    grid_df = grid_refinement_report()

fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4.8))

# ------------------------------------------------------------
# Convergence history
# ------------------------------------------------------------
for name, res in [('Jacobi', res_jac),
                  ('GS', res_gs),
                  ('SOR', res_sor)]:
    axes2[0].semilogy(res, label=name, linewidth=1.6)

axes2[0].set_xlabel('Iteration')
axes2[0].set_ylabel('max|T_new - T_old| (C)')
axes2[0].set_title('Convergence history')
axes2[0].grid(alpha=0.3)
axes2[0].legend()


# ------------------------------------------------------------
# SOR: iterations vs omega
# ------------------------------------------------------------
axes2[1].plot(omegas, iters_sweep, 'o-', color='purple', markersize=4)
axes2[1].axvline(omega_opt, color='k', linestyle='--',
                 linewidth=1,
                 label=f'optimal omega={omega_opt:.3f}')

axes2[1].set_xlabel('omega')
axes2[1].set_ylabel('Iterations (TOL=1e-6)')
axes2[1].set_title('SOR: iterations vs omega')
axes2[1].grid(alpha=0.3)
axes2[1].legend()


# ------------------------------------------------------------
# Solve time by method
# ------------------------------------------------------------
methods = ['Direct', 'Jacobi', 'GS', 'SOR']

times = [meta_direct['wall_s'],
         meta_jac['wall_s'],
         meta_gs['wall_s'],
         meta_sor['wall_s']]

axes2[2].bar(methods, times,
             color=['#8172B2', '#4C72B0', '#C44E52', '#CCB974'])

axes2[2].set_yscale('log')
axes2[2].set_ylabel('Wall time (s)')
axes2[2].set_title('Solve time by method')
axes2[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

#======================================================================


