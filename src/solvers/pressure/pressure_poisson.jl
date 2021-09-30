"""
Solve the Poisson equation for the pressure.
compute pressure from pressure poisson_solver problem with right-hand side f
assume the Laplace matrix is known (A) and is possibly factorized (LU);
right hand side is given by f
we should have sum(f) = 0 for periodic and no-slip BC
"""
function pressure_poisson(f, t, setup, tol = 1e-14)
    @unpack A_fact = setup.discretization
        # Use pre-determined decomposition
        Δp = A_fact \ f
end
