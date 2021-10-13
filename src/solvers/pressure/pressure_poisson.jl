"""
    pressure_poisson(solver, f, t, setup, tol = 1e-14)

Solve the Poisson equation for the pressure with right hand side `f` at time `t`.

We should have `sum(f) = 0` for periodic and no-slip BC.
"""
function pressure_poisson end

function pressure_poisson(::DirectPressureSolver, f, t, setup, tol = 1e-14)
    # Assume the Laplace matrix is known (A) and is possibly factorized
    @unpack A_fact = setup.discretization

    # Use pre-determined decomposition
    Δp = A_fact \ f
end
