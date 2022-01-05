"""
    pressure_poisson(solver, f, t, setup)

Convenience function for initializing the pressure vector `p` before
calling `pressure_poisson!`.
"""
function pressure_poisson(solver, f, t, setup)
    (; Np) = setup.grid
    p = zeros(Np)
    pressure_poisson!(solver, p, f, t, setup)
end

"""
    pressure_poisson!(solver, p, f, t, setup, tol = 1e-14)

Solve the Poisson equation for the pressure with right hand side `f` at time `t`.
For periodic and no-slip BC, the sum of `f` should be zero. 
"""
function pressure_poisson! end

function pressure_poisson!(solver::DirectPressureSolver, p, f, t, setup, tol = 1e-14)
    # Assume the Laplace matrix is known (A) and is possibly factorized
    
    # Use pre-determined decomposition
    p .= solver.A_fact \ f
end

function pressure_poisson!(solver::CGPressureSolver, p, f, t, setup)
    # TODO: Preconditioner
    (; A) = setup.discretization
    (; abstol, reltol, maxiter) = solver
    cg!(p, A, f; abstol, reltol, maxiter)
end

function pressure_poisson!(solver::FourierPressureSolver, p, f, t, setup)
    (; Npx, Npy) = setup.grid
    (; Â, f̂, p̂) = solver

    f̂[:] = f

    # Fourier transform of right hand side
    fft!(f̂);

    # Solve for coefficients in Fourier space
    @. p̂ = -f̂ / Â;
    
    # Transform back
    ifft!(p̂) 
    @. p[:] = real(@view p̂[:])

    p
end
