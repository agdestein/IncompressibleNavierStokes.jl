"""
    pressure_poisson(solver, f, setup)

Convenience function for initializing the pressure vector `p` before
calling `pressure_poisson!`.
"""
function pressure_poisson end

function pressure_poisson(solver, f, setup)
    p = zeros(setup.grid.Np)
    pressure_poisson!(solver, p, f)
end

function pressure_poisson(solver::DirectPressureSolver, f)
    # Assume the Laplace matrix is known (A) and is possibly factorized

    # Use pre-determined decomposition
    solver.A_fact \ f
end

function pressure_poisson(solver::CGPressureSolver, f)
    (; A, abstol, reltol, maxiter) = solver
    cg(A, f; abstol, reltol, maxiter)
end

function pressure_poisson(solver::FourierPressureSolver, f)
    (; Ahat) = solver

    f = reshape(f, size(Ahat))

    # Fourier transform of right hand side
    fhat = fft(f)

    # Solve for coefficients in Fourier space
    phat = @. -fhat / Ahat

    # Transform back
    p = ifft(phat)

    reshape(real.(p), :)
end

"""
    pressure_poisson!(solver, p, f)

Solve the Poisson equation for the pressure with right hand side `f` at time `t`.
For periodic and no-slip BC, the sum of `f` should be zero.
"""
function pressure_poisson! end

function pressure_poisson!(solver::DirectPressureSolver, p, f)
    # Assume the Laplace matrix is known (A) and is possibly factorized

    # Use pre-determined decomposition
    p .= solver.A_fact \ f
end

function pressure_poisson!(solver::CGPressureSolver, p, f)
    # TODO: Preconditioner
    (; A, abstol, reltol, maxiter) = solver
    cg!(p, A, f; abstol, reltol, maxiter)
end

function pressure_poisson!(solver::FourierPressureSolver, p, f)
    (; Ahat, fhat, phat) = solver

    fhat[:] = f

    # Fourier transform of right hand side
    fft!(fhat)

    # Solve for coefficients in Fourier space
    @. phat = -fhat / Ahat

    # Transform back
    ifft!(phat)
    @. p[:] = real(@view phat[:])

    p
end
