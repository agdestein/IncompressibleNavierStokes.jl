function pressure_poisson(pressure_solver, f, t, setup)
    @unpack Np = setup.grid
    p = zeros(Np)
    pressure_poisson!(pressure_solver, p, f, t, setup)
end

"""
    pressure_poisson!(solver, p, f, t, setup, tol = 1e-14)

Solve the Poisson equation for the pressure with right hand side `f` at time `t`.

We should have `sum(f) = 0` for periodic and no-slip BC.
"""
function pressure_poisson! end

function pressure_poisson!(::DirectPressureSolver, p, f, t, setup, tol = 1e-14)
    # Assume the Laplace matrix is known (A) and is possibly factorized
    @unpack A_fact = setup.discretization

    # Use pre-determined decomposition
    p .= A_fact \ f
end

function pressure_poisson!(::CGPressureSolver, p, f, t, setup)
    @unpack A = setup.discretization
    # TODO: Pass `abstol`, `reltol` and `maxiter`
    # TODO: Preconditioner
    cg!(p, A, f)
end

function pressure_poisson!(::FourierPressureSolver, p, f, t, setup)
    @unpack Npx, Npy = setup.grid
    @unpack Â = setup.solver_settings

    # Fourier transform of right hand side
    f̂ = fft(reshape(f, Npx, Npy));

    # Solve for coefficients in Fourier space
    p̂ = -f̂ ./ Â;

    # Transform back
    p .= real.(reshape(ifft(p̂), length(p)))
end
