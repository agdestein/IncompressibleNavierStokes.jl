"""
    pressure_poisson(solver, f)

Solve the Poisson equation for the pressure with right hand side `f` at time `t`.
For periodic and no-slip BC, the sum of `f` should be zero.

Non-mutating/allocating/out-of-place version.

See also [`pressure_poisson!`](@ref).
"""
function pressure_poisson end

pressure_poisson(solver, f) = pressure_poisson!(
    solver,
    KernelAbstractions.zeros(get_backend(f), typeof(solver.setup.Re), solver.setup.grid.N),
    f,
)

"""
    pressure_poisson!(solver, p, f)

Solve the Poisson equation for the pressure with right hand side `f` at time `t`.
For periodic and no-slip BC, the sum of `f` should be zero.

Mutating/non-allocating/in-place version.

See also [`pressure_poisson`](@ref).
"""
function pressure_poisson! end

function pressure_poisson!(solver::DirectPressureSolver, p, f)
    # Assume the Laplace matrix is known (A) and is possibly factorized

    f = view(f, :)
    p = view(p, :)

    # Use pre-determined decomposition
    p .= solver.A_fact \ f
end

function pressure_poisson!(solver::CGPressureSolver, p, f)
    (; A, abstol, reltol, maxiter) = solver
    f = view(f, :)
    p = view(p, :)
    cg!(p, A, f; abstol, reltol, maxiter)
end

function pressure_poisson!(solver::CGPressureSolverManual, p, f)
    (; setup, abstol, reltol, maxiter, r, G, M, q) = solver
    (; Ip) = setup.grid
    T = typeof(reltol)
    
    # Initial residual
    pressuregradient!(G, p, setup)
    divergence!(M, G, setup)

    # Intialize
    q .= 0
    r .= f .- M
    residual = norm(r[Ip])
    prev_residual = one(residual)
    tolerance = max(reltol * residual, abstol)
    iteration = 0

    while iteration < maxiter && residual > tolerance
        β = residual^2 / prev_residual^2
        q .= r .+ β .* q

        pressuregradient!(G, q, setup)
        divergence!(M, G, setup)
        α = residual^2 / sum(q[Ip] .* M[Ip])

        p .+= α .* q
        r .-= α .* M

        # Periodic paddding (maybe)
        apply_bc_p!(p, T(0), setup)

        prev_residual = residual
        residual = norm(r[Ip])

        iteration += 1
    end

    p
end

function pressure_poisson!(solver::SpectralPressureSolver, p, f)
    (; setup, Ahat, fhat, phat) = solver
    (; Ip) = setup.grid

    f = @view f[Ip]

    phat .= complex.(f)

    # Fourier transform of right hand side
    fft!(phat)

    # Solve for coefficients in Fourier space
    @. phat = -phat / Ahat

    # Transform back
    ifft!(phat)
    @. p[Ip] = real(phat)

    p
end
