"""
    create_initial_conditions(
        setup,
        initial_velocity,
        t = 0;
        pressure_solver = CGPressureSolverManual(setup),
    )

Create initial vectors `(u, p)` at starting time `t`.
The initial conditions of `u[α]` are specified by the function
`initial_velocity(Dimension(α), x...)`.
"""
function create_initial_conditions(
    setup,
    initial_velocity,
    t = convert(eltype(setup.grid.x[1]), 0);
    pressure_solver = CGPressureSolverManual(setup),
)
    (; grid) = setup
    (; dimension, N, Iu, Ip, x, xp, Ω) = grid

    T = eltype(x[1])
    D = dimension()

    # Allocate velocity and pressure
    u = ntuple(d -> KernelAbstractions.zeros(get_backend(x[1]), T, N...), D)
    p = KernelAbstractions.zeros(get_backend(x[1]), T, N...)

    # Initial velocities
    for α = 1:D
        xin = ntuple(
            β -> reshape(α == β ? x[β][2:end] : xp[β], ntuple(Returns(1), β - 1)..., :),
            D,
        )
        u[α][Iu[α]] .= initial_velocity.((Dimension(α),), xin...)[Iu[α]]
    end

    apply_bc_u!(u, t, setup)

    # Kinetic energy and momentum of initial velocity field
    # Iteration 1 corresponds to t₀ = 0 (for unsteady simulations)
    maxdiv = maximum(abs, divergence(u, setup))

    # TODO: Maybe eps(T)^(3//4)
    if maxdiv > 1e-12
        @warn "Initial velocity field not (discretely) divergence free: $maxdiv.\n" *
              "Performing additional projection."

        # Make velocity field divergence free
        f = divergence(u, setup)
        @. f *= Ω
        Δp = pressure_poisson(pressure_solver, f)
        p .= Δp
        apply_bc_p!(p, t, setup)
        G = pressuregradient(p, setup)
        for α = 1:D
            u[α] .-= G[α]
        end
    end

    p = pressure_additional_solve(pressure_solver, u, t, setup)
    apply_bc_p!(p, t, setup)

    # Initial conditions, including initial boundary condititions
    u, p
end

function create_spectrum(N; A, σ, s, backend)
    T = typeof(A)
    D = length(N)
    K = N .÷ 2
    k = ntuple(
        α -> reshape(1:K[α], ntuple(Returns(1), α - 1)..., :, ntuple(Returns(1), D - α)...),
        D,
    )
    a = KernelAbstractions.ones(backend, Complex{T}, K)
    AT = typeof(a)
    # k = AT.(Array{Complex{T}}.(k))
    # k = AT.(k)
    τ = T(2π)
    a .*= A / sqrt(τ^2 * 2σ^2)
    for α = 1:D
        kα = k[α]
        @. a *= exp(-max(abs(kα) - s, s)^2 / 2σ^2)
    end
    @. a *= randn(T) * exp(im * τ * rand(T))
    for α = 1:D
        a = cat(a, reverse(a; dims = α); dims = α)
    end
    a
end

"""
    random_field(
        setup, t = 0;
        A = setup.grid.N[1] * 10_000,
        σ = 30,
        s = 5,
        pressure_solver = DirectPressureSolver(setup),
    )

Create random field.

- `K`: Maximum wavenumber
- `A`: Eddy amplitude
- `σ`: Variance
- `s` Wavenumber offset before energy starts decaying
"""
function random_field(
    setup,
    t = zero(eltype(setup.grid.x[1]));
    A = convert(eltype(setup.grid.x[1]), setup.grid.N[1] * 7_500),
    σ = convert(eltype(setup.grid.x[1]), 30),
    s = convert(eltype(setup.grid.x[1]), 5),
    pressure_solver = DirectPressureSolver(setup),
)
    (; dimension, x, N, Ip, Ω) = setup.grid

    D = dimension()
    T = eltype(x[1])
    backend = get_backend(x[1])

    u = ntuple(α -> real.(ifft(create_spectrum(N; A, σ, s, backend))), D)
    apply_bc_u!(u, t, setup)
    M = divergence(u, setup)
    @. M *= Ω
    p = zero(M)

    # Make velocity field divergence free
    pressure_poisson!(pressure_solver, p, M)
    apply_bc_p!(p, t, setup)
    G = pressuregradient(p, setup)
    for α = 1:D
        @. u[α] -= G[α]
    end
    apply_bc_u!(u, t, setup)
    p = pressure_additional_solve(pressure_solver, u, t, setup)
    apply_bc_p!(p, t, setup)

    u, p
end
