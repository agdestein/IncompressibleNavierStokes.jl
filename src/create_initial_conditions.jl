"""
    create_initial_conditions(
        setup,
        initial_velocity,
        t = 0;
        psolver = DirectPressureSolver(setup),
        doproject = true,
    )

Create divergence free initial velocity field `u` at starting time `t`.
The initial conditions of `u[α]` are specified by the function
`initial_velocity(Dimension(α), x...)`.
"""
function create_initial_conditions(
    setup,
    initial_velocity,
    t = convert(eltype(setup.grid.x[1]), 0);
    psolver = DirectPressureSolver(setup),
    doproject = true,
)
    (; grid) = setup
    (; dimension, N, Iu, Ip, x, xp, Ω) = grid

    T = eltype(x[1])
    D = dimension()

    # Allocate velocity
    u = ntuple(d -> similar(x[1], N), D)

    # Initial velocities
    for α = 1:D
        xin = ntuple(
            β -> reshape(α == β ? x[β][2:end] : xp[β], ntuple(Returns(1), β - 1)..., :),
            D,
        )
        u[α][Iu[α]] .= initial_velocity.((Dimension(α),), xin...)[Iu[α]]
    end

    # Make velocity field divergence free
    apply_bc_u!(u, t, setup)
    if doproject 
        u = project(u, setup; psolver)
        apply_bc_u!(u, t, setup)
    end

    # Initial conditions, including initial boundary conditions
    u
end

function create_spectrum(; setup, A, σ, s)
    (; dimension, x, N) = setup.grid
    T = eltype(x[1])
    D = dimension()
    K = N .÷ 2
    k = ntuple(
        α -> reshape(1:K[α], ntuple(Returns(1), α - 1)..., :, ntuple(Returns(1), D - α)...),
        D,
    )
    a = fill!(similar(x[1], Complex{T}, K), 1)
    τ = T(2π)
    a .*= prod(N) * A / sqrt(τ^2 * 2σ^2)
    for α = 1:D
        kα = k[α]
        @. a *= exp(-max(abs(kα) - s, 0)^2 / 2σ^2)
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
        A = 10,
        σ = 30,
        s = 5,
        psolver = DirectPressureSolver(setup),
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
    A = convert(eltype(setup.grid.x[1]), 10),
    σ = convert(eltype(setup.grid.x[1]), 30),
    s = convert(eltype(setup.grid.x[1]), 5),
    psolver = DirectPressureSolver(setup),
)
    (; dimension, x, Ip, Ω) = setup.grid
    D = dimension()
    T = eltype(x[1])
    backend = get_backend(x[1])

    # Create random velocity field
    u = ntuple(α -> real.(ifft(create_spectrum(; setup, A, σ, s))), D)

    # Make velocity field divergence free
    apply_bc_u!(u, t, setup)
    project(u, setup; psolver)
    apply_bc_u!(u, t, setup)
end
