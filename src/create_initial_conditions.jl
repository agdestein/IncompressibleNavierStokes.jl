"""
Create divergence free initial velocity field `u` at starting time `t`.
The initial conditions of `u[α]` are specified by the function
`initial_velocity(Dimension(α), x...)`.
"""
function create_initial_conditions(
    setup,
    initial_velocity,
    t = convert(eltype(setup.grid.x[1]), 0);
    psolver = default_psolver(setup),
    doproject = true,
)
    (; grid) = setup
    (; dimension, N, Iu, Ip, x, xp, Ω) = grid

    T = eltype(x[1])
    D = dimension()

    # Allocate velocity
    u = ntuple(d -> fill!(similar(x[1], N), 0), D)

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

# function create_spectrum(; setup, A, σ, s, rng = Random.default_rng())
#     (; dimension, x, N) = setup.grid
#     T = eltype(x[1])
#     D = dimension()
#     K = N .÷ 2
#     k = ntuple(
#         α -> reshape(1:K[α], ntuple(Returns(1), α - 1)..., :, ntuple(Returns(1), D - α)...),
#         D,
#     )
#     a = fill!(similar(x[1], Complex{T}, K), 1)
#     τ = T(2π)
#     a .*= prod(N) * A / sqrt(τ^2 * 2σ^2)
#     for α = 1:D
#         kα = k[α]
#         @. a *= exp(-max(abs(kα) - s, 0)^2 / 2σ^2)
#     end
#     @. a *= randn(rng, T) * exp(im * τ * rand(rng, T))
#     for α = 1:D
#         a = cat(a, reverse(a; dims = α); dims = α)
#     end
#     a
# end

function create_spectrum(; setup, kp, rng = Random.default_rng())
    (; dimension, x, N) = setup.grid
    T = eltype(x[1])
    D = dimension()
    τ = T(2π)
    @assert all(iseven, N) "Spectrum only implemented for even number of volumes."

    # Maximum wavenumber (remove ghost volumes)
    K = @. (N - 2) ÷ 2

    # Wavenumber vectors
    kk = ntuple(
        α -> reshape(
            0:K[α]-1,
            ntuple(Returns(1), α - 1)...,
            :,
            ntuple(Returns(1), D - α)...,
        ),
        D,
    )

    # Wavevector magnitude
    k = fill!(similar(x[1], K), 0)
    for α = 1:D
        @. k += kk[α]^2
    end
    k .= sqrt.(k)

    # Shared magnitude
    A = T(8τ / 3) / kp^5

    # Velocity magnitude
    # a = @. complex(1) * sqrt(A * k^4 * exp(-(k / kp)^2))
    a = @. complex(1) * sqrt(A * k^4 * exp(-τ * (k / kp)^2))
    a .*= prod(N)

    # Apply random phase shift
    ξ = ntuple(α -> rand!(rng, similar(x[1], K)), D)
    for α = 1:D
        a = cat(a, reverse(a; dims = α); dims = α)
        ξ = ntuple(D) do β
            s = α == β ? -1 : 1
            ξβ = ξ[β]
            cat(ξβ, reverse(s .* ξβ; dims = α); dims = α)
        end
    end
    ξ = sum(ξ)
    a = @. exp(im * τ * ξ) * a

    KK = 2 .* K
    kkkk = ntuple(
        α -> reshape(
            0:KK[α]-1,
            ntuple(Returns(1), α - 1)...,
            :,
            ntuple(Returns(1), D - α)...,
        ),
        D,
    )
    knorm = fill!(similar(x[1], KK), 0)
    for α = 1:D
        @. knorm += kkkk[α]^2
    end
    knorm .= sqrt.(knorm)

    # Create random unit vector for each wavenumber
    if D == 2
        θ = rand!(rng, similar(x[1], KK))
        e = (cospi.(2 .* θ), sinpi.(2 .* θ))
    elseif D == 3
        θ = rand!(rng, similar(x[1], KK))
        ϕ = rand!(rng, similar(x[1], KK))
        e = (sinpi.(θ) .* cospi.(2 .* ϕ), sinpi.(θ) .* sinpi.(2 .* ϕ), cospi.(θ))
    end

    # Remove non-divergence free part: (I - k k^T / k^2) e
    ke = sum(α -> e[α] .* kkkk[α], 1:D)
    for α = 1:D
        e0 = e[α][1:1] # CUDA doesn't like e[α][1]
        @. e[α] -= kkkk[α] * ke / knorm^2
        # Restore k=0 component, which is divergence free anyways
        e[α][1:1] .= e0
    end

    # Normalize
    enorm = sqrt.(sum(α -> e[α] .^ 2, 1:D))
    for α = 1:D
        e[α] ./= enorm
    end

    # Split velocity magnitude a into velocity components a*eα
    uhat = ntuple(D) do α
        eα = e[α]
        # for β = 1:D
        #     eα = cat(eα, reverse(eα; dims = β); dims = β)
        # end
        a .* eα
    end
end

"""
Create random field, as in [Orlandi2000](@cite).

- `A`: Eddy amplitude scaling
- `kp`: Peak energy wavenumber
"""
function random_field(
    setup,
    t = zero(eltype(setup.grid.x[1]));
    A = 1,
    kp = 10,
    psolver = default_psolver(setup),
    rng = Random.default_rng(),
)
    (; grid, boundary_conditions) = setup
    (; dimension, N, x, Δ, Ip, Ω) = setup.grid
    D = dimension()
    T = eltype(x[1])

    @assert all(==(PeriodicBC()), boundary_conditions) "Random field requires periodic boundary conditions."
    @assert all(α -> all(==(Δ[α][1]), Δ[α]), 1:D) "Random field requires uniform grid spacing."
    @assert all(iseven, N) "Random field requires even number of volumes."

    # Create random velocity field
    uhat = create_spectrum(; setup, kp, rng)
    u = ifft.(uhat)
    u = map(u -> A .* real.(u), u)
    @show size.(u)

    # Add ghost volumes (one on each side for periodic)
    u = pad_circular.(u, 1; dims = 1:D)
    @show size.(u)
    error()

    # # Interpolate to staggered grid
    # interpolate_p_u!(u, setup)

    # Make velocity field divergence free on staggered grid
    # (it is already diergence free on the "spectral grid")
    apply_bc_u!(u, t, setup)
    u = project(u, setup; psolver)
    apply_bc_u!(u, t, setup)
end
