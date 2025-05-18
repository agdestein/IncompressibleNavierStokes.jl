"Create empty scalar field."
scalarfield(setup) = fill!(similar(setup.grid.x[1], setup.grid.N), 0)

"Create empty vector field."
vectorfield(setup) =
    fill!(similar(setup.grid.x[1], setup.grid.N..., setup.grid.dimension()), 0)

"Non-symmetric tensor field, stored as a named tuple `σ.ij`."
tensorfield(setup) = tensorfield(setup.grid.dimension, setup)
tensorfield(::Dimension{2}, setup) = (;
    xx = scalarfield(setup),
    yx = scalarfield(setup),
    xy = scalarfield(setup),
    yy = scalarfield(setup),
)
tensorfield(::Dimension{3}, setup) = (;
    xx = scalarfield(setup),
    yx = scalarfield(setup),
    zx = scalarfield(setup),
    xy = scalarfield(setup),
    yy = scalarfield(setup),
    zy = scalarfield(setup),
    xz = scalarfield(setup),
    yz = scalarfield(setup),
    zz = scalarfield(setup),
)

"Symmetric tensor field, stored as a named tuple `σ.ij`."
symmetric_tensorfield(setup) = symmetric_tensorfield(setup.grid.dimension, setup)
symmetric_tensorfield(::Dimension{2}, setup) =
    (; xx = scalarfield(setup), xy = scalarfield(setup), yy = scalarfield(setup))
symmetric_tensorfield(::Dimension{3}, setup) = (;
    xx = scalarfield(setup),
    xy = scalarfield(setup),
    xz = scalarfield(setup),
    yy = scalarfield(setup),
    yz = scalarfield(setup),
    zz = scalarfield(setup),
)

"""
Create divergence free velocity field `u` with boundary conditions at time `t`.
The initial conditions of `u[α]` are specified by the function
`ufunc(α, x...)`.
"""
function velocityfield(
    setup,
    ufunc,
    t = convert(eltype(setup.grid.x[1]), 0);
    psolver = default_psolver(setup),
    doproject = true,
)
    (; grid) = setup
    (; dimension, Iu, xu) = grid

    D = dimension()

    # Allocate velocity
    u = vectorfield(setup)

    # Initial velocities
    for α = 1:D
        xin = ntuple(
            β -> reshape(xu[α][β][Iu[α].indices[β]], ntuple(Returns(1), β - 1)..., :),
            D,
        )
        u[Iu[α], α] .= ufunc.(α, xin...)
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

"Create temperature field from function with boundary conditions at time `t`."
function temperaturefield(setup, tempfunc, t = zero(eltype(setup.grid.x[1])))
    (; grid) = setup
    (; dimension, N, Ip, xp) = grid
    D = dimension()
    xin = ntuple(β -> reshape(xp[β][Ip.indices[β]], ntuple(Returns(1), β - 1)..., :), D)
    temperature = scalarfield(setup)
    temperature[Ip] .= tempfunc.(xin...)
    apply_bc_temp!(temperature, t, setup)
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
            0:(K[α]-1),
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
            0:(KK[α]-1),
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
    stack(uhat)
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
    (; dimension, N, Δ) = grid
    D = dimension()

    assert_uniform_periodic(setup, "Random field")

    # Create random velocity field
    uhat = create_spectrum(; setup, kp, rng)
    u = ifft(uhat, 1:D)
    u = @. A * real(u)

    # Add ghost volumes (one on each side for periodic)
    u = pad_circular(u, 1; dims = 1:D)

    # # Interpolate to staggered grid
    # interpolate_p_u!(u, setup)

    # Make velocity field divergence free on staggered grid
    # (it is already diergence free on the "spectral grid")
    apply_bc_u!(u, t, setup)
    u = project(u, setup; psolver)
    apply_bc_u!(u, t, setup)
end
