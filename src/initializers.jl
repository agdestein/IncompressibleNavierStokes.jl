"Create empty scalar field."
scalarfield(setup) = fill!(similar(setup.x[1], setup.N), 0)

"Create empty vector field."
vectorfield(setup) = fill!(similar(setup.x[1], setup.N..., setup.dimension()), 0)

"Non-symmetric tensor field, stored as a named tuple `σ.ij`."
tensorfield(setup) = tensorfield(setup.dimension, setup)
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
symmetric_tensorfield(setup) = symmetric_tensorfield(setup.dimension, setup)
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
    t = convert(eltype(setup.x[1]), 0);
    psolver = default_psolver(setup),
    doproject = true,
)
    (; dimension, Iu, xu) = setup

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
function temperaturefield(setup, tempfunc, t = zero(eltype(setup.x[1])))
    (; dimension, N, Ip, xp) = setup
    D = dimension()
    xin = ntuple(β -> reshape(xp[β][Ip.indices[β]], ntuple(Returns(1), β - 1)..., :), D)
    temperature = scalarfield(setup)
    temperature[Ip] .= tempfunc.(xin...)
    apply_bc_temp!(temperature, t, setup)
end

"""
Initial energy spectrum profile ``E(k) \\propto k^4 e^{-2 (k / k_p)^2}``
with peak wavenumber `kpeak`, as in Orlandi [Orlandi2000](@cite).
"""
orlandi_profile(k; kpeak = 10) = k^4 * exp(-2 * (k / kpeak)^2)

"""
Create a random divergence-free velocity field with a prescribed energy
spectrum profile. The energy in the wavenumber shell `κ ≤ |k| < κ + 1` is
`totalenergy * profile(κ; kwargs...) / p`, where `p` normalizes the profile
such that the mean kinetic energy density `⟨u_i u_i⟩ / 2` is exactly
`totalenergy`. By default, the Orlandi form [`orlandi_profile`](@ref) is used
(pass e.g. `kpeak = 5` to move the peak).

The field is constructed in spectral space from white noise, projected onto
the divergence-free space of the *staggered* grid (using the modified
wavenumbers of the staggered divergence operator), and rescaled shell-wise.
The resulting field is thus exactly divergence-free on the staggered grid and
has exactly the prescribed spectrum.
"""
function random_field(
    setup,
    t = zero(eltype(setup.x[1]));
    profile = orlandi_profile,
    totalenergy = one(eltype(setup.x[1])),
    rng = Random.default_rng(),
    kwargs...,
)
    (; dimension, Np, Ip, x, xlims, backend) = setup
    D = dimension()
    T = eltype(x[1])

    assert_uniform_periodic(setup, "Random field")

    # Gaussian white noise per component. Taking the RFFT of a real field
    # automatically gives Hermitian-symmetric spectral coefficients with
    # random phases.
    v = similar(x[1], Np)
    plan = plan_rfft(v)
    uhat = ntuple(α -> (randn!(rng, v); plan * v), D)

    # Spectral symbol of the staggered divergence operator
    # (backward difference): m[α] = (1 - exp(-i θ[α])) / h[α],
    # with θ[α] = 2π k[α] / Np[α] the phase shift per grid point.
    m = ntuple(D) do α
        L = xlims[α][2] - xlims[α][1]
        h = L / Np[α]
        kint = α == 1 ? collect(0:(Np[1]÷2)) : map(i -> fftfreq_int(Np[α], i), 1:Np[α])
        mα = map(k -> (1 - exp(-im * T(2π) * k / Np[α])) / h, kint)
        reshape(adapt(backend, mα), ntuple(Returns(1), α - 1)..., :)
    end

    # Project onto the divergence-free space: û ← û - conj(m) (m ⋅ û) / |m|².
    # This makes the staggered-grid divergence of the final field exactly
    # zero. The `k = 0` mode (where `m = 0`) is left intact.
    mdotu = m[1] .* uhat[1]
    for α = 2:D
        @. mdotu += m[α] * uhat[α]
    end
    m2 = .+(map(mα -> abs2.(mα), m)...)
    for α = 1:D
        @. uhat[α] -= ifelse(m2 > 0, conj(m[α]) * mdotu / m2, zero(mdotu))
    end

    # Current spectral energy distribution (unnormalized)
    efield = abs2.(uhat[1])
    for α = 2:D
        @. efield += abs2(uhat[α])
    end

    # Rescale all wavenumber shells to match the target profile.
    # The last shell contains the largest wavenumber on the grid.
    kdiag = isqrt(sum(α -> (Np[α] ÷ 2)^2, 1:D))
    stuff = spectral_stuff(setup; kmax = kdiag)
    totalprofile = sum(κ -> T(profile(T(κ); kwargs...)), 0:kdiag)
    Ntot = T(prod(Np))
    for (j, κ) in enumerate(stuff.κ)
        eshell = sum(view(efield, stuff.energyinds[j])) / (2 * Ntot^2)
        e0 = totalenergy * T(profile(T(κ); kwargs...)) / totalprofile
        factor = eshell > 0 ? sqrt(e0 / eshell) : zero(T)
        for α = 1:D
            shell = view(uhat[α], stuff.inds[j])
            shell .*= factor
        end
    end

    # Transform to physical space and fill the staggered velocity components
    u = vectorfield(setup)
    for α = 1:D
        ldiv!(v, plan, uhat[α])
        copyto!(view(u, Ip, α), v)
    end
    apply_bc_u!(u, t, setup)
    u
end
