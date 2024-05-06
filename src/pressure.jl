# Note: poisson!(...) is a wrapper around psolver!(...) since we need to define
# a shared rrule. This could also be achieved by something like:
#     abstract type AbstractPSolver end
#     rrule(s::AbstractPSolver)(f) = (s(f), φ -> (NoTangent(), s(φ)))
#     (s::AbstractPSolver)(f) = s(zero(...), f)
#     struct PSolver < AbstractPSolver ... end
#     (::PSolver)(p, f) = # solve Poisson

"""
    poisson(psolver, f)

Solve the Poisson equation for the pressure with right hand side `f` at time `t`.
For periodic and no-slip BC, the sum of `f` should be zero.

Non-mutating/allocating/out-of-place version.

See also [`poisson!`](@ref).
"""
poisson(psolver, f) = poisson!(psolver, zero(f), f)

# Laplacian is auto-adjoint
ChainRulesCore.rrule(::typeof(poisson), psolver, f) =
    (poisson(psolver, f), φ -> (NoTangent(), NoTangent(), poisson(psolver, unthunk(φ))))

"""
    poisson!(solver, p, f)

Solve the Poisson equation for the pressure with right hand side `f` at time `t`.
For periodic and no-slip BC, the sum of `f` should be zero.

Mutating/non-allocating/in-place version.

See also [`poisson`](@ref).
"""
poisson!(psolver, p, f) = psolver(p, f)

"""
    pressure!(p, u, temp, t, setup; psolver, F, div)

Compute pressure from velocity field. This makes the pressure compatible with the velocity
field, resulting in same order pressure as velocity.
"""
function pressure!(p, u, temp, t, setup; psolver, F, div)
    (; grid) = setup
    (; dimension, Iu, Ip, Ω) = grid
    D = dimension()
    momentum!(F, u, temp, t, setup)
    apply_bc_u!(F, t, setup; dudt = true)
    divergence!(div, F, setup)
    @. div *= Ω
    poisson!(psolver, p, div)
    apply_bc_p!(p, t, setup)
    p
end

"""
    pressure(u, temp, t, setup; psolver)

Compute pressure from velocity field. This makes the pressure compatible with the velocity
field, resulting in same order pressure as velocity.
"""
function pressure(u, temp, t, setup; psolver)
    (; grid) = setup
    (; dimension, Iu, Ip, Ω) = grid
    D = dimension()
    F = momentum(u, temp, t, setup)
    F = apply_bc_u(F, t, setup; dudt = true)
    div = divergence(F, setup)
    div = @. div * Ω
    p = poisson(psolver, div)
    p = apply_bc_p(p, t, setup)
    p
end

"""
    project(u, setup; psolver)

Project velocity field onto divergence-free space.
"""
function project(u, setup; psolver)
    (; Ω) = setup.grid
    T = eltype(u[1])

    # Divergence of tentative velocity field
    div = divergence(u, setup)
    div = @. div * Ω

    # Solve the Poisson equation
    p = poisson(psolver, div)

    # Apply pressure correction term
    p = apply_bc_p(p, T(0), setup)
    G = pressuregradient(p, setup)
    u .- G
end

"""
    project!(u, setup; psolver, div, p)

Project velocity field onto divergence-free space.
"""
function project!(u, setup; psolver, div, p)
    (; Ω) = setup.grid
    T = eltype(u[1])

    # Divergence of tentative velocity field
    divergence!(div, u, setup)
    @. div *= Ω

    # Solve the Poisson equation
    poisson!(psolver, p, div)
    apply_bc_p!(p, T(0), setup)

    # Apply pressure correction term
    applypressure!(u, p, setup)
end

"""
    default_psolver(setup)

Get default Poisson solver from setup.
"""
function default_psolver(setup)
    (; grid, boundary_conditions) = setup
    (; dimension, Δ) = grid
    D = dimension()
    Δx = first.(Array.(Δ))
    isperiodic =
        all(bc -> bc[1] isa PeriodicBC && bc[2] isa PeriodicBC, boundary_conditions)
    isuniform = all(α -> all(≈(Δx[α]), Δ[α]), 1:D)
    if isperiodic && isuniform
        psolver_spectral(setup)
    else
        psolver_direct(setup)
    end
end

"""
    poisson_direct(setup)

Create direct Poisson solver using an appropriate matrix decomposition.
"""
psolver_direct(setup) = psolver_direct(setup.grid.x[1], setup) # Dispatch on array type

# CPU version
function psolver_direct(::Array, setup)
    (; grid, boundary_conditions) = setup
    (; x, Np, Ip) = grid
    T = Float64 # This is currently required for SuiteSparse
    L = laplacian_mat(setup)
    isdefinite =
        any(bc -> bc[1] isa PressureBC || bc[2] isa PressureBC, boundary_conditions)
    if isdefinite
        # No extra DOF
        ftemp = zeros(T, prod(Np))
        ptemp = zeros(T, prod(Np))
        viewrange = (:)
    else
        # With extra DOF
        ftemp = zeros(T, prod(Np) + 1)
        ptemp = zeros(T, prod(Np) + 1)
        e = ones(T, size(L, 2))
        L = [L e; e' 0]
        viewrange = 1:prod(Np)
    end
    fact = factorize(L)
    function psolve!(p, f)
        copyto!(view(ftemp, viewrange), view(view(f, Ip), :))
        ptemp .= fact \ ftemp
        copyto!(view(view(p, Ip), :), eltype(p).(view(ptemp, viewrange)))
        p
    end
end

"""
    psolver_cg_matrix(setup; kwargs...)

Conjugate gradients iterative Poisson solver.
The `kwargs` are passed to the `cg!` function
from IterativeSolvers.jl.
"""
function psolver_cg_matrix(setup; kwargs...)
    (; x, Np, Ip) = grid
    L = laplacian_mat(setup)
    isdefinite =
        any(bc -> bc[1] isa PressureBC || bc[2] isa PressureBC, boundary_conditions)
    if isdefinite
        # No extra DOF
        ftemp = fill!(similar(x[1], prod(Np)), 0)
        ptemp = fill!(similar(x[1], prod(Np)), 0)
        viewrange = (:)
    else
        # With extra DOF
        ftemp = fill!(similar(x[1], prod(Np) + 1), 0)
        ptemp = fill!(similar(x[1], prod(Np) + 1), 0)
        e = fill!(similar(x[1], prod(Np)), 1)
        L = [L e; e' 0]
        viewrange = 1:prod(Np)
    end
    function psolve!(p, f)
        copyto!(view(ftemp, viewrange), view(view(f, Ip), :))
        cg!(ptemp, L, ftemp; kwargs...)
        copyto!(view(view(p, Ip), :), view(ptemp, viewrange))
        p
    end
end

# Preconditioner
function create_laplace_diag(setup)
    (; grid, workgroupsize) = setup
    (; dimension, Δ, Δu, N, Np, Ip, Ω) = grid
    D = dimension()
    δ = Offset{D}()
    @kernel function _laplace_diag!(z, p, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        d = zero(eltype(z))
        for α = 1:length(I)
            d -= Ω[I] / Δ[α][I[α]] * (1 / Δu[α][I[α]] + 1 / Δu[α][I[α]-1])
        end
        z[I] = -p[I] / d
    end
    ndrange = Np
    I0 = first(Ip)
    I0 -= oneunit(I0)
    laplace_diag(z, p) = _laplace_diag!(get_backend(z), workgroupsize)(z, p, I0; ndrange)
end

"""
    psolver_cg(
        setup;
        abstol = zero(eltype(setup.grid.x[1])),
        reltol = sqrt(eps(eltype(setup.grid.x[1]))),
        maxiter = prod(setup.grid.Np),
        preconditioner = create_laplace_diag(setup),
    )

Conjugate gradients iterative Poisson solver.
"""
function psolver_cg(
    setup;
    abstol = zero(eltype(setup.grid.x[1])),
    reltol = sqrt(eps(eltype(setup.grid.x[1]))),
    maxiter = prod(setup.grid.Np),
    preconditioner = create_laplace_diag(setup),
)
    (; grid, workgroupsize) = setup
    (; Np, Ip, Ω) = grid
    T = eltype(setup.grid.x[1])
    r = similar(setup.grid.x[1], setup.grid.N)
    L = similar(setup.grid.x[1], setup.grid.N)
    q = similar(setup.grid.x[1], setup.grid.N)
    function psolve!(p, f)
        function innerdot(a, b)
            @kernel function innerdot!(d, a, b, I0)
                I = @index(Global, Cartesian)
                I = I + I0
                d[I-I+I0] += a[I] * b[I]
                # a[I] = b[I]
            end
            # d = zero(eltype(a))
            I0 = first(Ip)
            I0 -= oneunit(I0)
            d = fill!(similar(a, ntuple(Returns(1), length(I0))), 0),
            innerdot!(get_backend(a), workgroupsize)(d, a, b, I0; ndrange = Np)
            d[]
        end

        p .= 0

        # Initial residual
        laplacian!(L, p, setup)

        # Initialize
        q .= 0
        r .= f .- L
        ρ_prev = one(T)
        # residual = norm(r[Ip])
        residual = sqrt(sum(abs2, view(r, Ip)))
        # residual = norm(r)
        tolerance = max(reltol * residual, abstol)
        iteration = 0

        while iteration < maxiter && residual > tolerance
            preconditioner(L, r)

            # ρ = sum(L[Ip] .* r[Ip])
            ρ = dot(view(L, Ip), view(r, Ip))
            # ρ = innerdot(L, r)
            # ρ = dot(L, r)

            β = ρ / ρ_prev
            q .= L .+ β .* q

            # Periodic/symmetric padding (maybe)
            apply_bc_p!(q, T(0), setup)
            laplacian!(L, q, setup)
            # α = ρ / sum(q[Ip] .* L[Ip])
            α = ρ / dot(view(q, Ip), view(L, Ip))
            # α = ρ / innerdot(q, L)
            # α = ρ / dot(q, L)

            p .+= α .* q
            r .-= α .* L

            ρ_prev = ρ
            # residual = norm(r[Ip])
            residual = sqrt(sum(abs2, view(r, Ip)))
            # residual = sqrt(sum(abs2, r))
            # residual = sqrt(innerdot(r, r))

            iteration += 1
        end

        # @show iteration residual tolerance

        p
    end
end

"""
    psolver_spectral(setup)

Create spectral Poisson solver from setup.
"""
function psolver_spectral(setup)
    (; grid, boundary_conditions) = setup
    (; dimension, Δ, Np, Ip, x) = grid

    D = dimension()
    T = eltype(Δ[1])

    Δx = first.(Array.(Δ))

    @assert(
        all(bc -> bc[1] isa PeriodicBC && bc[2] isa PeriodicBC, boundary_conditions),
        "Spectral psolver only implemented for periodic boundary conditions",
    )

    @assert(
        all(α -> all(≈(Δx[α]), Δ[α]), 1:D),
        "Spectral psolver requires uniform grid along each dimension",
    )

    # Fourier transform of the discretization
    # Assuming uniform grid, although Δx[1] and Δx[2] do not need to be the same

    k = ntuple(
        d -> reshape(
            0:Np[d]-1,
            ntuple(Returns(1), d - 1)...,
            :,
            ntuple(Returns(1), D - d)...,
        ),
        D,
    )

    Ahat = fill!(similar(x[1], Complex{T}, Np), 0)
    Tπ = T(π) # CUDA doesn't like pi
    for d = 1:D
        @. Ahat += sin(k[d] * Tπ / Np[d])^2 / Δx[d]^2
    end

    # Scale with Δx*Δy*Δz, since we solve the PDE in integrated form
    Ahat .*= 4 * prod(Δx)

    # Pressure is determined up to constant. By setting the constant
    # scaling factor to 1, we preserve the average.
    # Note use of singleton range 1:1 instead of scalar index 1
    # (otherwise CUDA gets annoyed)
    Ahat[1:1] .= 1

    # Placeholders for intermediate results
    phat = zero(Ahat)
    fhat = zero(Ahat)
    plan = plan_fft(fhat)

    function psolver(p, f)
        f = view(f, Ip)

        fhat .= complex.(f)

        # Fourier transform of right hand side
        mul!(phat, plan, fhat)

        # Solve for coefficients in Fourier space
        @. fhat = -phat / Ahat

        # Pressure is determined up to constant. We set this to 0 (instead of
        # phat[1] / 0 = Inf)
        # Note use of singleton range 1:1 instead of scalar index 1
        # (otherwise CUDA gets annoyed)
        fhat[1:1] .= 0

        # Transform back
        ldiv!(phat, plan, fhat)
        @. p[Ip] = real(phat)

        p
    end
end

"""
    psolver_spectral_lowmemory(setup)

Create spectral Poisson solver from setup.
This one is slower than `psolver_spectral` but occupies less memory.
"""
function psolver_spectral_lowmemory(setup)
    (; grid, boundary_conditions) = setup
    (; dimension, Δ, Np, Ip, x) = grid

    D = dimension()
    T = eltype(Δ[1])

    Δx = Array(Δ[1])[1]

    @assert(
        all(bc -> bc[1] isa PeriodicBC && bc[2] isa PeriodicBC, boundary_conditions),
        "Spectral psolver only implemented for periodic boundary conditions",
    )

    @assert(
        all(α -> all(≈(Δx), Δ[α]), 1:D),
        "Spectral psolver requires uniform grid along each dimension",
    )

    @assert all(n -> n == Np[1], Np)

    # Fourier transform of the discretization
    # Assuming uniform grid, although Δx[1] and Δx[2] do not need to be the same

    k = 0:Np[1]-1

    Tπ = T(π) # CUDA doesn't like pi
    ahat = fill!(similar(x[1], Np[1]), 0)
    @. ahat = 4 * Δx^D * sin(k * Tπ / Np[1])^2 / Δx^2

    # Placeholders for intermediate results
    phat = fill!(similar(x[1], Complex{T}, Np), 0)

    function psolve!(p, f)
        f = view(f, Ip)

        phat .= complex.(f)

        # Fourier transform of right hand side
        fft!(phat)

        # Solve for coefficients in Fourier space
        if D == 2
            ax = ahat
            ay = reshape(ahat, 1, :)
            @. phat = -phat / (ax + ay)
        else
            ax = ahat
            ay = reshape(ahat, 1, :)
            az = reshape(ahat, 1, 1, :)
            @. phat = -phat / (ax + ay + az)
        end

        # Pressure is determined up to constant. We set this to 0 (instead of
        # phat[1] / 0 = Inf)
        # Note use of singleton range 1:1 instead of scalar index 1
        # (otherwise CUDA gets annoyed)
        phat[1:1] .= 0

        # Transform back
        ifft!(phat)
        @. p[Ip] = real(phat)

        p
    end
end
