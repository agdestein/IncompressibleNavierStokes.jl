# Note: poisson!(...) is a wrapper around psolver!(...) since we need to define
# a shared rrule. This could also be achieved by something like:
#     abstract type AbstractPSolver end
#     rrule(s::AbstractPSolver)(f) = (s(f), φ -> (NoTangent(), s(φ)))
#     (s::AbstractPSolver)(f) = s(zero(...), f)
#     struct PSolver < AbstractPSolver ... end
#     (::PSolver)(p, f) = # solve Poisson

"""
Solve the Poisson equation for the pressure with right hand side `f` at time `t`.
For periodic and no-slip BC, the sum of `f` should be zero.

Differentiable version.
"""
poisson(psolver, f) = poisson!(psolver, copy(f))

# Laplacian is auto-adjoint
ChainRulesCore.rrule(::typeof(poisson), psolver, f) =
    (poisson(psolver, f), φ -> (NoTangent(), NoTangent(), poisson(psolver, unthunk(φ))))

"Solve the Poisson equation for the pressure (in-place version)."
poisson!(psolver, f) = psolver(f)

"Project velocity field onto divergence-free space (differentiable version)."
function project(u, setup; psolver)
    T = eltype(u)

    # Divergence of tentative velocity field
    div = divergence(u, setup)
    div = scalewithvolume(div, setup)

    # Solve the Poisson equation
    p = poisson(psolver, div)

    # Apply pressure correction term
    p = apply_bc_p(p, T(0), setup)
    G = pressuregradient(p, setup)
    u .- G
end

"Project velocity field onto divergence-free space (in-place version)."
function project!(u, setup; psolver, p)
    T = eltype(u)

    # Divergence of tentative velocity field
    divergence!(p, u, setup)
    scalewithvolume!(p, setup) # *Δx^D

    # Solve the Poisson equation
    poisson!(psolver, p)
    apply_bc_p!(p, T(0), setup)

    # Apply pressure correction term
    applypressure!(u, p, setup)
end

"Get default Poisson solver from setup."
function default_psolver(setup)
    (; dimension, Δ, boundary_conditions) = setup
    D = dimension()
    Δx = first.(Array.(Δ))
    isperiodic =
        all(bc -> bc[1] isa PeriodicBC && bc[2] isa PeriodicBC, boundary_conditions.u)
    isuniform = all(α -> all(≈(Δx[α]), Δ[α]), 1:D)
    if isperiodic && isuniform
        psolver_spectral(setup)
    else
        psolver_direct(setup)
    end
end

"Create direct Poisson solver using an appropriate matrix decomposition."
psolver_direct(setup) = psolver_direct(setup.x[1], setup) # Dispatch on array type

psolver_direct(::Any, setup) = error("""
    Unsupported array type.

    If you are using CUDA `CuArray`s, do

    ```julia
    using Pkg
    Pkg.add("CUDSS")
    ```

    This will trigger an extension that works for `CuArrays`.
    """)

# CPU version
function psolver_direct(::Array, setup)
    (; x, Np, Ip, boundary_conditions) = setup
    T = eltype(x[1])
    L = laplacian_mat(setup)
    isdefinite =
        any(bc -> bc[1] isa PressureBC || bc[2] isa PressureBC, boundary_conditions.u)
    if isdefinite
        # No extra DOF
        Ttemp = Float64 # This is currently required for SuiteSparse LU
        ftemp = zeros(Ttemp, prod(Np))
        ptemp = zeros(Ttemp, prod(Np))
        viewrange = (:)
        fact = lu(L)
    else
        # With extra DOF
        ftemp = zeros(T, prod(Np) + 1)
        ptemp = zeros(T, prod(Np) + 1)
        e = ones(T, size(L, 2))
        L = [L e; e' 0]
        maximum(L - L') < sqrt(eps(T)) || error("Matrix not symmetric")
        L = @. (L + L') / 2
        viewrange = 1:prod(Np)
        fact = ldlt(L)
    end
    # fact = factorize(L)
    function psolve!(p)
        copyto!(view(ftemp, viewrange), view(view(p, Ip), :))
        ptemp .= fact \ ftemp
        if isdefinite && !(0.0 isa T)
            # Convert from Float64 to T
            copyto!(view(view(p, Ip), :), T.(view(ptemp, viewrange)))
        else
            copyto!(view(view(p, Ip), :), view(ptemp, viewrange))
        end
        p
    end
end

sparseadapt(::CPU, A) = A

"""
Conjugate gradients iterative Poisson solver.
The `kwargs` are passed to the `cg!` function
from IterativeSolvers.jl.
"""
function psolver_cg_matrix(setup; kwargs...)
    (; x, Np, Ip, boundary_conditions, backend) = setup
    T = eltype(x[1])
    L = laplacian_mat(setup)
    isdefinite =
        any(bc -> bc[1] isa PressureBC || bc[2] isa PressureBC, boundary_conditions.u)
    if isdefinite
        # No extra DOF
        ftemp = fill!(similar(x[1], prod(Np)), 0)
        ptemp = fill!(similar(x[1], prod(Np)), 0)
        viewrange = (:)
    else
        # With extra DOF
        ftemp = fill!(similar(x[1], prod(Np) + 1), 0)
        ptemp = fill!(similar(x[1], prod(Np) + 1), 0)
        e = fill(T(1), prod(Np))
        L = [L e; e' 0]
        L = sparseadapt(backend, L)
        viewrange = 1:prod(Np)
    end
    function psolve!(p)
        copyto!(view(ftemp, viewrange), view(view(p, Ip), :))
        cg!(ptemp, L, ftemp; kwargs...)
        copyto!(view(view(p, Ip), :), view(ptemp, viewrange))
        p
    end
end

# Preconditioner
function create_laplace_diag(setup)
    (; Δ, Δu, Np, Ip, workgroupsize) = setup
    @kernel function _laplace_diag!(z, p, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        ΔI = getindex.(Δ, I.I)
        ΩI = prod(ΔI)
        d = zero(eltype(z))
        for α = 1:length(I)
            d -= ΩI / Δ[α][I[α]] * (1 / Δu[α][I[α]] + 1 / Δu[α][I[α]-1])
        end
        z[I] = -p[I] / d
    end
    ndrange = Np
    I0 = first(Ip)
    I0 -= oneunit(I0)
    laplace_diag(z, p) = _laplace_diag!(get_backend(z), workgroupsize)(z, p, I0; ndrange)
end

"Conjugate gradients iterative Poisson solver."
function psolver_cg(
    setup;
    abstol = zero(eltype(setup.x[1])),
    reltol = sqrt(eps(eltype(setup.x[1]))),
    maxiter = prod(setup.Np),
    preconditioner = create_laplace_diag(setup),
)
    (; Np, Ip, workgroupsize) = setup
    T = eltype(setup.x[1])
    r = scalarfield(setup)
    L = scalarfield(setup)
    q = scalarfield(setup)
    function psolve!(p)
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

        # Initialize
        q .= 0
        laplacian!(L, q, setup) # Initial residual
        r .= p .- L
        ρ_prev = one(T)
        # residual = norm(r[Ip])
        residual = sqrt(sum(abs2, view(r, Ip)))
        # residual = norm(r)
        tolerance = max(reltol * residual, abstol)
        iteration = 0

        p .= 0

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
Create FFT/DCT Poisson solver from setup.
This solver does FFT in periodic directions, and DCT in Dirichlet directions.
Only works on uniform grids, with periodic/dirichtlet BC.
If there is only Periodic BC, then [`psolver_spectral`](@ref) is faster,
and uses half as much memory.

Warning: This does transform one dimension at a time.
With `Float32` precision, this can lead to large errors.
"""
function psolver_transform(setup)
    (; dimension, Δ, Np, Ip, x, xlims, boundary_conditions) = setup

    D = dimension()
    T = eltype(Δ[1])

    Δx = map(xlims, Np) do l, n
        (l[2] - l[1]) / n
    end

    @assert all(
        bc -> all(b -> b isa PeriodicBC || b isa DirichletBC, bc),
        boundary_conditions.u,
    )
    @assert all(i -> all(≈(Δx[i]), Δ[i][Ip.indices[i]]), eachindex(Δx))
    perdirs = map(bc -> bc[1] isa PeriodicBC, boundary_conditions.u)

    # Fourier transform of the discrete Laplacian
    # Assuming uniform grid, although Δx[1] and Δx[2] do not need to be the same
    ahat = ntuple(D) do i
        n = Np[i]
        k = 0:(n-1)
        h = Δx[i]
        ahat = similar(x[1], n)
        Ω = prod(Δx)
        if perdirs[i]
            @. ahat = -4 * Ω * sinpi(k / n)^2 / h^2
        else
            @. ahat = 2 * Ω * (cospi(k / n) - 1) / h^2
        end
        ahat
    end

    # Placeholders for intermediate results
    # phat = similar(x[1], Complex{T}, Np)
    p = similar(x[1], Np)

    stuff = manual_dct_stuff(p)
    phat = stuff.uhat

    function psolve!(pfull)
        # Buffer of the right size (cannot work on view directly)
        copyto!(p, view(pfull, Ip))

        # Transform of right hand side
        # Do DCTs first, then FFTs
        for i in eachindex(ahat)
            if perdirs[i]
            else
                # dct!(phat, i)
                # dct!(phat, i)
                manual_dct!(p, i, stuff)
            end
        end
        copyto!(phat, p)
        for i in eachindex(ahat)
            if perdirs[i]
                fft!(phat, i)
            else
            end
        end

        # Solve for coefficients in Fourier space
        if D == 2
            ax = reshape(ahat[1], :)
            ay = reshape(ahat[2], 1, :)
            @. phat = phat / (ax + ay)
        else
            ax = reshape(ahat[1], :)
            ay = reshape(ahat[2], 1, :)
            az = reshape(ahat[3], 1, 1, :)
            @. phat = phat / (ax + ay + az)
        end

        # Pressure is determined up to constant. We set this to 0 (instead of
        # phat[1] / 0 = Inf)
        # Note use of singleton range 1:1 instead of scalar index 1
        # (otherwise CUDA gets annoyed)
        phat[1:1] .= 0

        # Inverse transform: FFTs first, then DCTs
        for i in ahat |> eachindex |> reverse
            if perdirs[i]
                ifft!(phat, i)
            else
            end
        end
        @. p = real(phat)
        for i in ahat |> eachindex |> reverse
            if perdirs[i]
            else
                # idct!(phat, i)
                manual_idct!(p, i, stuff)
            end
        end

        # Put results in full size array
        # copyto!(view(p, Ip), phat)
        # view(p, Ip) .= real.(phat)
        copyto!(view(pfull, Ip), p)

        pfull
    end
end

"Create spectral Poisson solver from setup."
function psolver_spectral(setup)
    (; dimension, Δ, Np, Ip, x, boundary_conditions) = setup

    D = dimension()
    T = eltype(Δ[1])

    Δx = first.(Array.(Δ))

    assert_uniform_periodic(setup, "Spectral psolver")

    # Since we use rfft, the first dimension is halved
    kmax = ntuple(α -> α == 1 ? div(Np[α], 2) + 1 : Np[α], D)

    # Fourier transform of the discrete Laplacian
    # Assuming uniform grid, although Δx[1] and Δx[2] do not need to be the same
    ahat = ntuple(D) do α
        k = 0:(kmax[α]-1)
        ahat = similar(x[1], kmax[α])
        Ω = prod(Δx)
        @. ahat = 4 * Ω * sinpi(k / Np[α])^2 / Δx[α]^2
        ahat
    end

    # Placeholders for intermediate results
    phat = similar(x[1], Complex{T}, kmax)
    pI = similar(x[1], Np)
    plan = plan_rfft(pI)

    function psolve!(p)
        # Buffer of the right size (cannot work on view directly)
        copyto!(pI, view(p, Ip))

        # Fourier transform of right hand side
        mul!(phat, plan, pI)

        # Solve for coefficients in Fourier space
        if D == 2
            ax = reshape(ahat[1], :)
            ay = reshape(ahat[2], 1, :)
            @. phat = -phat / (ax + ay)
        else
            ax = reshape(ahat[1], :)
            ay = reshape(ahat[2], 1, :)
            az = reshape(ahat[3], 1, 1, :)
            @. phat = -phat / (ax + ay + az)
        end

        # Pressure is determined up to constant. We set this to 0 (instead of
        # phat[1] / 0 = Inf)
        # Note use of singleton range 1:1 instead of scalar index 1
        # (otherwise CUDA gets annoyed)
        phat[1:1] .= 0

        # Inverse Fourier transform
        ldiv!(pI, plan, phat)

        # Put results in full size array
        copyto!(view(p, Ip), pI)

        p
    end
end

# for fast cg solver with AMGX
# Implemented as an extension

"""
Poisson solver using conjugate gradient method from [AMGX](https://github.com/NVIDIA/AMGX).

Becomes available `using AMGX`.
"""
function psolver_cg_AMGX end

"""
Close all objects created by `amgx_setup`.

Becomes available `using AMGX`.
"""
function close_amgx end

"""
Initializes AMGX, all needed objects are returned in a named tuple.
Needs to be followed by `amgx_close` after use.

Becomes available `using AMGX`.
"""
function amgx_setup end
