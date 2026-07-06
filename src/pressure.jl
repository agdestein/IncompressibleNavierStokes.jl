# Note: poisson!(...) is a wrapper around psolver!(...) since we need to define
# a shared rrule. This could also be achieved by something like:
#     abstract type AbstractPSolver end
#     rrule(s::AbstractPSolver)(f) = (s(f), φ -> (NoTangent(), s(φ)))
#     (s::AbstractPSolver)(f) = s(zero(...), f)
#     struct PSolver < AbstractPSolver ... end
#     (::PSolver)(p, f) = # solve Poisson

"""
Solve the Poisson equation for the pressure with right hand side `f`.
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
    (; dimension, Δ, Ip, boundary_conditions) = setup
    D = dimension()
    Δ = Array.(Δ)
    Δx = first.(Δ)
    isperiodic =
        map(bc -> bc[1] isa PeriodicBC && bc[2] isa PeriodicBC, boundary_conditions.u)
    iswall =
        map(bc -> bc[1] isa DirichletBC && bc[2] isa DirichletBC, boundary_conditions.u)
    isuniform = map(α -> all(≈(Δ[α][Ip.indices[α][1]]), Δ[α][Ip.indices[α]]), 1:D)
    ischannel = map(1:D) do α
        iswall[α] && all(β -> β == α || isperiodic[β] && isuniform[β], 1:D)
    end
    if all(isperiodic) && all(isuniform)
        psolver_spectral(setup)
    elseif any(ischannel)
        psolver_tridiagonal(setup)
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
    using CUDSS
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
    (; Ip) = setup
    T = eltype(setup.x[1])
    r = scalarfield(setup)
    L = scalarfield(setup)
    q = scalarfield(setup)
    function psolve!(p)
        # Initialize (initial guess is zero, so residual is rhs)
        q .= 0
        r .= p
        ρ_prev = one(T)
        residual = sqrt(sum(abs2, view(r, Ip)))
        tolerance = max(reltol * residual, abstol)
        iteration = 0

        p .= 0

        while iteration < maxiter && residual > tolerance
            preconditioner(L, r)

            ρ = dot(view(L, Ip), view(r, Ip))

            β = ρ / ρ_prev
            q .= L .+ β .* q

            # Periodic/symmetric padding (maybe)
            apply_bc_p!(q, T(0), setup)
            laplacian!(L, q, setup)
            α = ρ / dot(view(q, Ip), view(L, Ip))

            p .+= α .* q
            r .-= α .* L

            ρ_prev = ρ
            residual = sqrt(sum(abs2, view(r, Ip)))

            iteration += 1
        end

        p
    end
end

"""
Create FFT/DCT Poisson solver from setup.
This solver does FFT in periodic directions, and DCT in Dirichlet directions.
Only works on uniform grids, with periodic/dirichlet BC.
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
        _ahat = similar(x[1], n)
        Ω = prod(Δx)
        if perdirs[i]
            @. _ahat = -4 * Ω * sinpi(k / n)^2 / h^2
        else
            @. _ahat = 2 * Ω * (cospi(k / n) - 1) / h^2
        end
        _ahat
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
        copyto!(phat, p) # phat is complex, p is real
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
        @. p = real(phat) # phat is now real in value, but complex in type
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

"""
FFT/tri-diagonal Poisson solver for channel-like setups:
one wall-bounded direction (`DirichletBC` on both sides) and
periodic boundary conditions in the other direction(s).
FFTs in the periodic directions decouple the Poisson equation into an
independent tri-diagonal system along the wall-normal direction for each
Fourier mode. The systems are solved in a batched Thomas-algorithm kernel
(one mode per thread), so the solver works on the GPU as well.

The periodic directions require uniform grid spacing, but the wall-normal
direction may be arbitrarily stretched. This makes this solver a fast direct
method for channel flows on stretched wall-normal grids, where
[`psolver_transform`](@ref) (fully uniform grids) does not apply.

The wall-normal direction is inferred from the boundary conditions.
It can also be chosen explicitly with `dir` (it must still be wall-bounded).
"""
function psolver_tridiagonal(setup; dir = nothing)
    (; dimension, Δ, Δu, Np, Ip, x, boundary_conditions, backend, workgroupsize) = setup
    D = dimension()
    T = eltype(x[1])

    iswall =
        map(bc -> bc[1] isa DirichletBC && bc[2] isa DirichletBC, boundary_conditions.u)
    isperiodic =
        map(bc -> bc[1] isa PeriodicBC && bc[2] isa PeriodicBC, boundary_conditions.u)
    if isnothing(dir)
        dir = findfirst(iswall)
        isnothing(dir) && error(
            "psolver_tridiagonal: no wall-bounded direction " *
            "(DirichletBC on both sides) found",
        )
    end
    iswall[dir] || error(
        "psolver_tridiagonal: direction $dir is not wall-bounded " *
        "(needs DirichletBC on both sides)",
    )
    all(β -> β == dir || isperiodic[β], 1:D) ||
        error("psolver_tridiagonal: all directions except $dir must be periodic")

    # Do coefficient assembly on the CPU
    Δcpu = adapt(Array, Δ)
    Δucpu = adapt(Array, Δu)

    # Uniform grid spacing in the periodic directions
    h = ntuple(β -> Δcpu[β][Ip.indices[β][1]], D)
    for β = 1:D
        β == dir && continue
        all(≈(h[β]), Δcpu[β][Ip.indices[β]]) || error(
            "psolver_tridiagonal: periodic direction $β must have uniform grid spacing",
        )
    end

    # Tri-diagonal wall-normal Laplacian (`a`: subdiagonal, `b`: diagonal,
    # `c`: superdiagonal). The one-sided terms drop out at the walls
    # (homogeneous Neumann pressure BC, see `laplacian!`).
    n = Np[dir]
    iw = Ip.indices[dir]
    a = zeros(T, n)
    b = zeros(T, n)
    c = zeros(T, n)
    for j = 1:n
        i = iw[j]
        j == 1 || (a[j] = 1 / (Δcpu[dir][i] * Δucpu[dir][i-1]))
        j == n || (c[j] = 1 / (Δcpu[dir][i] * Δucpu[dir][i]))
        b[j] = -(a[j] + c[j])
    end

    # Eigenvalues of the periodic Laplacian stencils (modified wavenumbers)
    lam = ntuple(D) do β
        if β == dir
            zeros(T, 1) # Unused placeholder
        else
            k = T.(0:(Np[β]-1))
            @. -4 * sinpi(k / Np[β])^2 / h[β]^2
        end
    end

    # The right hand side comes volume-scaled (`W M u`); strip the scaling
    hper = prod(β -> β == dir ? one(T) : h[β], 1:D)
    invΩ = map(j -> 1 / (hper * Δcpu[dir][iw[j]]), 1:n)
    invΩ = reshape(invΩ, ntuple(β -> β == dir ? n : 1, D))

    (; a, b, c, lam, invΩ) = adapt(backend, (; a, b, c, lam, invΩ))

    # Buffers and FFT plans (one plan per periodic direction)
    pI = similar(x[1], Np)
    phat = similar(x[1], Complex{T}, Np)
    thomas_w = similar(x[1], Np)
    plans = ntuple(D) do β
        β == dir ? nothing : (plan_fft!(phat, β), plan_bfft!(phat, β))
    end
    fftscale = T(prod(β -> β == dir ? 1 : Np[β], 1:D))

    # Thomas algorithm, one tri-diagonal system per Fourier mode.
    # The zero mode is singular (pure Neumann): pin `phat[1] = 0` there.
    @kernel function tridiagonal!(phat, w, a, b, c, lam, ::Val{dir}) where {dir}
        K = @index(Global, Cartesian)
        nj = size(phat, dir)
        mode = ntuple(β -> β < dir ? K[β] : β == dir ? 1 : K[β-1], Val(ndims(phat)))
        idx(j) = CartesianIndex(ntuple(β -> β == dir ? j : mode[β], Val(ndims(phat))))
        λ = zero(eltype(a))
        for β = 1:ndims(phat)
            β == dir || (λ += lam[β][mode[β]])
        end
        if iszero(λ)
            w[idx(1)] = 0
            phat[idx(1)] = 0
        else
            w[idx(1)] = c[1] / (b[1] + λ)
            phat[idx(1)] = phat[idx(1)] / (b[1] + λ)
        end
        for j = 2:nj
            denom = b[j] + λ - a[j] * w[idx(j-1)]
            w[idx(j)] = c[j] / denom
            phat[idx(j)] = (phat[idx(j)] - a[j] * phat[idx(j-1)]) / denom
        end
        for j = (nj-1):-1:1
            phat[idx(j)] -= w[idx(j)] * phat[idx(j+1)]
        end
    end
    ndrange = ntuple(β -> β < dir ? Np[β] : Np[β+1], D - 1)

    function psolve!(p)
        # Buffer of the right size (cannot work on view directly)
        copyto!(pI, view(p, Ip))
        phat .= pI .* invΩ

        # Fourier transform in the periodic directions
        for β = 1:D
            β == dir || plans[β][1] * phat
        end

        # Wall-normal tri-diagonal solve for each Fourier mode
        tridiagonal!(backend, workgroupsize)(
            phat,
            thomas_w,
            a,
            b,
            c,
            lam,
            Val(dir);
            ndrange,
        )

        # Pressure is determined up to a constant: give the zero mode a
        # zero mean, like the other solvers
        mode0 = view(phat, ntuple(β -> β == dir ? Colon() : 1, D)...)
        mode0 .-= sum(mode0) / n

        # Inverse transform (unnormalized; `fftscale` is divided out below)
        for β = 1:D
            β == dir || plans[β][2] * phat
        end
        @. pI = real(phat) / fftscale

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
