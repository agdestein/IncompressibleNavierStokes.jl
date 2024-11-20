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

"""
Compute pressure from velocity field. This makes the pressure compatible with the velocity
field, resulting in same order pressure as velocity.

Differentiable version.
"""
function pressure(u, temp, t, setup; psolver)
    F = momentum(u, temp, t, setup)
    F = apply_bc_u(F, t, setup; dudt = true)
    div = divergence(F, setup)
    div = scalewithvolume(div, setup)
    p = poisson(psolver, div)
    p = apply_bc_p(p, t, setup)
    p
end

"Compute pressure from velocity field (in-place version)."
function pressure!(p, u, temp, t, setup; psolver, F)
    momentum!(F, u, temp, t, setup)
    apply_bc_u!(F, t, setup; dudt = true)
    divergence!(p, F, setup)
    scalewithvolume!(p, setup)
    poisson!(psolver, p)
    apply_bc_p!(p, t, setup)
    p
end

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
    scalewithvolume!(p, setup)

    # Solve the Poisson equation
    poisson!(psolver, p)
    apply_bc_p!(p, T(0), setup)

    # Apply pressure correction term
    applypressure!(u, p, setup)
end

"Get default Poisson solver from setup."
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

"Create direct Poisson solver using an appropriate matrix decomposition."
psolver_direct(setup) = psolver_direct(setup.grid.x[1], setup) # Dispatch on array type

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
    (; grid, boundary_conditions) = setup
    (; x, Np, Ip) = grid
    T = eltype(x[1])
    L = laplacian_mat(setup)
    isdefinite =
        any(bc -> bc[1] isa PressureBC || bc[2] isa PressureBC, boundary_conditions)
    if isdefinite
        # No extra DOF
        Ttemp = Float64 # This is currently required for SuiteSparse LU
        ftemp = zeros(Ttemp, prod(Np))
        ptemp = zeros(Ttemp, prod(Np))
        viewrange = (:)
        fact = factorize(L)
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

"""
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
    function psolve!(p)
        copyto!(view(ftemp, viewrange), view(view(p, Ip), :))
        cg!(ptemp, L, ftemp; kwargs...)
        copyto!(view(view(p, Ip), :), view(ptemp, viewrange))
        p
    end
end

# Preconditioner
function create_laplace_diag(setup)
    (; grid, workgroupsize) = setup
    (; Δ, Δu, Np, Ip) = grid
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
    abstol = zero(eltype(setup.grid.x[1])),
    reltol = sqrt(eps(eltype(setup.grid.x[1]))),
    maxiter = prod(setup.grid.Np),
    preconditioner = create_laplace_diag(setup),
)
    (; grid, workgroupsize) = setup
    (; Np, Ip) = grid
    T = eltype(setup.grid.x[1])
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

"Create spectral Poisson solver from setup."
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

    # Since we use rfft, the first dimension is halved
    kmax = ntuple(α -> α == 1 ? div(Np[α], 2) + 1 : Np[α], D)

    # Fourier transform of the discrete Laplacian
    # Assuming uniform grid, although Δx[1] and Δx[2] do not need to be the same
    ahat = ntuple(D) do α
        k = 0:kmax[α]-1
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

# Wrap a function to return `nothing`, because Enzyme can not handle vector return values.
function enzyme_wrap(f::typeof(poisson!))
    function wrapped_f(y, args...)
        y .= f(args...)
        return nothing
    end
    return wrapped_f
end
function EnzymeRules.augmented_primal(
    config::RevConfigWidth{1},
    func::Const{typeof(enzyme_wrap(poisson!))},
    ::Type{<:Const},
    y::Duplicated,
    psolver::Const,
    div::Duplicated,
)
    primal = func.val(y.val, psolver.val, div.val)
    return AugmentedReturn(primal, nothing, nothing)
end
function EnzymeRules.reverse(
    config::RevConfigWidth{1},
    func::Const{typeof(enzyme_wrap(poisson!))},
    dret,
    tape,
    y::Duplicated,
    psolver::Const,
    div::Duplicated,
)
    auto_adj = copy(y.val)
    func.val(auto_adj, psolver.val, y.val)
    div.dval .+= auto_adj .* y.dval
    make_zero!(y.dval)
    return (nothing, nothing, nothing)
end
