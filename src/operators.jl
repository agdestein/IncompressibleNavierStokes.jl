"""
    δ = Offset{D}()

Cartesian index unit vector in `D = 2` or `D = 3` dimensions.
Calling `δ(α)` returns a Cartesian index with `1` in the dimension `α` and zeros
elsewhere.

See <https://b-fg.github.io/2023/05/07/waterlily-on-gpu.html>
for writing kernel loops using Cartesian indices.
"""
struct Offset{D} end

(::Offset{D})(α) where {D} = CartesianIndex(ntuple(β -> β == α ? 1 : 0, D))

"""
    divergence!(M, u, setup)

Compute divergence of velocity field (in-place version).
"""
function divergence!(M, u, setup)
    (; grid) = setup
    (; Δ, N, Ip) = grid
    D = length(u)
    δ = Offset{D}()
    @kernel function _divergence!(M, u, ::Val{α}, I0) where {α}
        I = @index(Global, Cartesian)
        I = I + I0
        M[I] += (u[α][I] - u[α][I-δ(α)]) / Δ[α][I[α]]
    end
    # All volumes have a right velocity
    # All volumes have a left velocity except the first one
    # Start at second volume
    ndrange = N .- 1
    I0 = 2 * oneunit(first(Ip))
    # ndrange = Np
    # I0 = first(Ip)
    I0 -= oneunit(I0)
    M .= 0
    for α = 1:D
        _divergence!(get_backend(M), WORKGROUP)(M, u, Val(α), I0; ndrange)
    end
    M
end

"""
    divergence(u, setup)

Compute divergence of velocity field.
"""
divergence(u, setup) = divergence!(
    KernelAbstractions.zeros(get_backend(u[1]), eltype(u[1]), setup.grid.N...),
    u,
    setup,
)

"""
    vorticity(u, setup)

Compute vorticity field.
"""
vorticity(u, setup) = vorticity!(
    setup.grid.dimension() == 2 ?
    KernelAbstractions.zeros(get_backend(u[1]), eltype(u[1]), setup.grid.N) :
    ntuple(
        α -> KernelAbstractions.zeros(get_backend(u[1]), eltype(u[1]), setup.grid.N),
        setup.grid.dimension(),
    ),
    u,
    setup,
)

"""
    vorticity!(ω, u, setup)

Compute vorticity field.
"""
vorticity!(ω, u, setup) = vorticity!(setup.grid.dimension, ω, u, setup)

function vorticity!(::Dimension{2}, ω, u, setup)
    (; grid) = setup
    (; dimension, Δu, N) = grid
    D = dimension()
    δ = Offset{D}()
    @kernel function _vorticity!(ω, u, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        ω[I] =
            (u[2][I+δ(1)] - u[2][I]) / Δu[1][I[1]] - (u[1][I+δ(2)] - u[1][I]) / Δu[2][I[2]]
    end
    I0 = CartesianIndex(ntuple(Returns(1), D))
    I0 -= oneunit(I0)
    _vorticity!(get_backend(ω), WORKGROUP)(ω, u, I0; ndrange = N .- 1)
    ω
end

function vorticity!(::Dimension{3}, ω, u, setup)
    (; grid) = setup
    (; dimension, Δu, N) = grid
    D = dimension()
    δ = Offset{D}()
    @kernel function _vorticity!(ω, u, ::Val{α}, I0) where {α}
        T = eltype(ω)
        I = @index(Global, Cartesian)
        I = I + I0
        α₊ = mod1(α + 1, D)
        α₋ = mod1(α - 1, D)
        ω[α][I] =
            (u[α₋][I+δ(α₊)] - u[α₋][I]) / Δu[α₊][I[α₊]] -
            (u[α₊][I+δ(α₋)] - u[α₊][I]) / Δu[α₋][I[α₋]]
    end
    I0 = CartesianIndex(ntuple(Returns(1), D))
    I0 -= oneunit(I0)
    for α = 1:D
        _vorticity!(get_backend(ω[1]), WORKGROUP)(ω, u, Val(α), I0; ndrange = N .- 1)
    end
    ω
end

"""
    convection!(F, u, setup)

Compute convective term.
"""
function convection!(F, u, setup)
    (; grid) = setup
    (; dimension, Δ, Δu, Nu, Iu, A) = grid
    D = dimension()
    δ = Offset{D}()
    @kernel function _convection!(F, u, ::Val{α}, ::Val{β}, I0) where {α,β}
        I = @index(Global, Cartesian)
        I = I + I0
        Δuαβ = α == β ? Δu[β] : Δ[β]
        uαβ1 = A[α][β][2][I[β]-1] * u[α][I-δ(β)] + A[α][β][1][I[β]] * u[α][I]
        uαβ2 = A[α][β][2][I[β]] * u[α][I] + A[α][β][1][I[β]+1] * u[α][I+δ(β)]
        uβα1 = A[β][α][2][I[α]-1] * u[β][I-δ(β)] + A[β][α][1][I[α]+1] * u[β][I-δ(β)+δ(α)]
        uβα2 = A[β][α][2][I[α]] * u[β][I] + A[β][α][1][I[α]] * u[β][I+δ(α)]
        F[α][I] -= (uαβ2 * uβα2 - uαβ1 * uβα1) / Δuαβ[I[β]]
    end
    for α = 1:D
        I0 = first(Iu[α])
        I0 -= oneunit(I0)
        for β = 1:D
            _convection!(get_backend(F[1]), WORKGROUP)(
                F,
                u,
                Val(α),
                Val(β),
                I0;
                ndrange = Nu[α],
            )
        end
    end
    F
end

"""
    diffusion!(F, u, setup)

Compute diffusive term.
"""
function diffusion!(F, u, setup)
    (; grid, Re) = setup
    (; dimension, Δ, Δu, Nu, Iu) = grid
    D = dimension()
    δ = Offset{D}()
    ν = 1 / Re
    @kernel function _diffusion!(F, u, ::Val{α}, ::Val{β}, I0) where {α,β}
        I = @index(Global, Cartesian)
        I = I + I0
        Δuαβ = (α == β ? Δu[β] : Δ[β])
        F[α][I] +=
            ν * (
                (u[α][I+δ(β)] - u[α][I]) / (β == α ? Δ[β][I[β]+1] : Δu[β][I[β]]) -
                (u[α][I] - u[α][I-δ(β)]) / (β == α ? Δ[β][I[β]] : Δu[β][I[β]-1])
            ) / Δuαβ[I[β]]
    end
    for α = 1:D
        I0 = first(Iu[α])
        I0 -= oneunit(I0)
        for β = 1:D
            _diffusion!(get_backend(F[1]), WORKGROUP)(
                F,
                u,
                Val(α),
                Val(β),
                I0;
                ndrange = Nu[α],
            )
        end
    end
    F
end

"""
    bodyforce!(F, u, setup)

Compute body force.
"""
function bodyforce!(F, u, t, setup)
    (; grid, bodyforce) = setup
    (; dimension, Δ, Δu, Nu, Iu, x, xp) = grid
    D = dimension()
    δ = Offset{D}()
    @kernel function _bodyforce!(F, force, ::Val{α}, t, I0) where {α}
        I = @index(Global, Cartesian)
        I = I + I0
        F[α][I] +=
            force(Dimension(α), ntuple(β -> α == β ? x[β][1+I[β]] : xp[β][I[β]], D)..., t)
    end
    for α = 1:D
        I0 = first(Iu[α])
        I0 -= oneunit(I0)
        isnothing(bodyforce) || _bodyforce!(get_backend(F[1]), WORKGROUP)(
            F,
            bodyforce,
            Val(α),
            t,
            I0;
            ndrange = Nu[α],
        )
    end
    F
end

"""
    momentum!(F, u, t, setup)

Right hand side of momentum equations, excluding pressure gradient.
Put the result in ``F``.
"""
function momentum!(F, u, t, setup)
    (; grid, closure_model) = setup
    (; dimension) = grid
    D = dimension()
    for α = 1:D
        F[α] .= 0
    end
    diffusion!(F, u, setup)
    convection!(F, u, setup)
    bodyforce!(F, u, t, setup)
    if !isnothing(closure_model)
        m = closure_model(u)
        for α = 1:D
            F[α] .+= m[α]
        end
    end
    F
end

"""
    momentum(u, t, setup)

Right hand side of momentum equations, excluding pressure gradient.
"""
momentum(u, t, setup) = momentum!(zero.(u), u, t, setup)

"""
    pressuregradient!(G, p, setup)

Compute pressure gradient (in-place).
"""
function pressuregradient!(G, p, setup)
    (; grid) = setup
    (; dimension, Δu, Nu, Iu) = grid
    D = dimension()
    δ = Offset{D}()
    @kernel function _pressuregradient!(G, p, ::Val{α}, I0) where {α}
        I = @index(Global, Cartesian)
        I = I0 + I
        G[α][I] = (p[I+δ(α)] - p[I]) / Δu[α][I[α]]
    end
    D = dimension()
    for α = 1:D
        I0 = first(Iu[α])
        I0 -= oneunit(I0)
        _pressuregradient!(get_backend(G[1]), WORKGROUP)(G, p, Val(α), I0; ndrange = Nu[α])
    end
    G
end

"""
    pressuregradient(p, setup)

Compute pressure gradient.
"""
pressuregradient(p, setup) = pressuregradient!(
    ntuple(
        α -> KernelAbstractions.zeros(get_backend(p), eltype(p), setup.grid.N),
        setup.grid.dimension(),
    ),
    p,
    setup,
)

"""
    laplacian!(L, p, setup)

Compute Laplacian of pressure field (in-place version).
"""
function laplacian!(L, p, setup)
    (; grid) = setup
    (; dimension, Δ, Δu, N, Np, Ip, Ω) = grid
    D = dimension()
    δ = Offset{D}()
    @kernel function _laplacian!(L, p, ::Val{α}, I0) where {α}
        I = @index(Global, Cartesian)
        I = I + I0
        L[I] +=
            Ω[I] / Δ[α][I[α]] *
            ((p[I+δ(α)] - p[I]) / Δu[α][I[α]] - (p[I] - p[I-δ(α)]) / Δu[α][I[α]-1])
    end
    L .= 0
    # All volumes have a right velocity
    # All volumes have a left velocity except the first one
    # Start at second volume
    ndrange = Np
    I0 = first(Ip)
    I0 -= oneunit(I0)
    for α = 1:D
        _laplacian!(get_backend(L), WORKGROUP)(L, p, Val(α), I0; ndrange)
    end
    L
end

function laplacian_mat(setup)
    (; grid, boundary_conditions) = setup
    (; dimension, x, N, Np, Ip, Δ, Δu, Ω) = grid
    backend = get_backend(x[1])
    T = eltype(x[1])
    D = dimension()
    δ = Offset{D}()
    Ia = first(Ip)
    Ib = last(Ip)
    I = KernelAbstractions.zeros(backend, CartesianIndex{D}, 0)
    J = KernelAbstractions.zeros(backend, CartesianIndex{D}, 0)
    val = KernelAbstractions.zeros(backend, T, 0)
    I0 = Ia - oneunit(Ia)
    for α = 1:D
        a, b = boundary_conditions[α]
        # i = Ip[ntuple(β -> α == β ? (Ia[α]+1:Ib[α]-1) : (:), D)...][:]
        # ia = Ip[ntuple(β -> α == β ? (Ia[α]:Ia[α]) : (:), D)...][:]
        # ib = Ip[ntuple(β -> α == β ? (Ib[α]:Ib[α]) : (:), D)...][:]
        i = Ip[ntuple(β -> α == β ? (2:Np[α]-1) : (:), D)...][:]
        ia = Ip[ntuple(β -> α == β ? (1:1) : (:), D)...][:]
        ib = Ip[ntuple(β -> α == β ? (Np[α]:Np[α]) : (:), D)...][:]
        for (aa, bb, j) in [(a, nothing, ia), (nothing, nothing, i), (nothing, b, ib)]
            ja = if isnothing(aa)
                j .- δ(α)
            elseif aa isa PressureBC
                error()
            elseif aa isa PeriodicBC
                ib
            elseif aa isa SymmetricBC
                ia
            elseif aa isa DirichletBC
                # The weight of the "left" is zero, but still needs a J in Ip, so
                # just set it to ia
                ia
            end
            jb = if isnothing(bb)
                j .+ δ(α)
            elseif bb isa PressureBC
                error()
            elseif bb isa PeriodicBC
                ia
            elseif bb isa SymmetricBC
                ib
            elseif bb isa DirichletBC
                # The weight of the "right" bc is zero, but still needs a J in Ip, so
                # just set it to ib
                ib
            end
            J = [J; ja; j; jb]
            I = [I; j; j; j]
            # val = vcat(
            #     val,
            #     map(I -> Ω[I] / Δ[α][I[α]] / Δu[α][I[α]-1], j),
            #     map(I -> -Ω[I] / Δ[α][I[α]] * (1 / Δu[α][I[α]] + 1 / Δu[α][I[α]-1]), j),
            #     map(I -> Ω[I] / Δ[α][I[α]] / Δu[α][I[α]], j),
            # )
            val = vcat(
                val,
                @.(Ω[j] / Δ[α][getindex.(j, α)] / Δu[α][getindex.(j, α)-1]),
                @.(
                    -Ω[j] / Δ[α][getindex.(j, α)] *
                    (1 / Δu[α][getindex.(j, α)] + 1 / Δu[α][getindex.(j, α)-1])
                ),
                @.(Ω[j] / Δ[α][getindex.(j, α)] / Δu[α][getindex.(j, α)]),
            )
        end
    end
    # Go back to CPU, otherwise get following error:
    # ERROR: CUDA error: an illegal memory access was encountered (code 700, ERROR_ILLEGAL_ADDRESS)
    I = Array(I)
    J = Array(J)
    I = I .- I0
    J = J .- I0
    # linear = copyto!(KernelAbstractions.zeros(backend, Int, Np), collect(LinearIndices(Ip)))
    linear = LinearIndices(Ip)
    I = linear[I]
    J = linear[J]
    
    # Assemble on CPU, since CUDA overwrites instead of adding
    L = sparse(I, J, Array(val))
    # II = copyto!(KernelAbstractions.zeros(backend, Int, length(I)), I)
    # JJ = copyto!(KernelAbstractions.zeros(backend, Int, length(J)), J)
    # sparse(II, JJ, val)

    Ω isa CuArray ? cu(L) : L
end

"""
    interpolate_u_p(u, setup)

Interpolate velocity to pressure points.
"""
interpolate_u_p(u, setup) = interpolate_u_p!(
    ntuple(
        α -> KernelAbstractions.zeros(get_backend(u[1]), eltype(u[1]), setup.grid.N),
        setup.grid.dimension(),
    ),
    u,
    setup,
)

"""
    interpolate_u_p!(up, u, setup)

Interpolate velocity to pressure points.
"""
function interpolate_u_p!(up, u, setup)
    (; boundary_conditions, grid, Re, bodyforce) = setup
    (; dimension, Np, Ip) = grid
    D = dimension()
    δ = Offset{D}()
    @kernel function _interpolate_u_p!(up, u, ::Val{α}, I0) where {α}
        I = @index(Global, Cartesian)
        I = I + I0
        up[α][I] = (u[α][I-δ(α)] + u[α][I]) / 2
    end
    for α = 1:D
        I0 = first(Ip)
        I0 -= oneunit(I0)
        _interpolate_u_p!(get_backend(up[1]), WORKGROUP)(up, u, Val(α), I0; ndrange = Np)
    end
    up
end

"""
    interpolate_ω_p(ω, setup)

Interpolate vorticity to pressure points.
"""
interpolate_ω_p(ω, setup) = interpolate_ω_p!(
    setup.grid.dimension() == 2 ?
    KernelAbstractions.zeros(get_backend(ω), eltype(ω), setup.grid.N) :
    ntuple(
        α -> KernelAbstractions.zeros(get_backend(ω[1]), eltype(ω[1]), setup.grid.N),
        setup.grid.dimension(),
    ),
    ω,
    setup,
)

"""
    interpolate_ω_p!(ωp, ω, setup)

Interpolate vorticity to pressure points.
"""
interpolate_ω_p!(ωp, ω, setup) = interpolate_ω_p!(setup.grid.dimension, ωp, ω, setup)

function interpolate_ω_p!(::Dimension{2}, ωp, ω, setup)
    (; boundary_conditions, grid, Re, bodyforce) = setup
    (; dimension, Np, Ip) = grid
    D = dimension()
    δ = Offset{D}()
    @kernel function _interpolate_ω_p!(ωp, ω, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        ωp[I] = (ω[I-δ(1)-δ(2)] + ω[I]) / 2
    end
    I0 = first(Ip)
    I0 -= oneunit(I0)
    _interpolate_ω_p!(get_backend(ωp), WORKGROUP)(ωp, ω, I0; ndrange = Np)
    ωp
end

function interpolate_ω_p!(::Dimension{3}, ωp, ω, setup)
    (; boundary_conditions, grid, Re) = setup
    (; dimension, Np, Ip) = grid
    D = dimension()
    δ = Offset{D}()
    @kernel function _interpolate_ω_p!(ωp, ω, ::Val{α}, I0) where {α}
        I = @index(Global, Cartesian)
        I = I + I0
        α₊ = mod1(α + 1, D)
        α₋ = mod1(α - 1, D)
        ωp[α][I] = (ω[α][I-δ(α₊)-δ(α₋)] + ω[α][I]) / 2
    end
    I0 = first(Ip)
    I0 -= oneunit(I0)
    for α = 1:D
        _interpolate_ω_p!(get_backend(ωp[1]), WORKGROUP)(ωp, ω, Val(α), I0; ndrange = Np)
    end
    ωp
end

"""
    Dfield!(d, G, p, setup; ϵ = eps(eltype(p)))

Compute the ``D``-field [LiJiajia2019](@cite) given by

```math
D = \\frac{2 | \\nabla p |}{\\nabla^2 p}.
```
"""
function Dfield!(d, G, p, setup; ϵ = eps(eltype(p)))
    (; boundary_conditions, grid) = setup
    (; dimension, Np, Ip, Δ) = grid
    T = eltype(p)
    D = dimension()
    δ = Offset{D}()
    @kernel function _Dfield!(d, G, p, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        g = zero(eltype(p))
        for α = 1:D
            g += (G[α][I-δ(α)] + G[α][I])^2
        end
        lap = zero(eltype(p))
        # for α = 1:D
        #     lap += (G[α][I] - G[α][I-δ(α)]) / Δ[α][I[α]]
        # end
        if D == 2
            lap += (G[1][I] - G[1][I-δ(1)]) / Δ[1][I[1]]
            lap += (G[2][I] - G[2][I-δ(2)]) / Δ[2][I[2]]
        elseif D == 3
            lap += (G[1][I] - G[1][I-δ(1)]) / Δ[1][I[1]]
            lap += (G[2][I] - G[2][I-δ(2)]) / Δ[2][I[2]]
            lap += (G[3][I] - G[3][I-δ(3)]) / Δ[3][I[3]]
        end
        lap = lap > 0 ? max(lap, ϵ) : min(lap, -ϵ)
        # lap = abs(lap)
        d[I] = sqrt(g) / 2 / lap
    end
    pressuregradient!(G, p, setup)
    I0 = first(Ip)
    I0 -= oneunit(I0)
    _Dfield!(get_backend(p), WORKGROUP)(d, G, p, I0; ndrange = Np)
    d
end


"""
    Dfield(p, setup)

Compute the ``D``-field.
"""
Dfield(p, setup) = Dfield!(
    zero(p),
    ntuple(
        α -> KernelAbstractions.zeros(get_backend(p), eltype(p), setup.grid.N),
        setup.grid.dimension(),
    ),
    p,
    setup,
)

"""
    Qfield!(Q, u, setup; ϵ = eps(eltype(Q)))

Compute ``Q``-field [Jeong1995](@cite) given by

```math
Q = - \\frac{1}{2} \\sum_{α, β} \\frac{\\partial u^α}{\\partial x^β}
\\frac{\\partial u^β}{\\partial x^α}.
```
"""
function Qfield!(Q, u, setup; ϵ = eps(eltype(Q)))
    (; boundary_conditions, grid) = setup
    (; dimension, Np, Ip, Δ) = grid
    D = dimension()
    δ = Offset{D}()
    @kernel function _Qfield!(Q, u, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        q = zero(eltype(Q))
        for α = 1:D, β = 1:D
            q -=
                (u[α][I] - u[α][I-δ(β)]) / Δ[β][I[β]] * (u[β][I] - u[β][I-δ(α)]) /
                Δ[α][I[α]] / 2
        end
        Q[I] = q
    end
    I0 = first(Ip)
    I0 -= oneunit(I0)
    _Qfield!(get_backend(u[1]), WORKGROUP)(Q, u, I0; ndrange = Np)
    Q
end

"""
    Qfield(u, setup)

Compute the ``Q``-field.
"""
Qfield(u, setup) = Qfield!(
    KernelAbstractions.zeros(get_backend(u[1]), eltype(u[1]), setup.grid.N),
    u,
    setup,
)

"""
    kinetic_energy(setup, u)

Compute total kinetic energy. The velocity components are interpolated to the
volume centers and squared.
"""
function kinetic_energy(u, setup)
    (; dimension, Ω, Ip) = setup.grid
    D = dimension()
    up = interpolate_u_p(u, setup)
    E = zero(eltype(up[1]))
    for α = 1:D
        # E += sum(I -> Ω[I] * up[α][I]^2, Ip)
        E += sum(Ω[Ip] .* up[α][Ip] .^ 2)
    end
    sqrt(E)
end
