# See https://b-fg.github.io/2023/05/07/waterlily-on-gpu.html
# for writing kernel loops

struct Offset{D} end
(::Offset{D})(α) where {D} = CartesianIndex(ntuple(β -> β == α ? 1 : 0, D))

"""
    divergence!(M, u, setup)

Compute divergence of velocity field (in-place version).
"""
function divergence!(M, u, setup)
    (; boundary_conditions, grid) = setup
    (; Δ, N, Ip) = grid
    D = length(u)
    δ = Offset{D}()
    @kernel function _divergence!(M, u, ::Val{α}, I0) where {α}
        I = @index(Global, Cartesian)
        I = I + I0
        M[I] += (u[α][I] - u[α][I-δ(α)]) / Δ[α][I[α]]
    end
    M .= 0
    # All volumes have a right velocity
    # All volumes have a left velocity except the first one
    # Start at second volume
    ndrange = N .- 1
    I0 = 2 * oneunit(first(Ip))
    I0 -= oneunit(I0)
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
    (; boundary_conditions, grid) = setup
    (; dimension, Δu, N) = grid
    D = dimension()
    δ = Offset{D}()
    @kernel function _vorticity!(ω, u, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        ω[I] =
            -Δu[1][I[1]] * (u[1][I+δ(2)] - u[1][I]) + Δu[2][I[2]] * (u[2][I+δ(1)] - u[2][I])
    end
    I0 = CartesianIndex(ntuple(Returns(1), D))
    I0 -= oneunit(I0)
    _vorticity!(get_backend(ω), WORKGROUP)(ω, u, I0; ndrange = N .- 1)
    ω
end

function vorticity!(::Dimension{3}, ω, u, setup)
    (; boundary_conditions, grid) = setup
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
    (; boundary_conditions, grid, Re) = setup
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
    (; boundary_conditions, grid, Re) = setup
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
    (; boundary_conditions, grid, Re, bodyforce) = setup
    (; dimension, Δ, Δu, Nu, Iu, x, xp) = grid
    D = dimension()
    δ = Offset{D}()
    @kernel function _bodyforce!(F, force, ::Val{α}, t, I0) where {α}
        I = @index(Global, Cartesian)
        I = I + I0
        F[α][I] += force[α](ntuple(β -> α == β ? x[β][1+I[β]] : xp[β][I[β]], D)..., t)
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
    isnothing(closure_model) || (F .+= closure_model(u))
    F
end

"""
    momentum(u, t, setup)

Right hand side of momentum equations, excluding pressure gradient.
"""
momentum(u, t, setup) = momentum!(
    ntuple(
        α -> KernelAbstractions.zeros(get_backend(u[1]), typeof(t), setup.grid.N),
        length(u),
    ),
    u,
    t,
    setup,
)

"""
    pressuregradient!(G, p, setup)

Compute pressure gradient (in-place).
"""
function pressuregradient!(G, p, setup)
    (; boundary_conditions, grid) = setup
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

interpolate_u_p(setup, u) = interpolate_u_p!(
    setup,
    ntuple(
        α -> KernelAbstractions.zeros(get_backend(u[1]), eltype(u[1]), setup.grid.N),
        setup.grid.dimension(),
    ),
    u,
)

function interpolate_u_p!(setup, up, u)
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

interpolate_ω_p(setup, ω) = interpolate_ω_p!(
    setup,
    setup.grid.dimension() == 2 ?
    KernelAbstractions.zeros(get_backend(ω), eltype(ω), setup.grid.N) :
    ntuple(
        α -> KernelAbstractions.zeros(get_backend(ω[1]), eltype(ω[1]), setup.grid.N),
        setup.grid.dimension(),
    ),
    ω,
)

interpolate_ω_p!(setup, ωp, ω) = interpolate_ω_p!(setup.grid.dimension, setup, ωp, ω)

function interpolate_ω_p!(::Dimension{2}, setup, ωp, ω)
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

function interpolate_ω_p!(::Dimension{3}, setup, ωp, ω)
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
    kinetic_energy(setup, u)

Compute total kinetic energy. The velocity components are interpolated to the
volume centers and squared.
"""
function kinetic_energy(setup, u)
    (; dimension, Ω, Ip) = setup.grid
    D = dimension()
    up = interpolate_u_p(setup, u)
    E = zero(eltype(up[1]))
    for α = 1:D
        # E += sum(I -> Ω[I] * up[α][I]^2, Ip)
        E += sum(Ω[Ip] .* up[α][Ip] .^ 2)
    end
    sqrt(E)
end
