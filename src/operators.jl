# Let this be constant for now
const WORKGROUP = 64

# See https://b-fg.github.io/2023/05/07/waterlily-on-gpu.html
# for writing kernel loops

struct Offset{D} end
(::Offset{D})(i) where {D} = CartesianIndex(ntuple(j -> j == i ? 1 : 0, D))

function divergence!(M, u, setup)
    (; boundary_conditions, grid) = setup
    (; Δ, Np, Ip, Ω) = grid
    D = length(u)
    δ = Offset{D}()
    @kernel function _divergence!(M, u, α, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        # D = length(I)
        # δ = Offset{D}()
        M[I] += Ω[I] / Δ[α][I[α]] * (u[α][I] - u[α][I-δ(α)])
    end
    M .= 0
    I0 = first(Ip)
    I0 -= oneunit(I0)
    for α = 1:D
        _divergence!(get_backend(M), WORKGROUP)(M, u, α, I0; ndrange = Np)
        synchronize(get_backend(M))
    end
    M
end

divergence(u, setup) = divergence!(
    KernelAbstractions.zeros(get_backend(u[1]), eltype(u[1]), setup.grid.N...),
    u,
    setup,
)

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
    synchronize(get_backend(ω))
    ω
end

function vorticity!(::Dimension{3}, ω, u, setup)
    (; boundary_conditions, grid) = setup
    (; dimension, Δu, Nu, Iω, Ωω) = grid
    D = dimension()
    δ = Offset{D}()
    @kernel function _vorticity!(ω, u, α, I0; Δu)
        T = eltype(ω)
        I = @index(Global, Cartesian)
        I = I + I0
        β = mod1(α + 1, D)
        γ = mod1(α - 1, D)
        ω[α][I] =
            -Ωω[I] / Δu[γ] * (u[β][I+δ(γ)] - u[β][I]) +
            Ωω[I] / Δu[β] * (u[γ][I+δ(β)] - u[γ][I])
    end
    for α = 1:D
        I0 = first(Iu[α])
        I0 -= oneunit(I0)
        _vorticity!(get_backend(ω), WORKGROUP)(ω, u, α, I0; Δu, ndrange = N .- 1)
        synchronize(get_backend(ω[1]))
    end
    ω
end

function convection!(F, u, setup)
    (; boundary_conditions, grid, Re, bodyforce) = setup
    (; dimension, Δ, Δu, Nu, Iu, Γu, Ωu) = grid
    D = dimension()
    δ = Offset{D}()
    @kernel function _convection!(F, u, α, β, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        Δuαβ = (α == β ? Δu[β] : Δ[β])
        F[α][I] -=
            (
                (u[α][I] + u[α][I+δ(β)]) / 2 * (u[β][I] + u[β][I+δ(α)]) / 2 -
                (u[α][I-δ(β)] + u[α][I]) / 2 * (u[β][I-δ(β)] + u[β][I-δ(β)+δ(α)]) / 2
            ) / Δuαβ[I[β]]
    end
    for α = 1:D
        I0 = first(Iu[α])
        I0 -= oneunit(I0)
        for β = 1:D
            _convection!(get_backend(F[1]), WORKGROUP)(F, u, α, β, I0; ndrange = Nu[α])
            synchronize(get_backend(F[1]))
        end
    end
    F
end

# Add ϵ in denominator for "infinitely thin" volumes
function diffusion!(F, u, setup; ϵ = eps(eltype(F[1])))
    (; boundary_conditions, grid, Re, bodyforce) = setup
    (; dimension, Δ, Δu, Nu, Iu, Ωu, Γu) = grid
    D = dimension()
    δ = Offset{D}()
    ν = 1 / Re
    @kernel function _diffusion!(F, u, α, β, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        Δuαβ = (α == β ? Δu[β] : Δ[β])
        F[α][I] +=
            ν * (
                (u[α][I+δ(β)] - u[α][I]) / ((β == α ? Δ : Δu)[β][I[β]] + ϵ) -
                (u[α][I] - u[α][I-δ(β)]) / ((β == α ? Δ : Δu)[β][(I-δ(β))[β]] + ϵ)
            ) / Δuαβ[I[β]]
    end
    for α = 1:D
        I0 = first(Iu[α])
        I0 -= oneunit(I0)
        for β = 1:D
            _diffusion!(get_backend(F[1]), WORKGROUP)(F, u, α, β, I0; ndrange = Nu[α])
            synchronize(get_backend(F[1]))
        end
    end
    F
end

function bodyforce!(F, u, t, setup)
    (; boundary_conditions, grid, Re, bodyforce) = setup
    (; dimension, Δ, Δu, Nu, Iu, x, xp) = grid
    D = dimension()
    δ = Offset{D}()
    @kernel function _bodyforce!(F, force, α, t, I0)
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
            α,
            t,
            I0;
            ndrange = Nu[α],
        )
        synchronize(get_backend(F[1]))
    end
    F
end

function momentum!(F, u, t, setup)
    (; grid, closure_model) = setup
    (; dimension) = grid
    D = dimension()
    for α = 1:D
        F[α] .= 0
    end
    convection!(F, u, setup)
    diffusion!(F, u, setup)
    bodyforce!(F, u, t, setup)
    isnothing(closure_model) || (F .+= closure_model(u))
    F
end

function pressuregradient!(G, p, setup)
    (; boundary_conditions, grid) = setup
    (; dimension, Δu, Np, Iu) = grid
    D = dimension()
    δ = Offset{D}()
    @kernel function _pressuregradient!(G, p, α, I0)
        I = @index(Global, Cartesian)
        I = I0 + I
        G[α][I] = (p[I+δ(α)] - p[I]) / Δu[α][I[α]]
    end
    D = dimension()
    for α = 1:D
        I0 = first(Iu[α])
        I0 -= oneunit(I0)
        _pressuregradient!(get_backend(G[1]), WORKGROUP)(G, p, α, I0; ndrange = Np)
        synchronize(get_backend(G[1]))
    end
    G
end

pressuregradient(p, setup) = pressuregradient!(
    ntuple(
        α -> KernelAbstractions.zeros(get_backend(p), eltype(p), setup.grid.N),
        setup.grid.dimension(),
    ),
    p,
    setup,
)
