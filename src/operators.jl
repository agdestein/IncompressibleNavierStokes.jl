# Note on implementation:
# This file contains various differential operators.
#
# Each operator comes with
#
# - an modifying in-place version, e.g. `divergence!(div, u, setup)`,
# - an allocating out-of-place version, e.g. `div = divergence(u, setup)`.
#
# The out-of-place versions can be used as building blocks in a
# Zygote-differentiable program, thanks to the `rrule` methods
# defined.
#
# The domain is divided into `N = (N[1], ..., N[D])` finite volumes.
# These also include ghost volumes, possibly outside the domain.
# For a Cartesian index `I`, volume center fields are naturally in the center,
# but volume face fields are always to the _right_ of volume I.
#
# _All_ fields have the size `N`. These `N` components include
#
# - degrees of freedom
# - boundary values, which are still used, but are filled in separately
# - unused values, which are never used at all. These are still there so that
#   we can guarantee that `ω[I]`, `u[1][I]`, `u[2][I]`, and `p[I]` etc. are
#   at their canonical position in to the volume `I`. Otherwise we would
#   need an offset for each BC type and each combination. Asymptotically
#   speaking (for large `N`), the additional memory footprint of having these
#   around is negligible.
#
# The operators are implemented as kernels.
# The kernels are called for each index in `ndrange`, typically set
# to the degrees of freedom of the output quantity. Boundary values for the
# output quantity are filled in separately, by calling `apply_bc_*` when needed.
# It is assumed that the appropriate boundary values for the input fields are
# already filled in.
#
# The adjoint kernels are written manually for now.
# In the future, Enzyme.jl might be able to do this automatically.

"""
Cartesian index unit vector in `D = 2` or `D = 3` dimensions.
Calling `Offset{D}()(α)` returns a Cartesian index with `1` in the dimension `α` and zeros
elsewhere.

See <https://b-fg.github.io/2023/05/07/waterlily-on-gpu.html>
for writing kernel loops using Cartesian indices.
"""
struct Offset{D} end

@inline (::Offset{D})(α) where {D} = CartesianIndex(ntuple(β -> β == α ? 1 : 0, D))

"""
Average scalar field `ϕ` in the `α`-direction.
"""
@inline function avg(ϕ, Δ, I, α)
    e = Offset{length(I.I)}()
    (Δ[α][I[α]+1] * ϕ[I] + Δ[α][I[α]] * ϕ[I+e(α)]) / (Δ[α][I[α]] + Δ[α][I[α]+1])
end

"""
Compute divergence of velocity field (in-place version).
"""
function divergence!(div, u, setup)
    (; grid, workgroupsize) = setup
    (; Δ, N, Ip, Np) = grid
    D = length(u)
    e = Offset{D}()
    @kernel function div!(div, u, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        d = zero(eltype(div))
        for α = 1:D
            d += (u[α][I] - u[α][I-e(α)]) / Δ[α][I[α]]
        end
        div[I] = d
    end
    I0 = first(Ip)
    I0 -= oneunit(I0)
    div!(get_backend(div), workgroupsize)(div, u, I0; ndrange = Np)
    div
end

function divergence_adjoint!(u, φ, setup)
    (; grid, workgroupsize) = setup
    (; Δ, N, Ip) = grid
    D = length(u)
    e = Offset{D}()
    @kernel function adj!(u, φ)
        I = @index(Global, Cartesian)
        for α = 1:D
            u[α][I] = zero(eltype(u[1]))
            I ∈ Ip && (u[α][I] += φ[I] / Δ[α][I[α]])
            I + e(α) ∈ Ip && (u[α][I] -= φ[I+e(α)] / Δ[α][I[α]+1])
        end
    end
    adj!(get_backend(u[1]), workgroupsize)(u, φ; ndrange = N)
    u
end

"""
Compute divergence of velocity field.
"""
divergence(u, setup) = divergence!(fill!(similar(u[1], setup.grid.N), 0), u, setup)

ChainRulesCore.rrule(::typeof(divergence), u, setup) = (
    divergence(u, setup),
    φ -> (
        NoTangent(),
        Tangent{typeof(u)}(divergence_adjoint!(similar.(u), φ, setup)...),
        NoTangent(),
    ),
)

"""
Compute pressure gradient (in-place).
"""
function pressuregradient!(G, p, setup)
    (; grid, workgroupsize) = setup
    (; dimension, Δu, Nu, Iu) = grid
    D = dimension()
    e = Offset{D}()
    @kernel function G!(G, p, ::Val{α}, I0) where {α}
        I = @index(Global, Cartesian)
        I = I0 + I
        G[α][I] = (p[I+e(α)] - p[I]) / Δu[α][I[α]]
    end
    D = dimension()
    for α = 1:D
        I0 = first(Iu[α])
        I0 -= oneunit(I0)
        G!(get_backend(G[1]), workgroupsize)(G, p, Val(α), I0; ndrange = Nu[α])
    end
    G
end

function pressuregradient_adjoint!(pbar, φ, setup)
    (; grid, workgroupsize) = setup
    (; dimension, Δu, N, Iu) = grid
    D = dimension()
    e = Offset{D}()
    @kernel function adj!(p, φ)
        I = @index(Global, Cartesian)
        p[I] = zero(eltype(p))
        for α = 1:D
            I - e(α) ∈ Iu[α] && (p[I] += φ[α][I-e(α)] / Δu[α][I[α]-1])
            I ∈ Iu[α] && (p[I] -= φ[α][I] / Δu[α][I[α]])
        end
    end
    adj!(get_backend(pbar), workgroupsize)(pbar, φ; ndrange = N)
    pbar
end

"""
Compute pressure gradient.
"""
pressuregradient(p, setup) =
    pressuregradient!(ntuple(α -> zero(p), setup.grid.dimension()), p, setup)

ChainRulesCore.rrule(::typeof(pressuregradient), p, setup) = (
    pressuregradient(p, setup),
    φ -> (NoTangent(), pressuregradient_adjoint!(similar(p), (φ...,), setup), NoTangent()),
)

"""
Subtract pressure gradient (in-place).
"""
function applypressure!(u, p, setup)
    (; grid, workgroupsize) = setup
    (; dimension, Δu, Nu, Iu) = grid
    D = dimension()
    e = Offset{D}()
    @kernel function apply!(u, p, ::Val{α}, I0) where {α}
        I = @index(Global, Cartesian)
        I = I0 + I
        u[α][I] -= (p[I+e(α)] - p[I]) / Δu[α][I[α]]
    end
    D = dimension()
    for α = 1:D
        I0 = first(Iu[α])
        I0 -= oneunit(I0)
        apply!(get_backend(u[1]), workgroupsize)(u, p, Val(α), I0; ndrange = Nu[α])
    end
    u
end

# function applypressure_adjoint!(pbar, φ, u, setup)
#     (; grid, workgroupsize) = setup
#     (; dimension, Δu, N, Iu) = grid
#     D = dimension()
#     e = Offset{D}()
#     @kernel function adj!(p, φ)
#         I = @index(Global, Cartesian)
#         p[I] = zero(eltype(p))
#         for α = 1:D
#             I - e(α) ∈ Iu[α] && (p[I] += φ[α][I-e(α)] / Δu[α][I[α]-1])
#             I ∈ Iu[α] && (p[I] -= φ[α][I] / Δu[α][I[α]])
#         end
#     end
#     adj!(get_backend(pbar), workgroupsize)(pbar, φ; ndrange = N)
#     pbar
# end
#
# """
# Compute pressure gradient.
# """
# applypressure(u, p, setup) =
#     applypressure!(copy.(u), p, setup)
#
# ChainRulesCore.rrule(::typeof(applypressure), p, setup) = (
#     applypressure(u, p, setup),
#     φ -> (NoTangent(), applypressure_adjoint!(similar(p), (φ...,), setup), NoTangent()),
# )

"""
Compute Laplacian of pressure field (in-place version).
"""
function laplacian!(L, p, setup)
    (; grid, workgroupsize, boundary_conditions) = setup
    (; dimension, Δ, Δu, N, Np, Ip, Ω) = grid
    D = dimension()
    e = Offset{D}()
    # @kernel function lap!(L, p, I0)
    #     I = @index(Global, Cartesian)
    #     I = I + I0
    #     lap = zero(eltype(p))
    #     for α = 1:D
    #         # bc = boundary_conditions[α]
    #         if bc[1] isa PressureBC && I[α] == I0[α] + 1
    #             lap +=
    #                 Ω[I] / Δ[α][I[α]] *
    #                 ((p[I+e(α)] - p[I]) / Δu[α][I[α]] - (p[I]) / Δu[α][I[α]-1])
    #         elseif bc[2] isa PressureBC && I[α] == I0[α] + Np[α]
    #             lap +=
    #                 Ω[I] / Δ[α][I[α]] *
    #                 ((-p[I]) / Δu[α][I[α]] - (p[I] - p[I-e(α)]) / Δu[α][I[α]-1])
    #         elseif bc[1] isa DirichletBC && I[α] == I0[α] + 1
    #             lap += Ω[I] / Δ[α][I[α]] * ((p[I+e(α)] - p[I]) / Δu[α][I[α]])
    #         elseif bc[2] isa DirichletBC && I[α] == I0[α] + Np[α]
    #             lap += Ω[I] / Δ[α][I[α]] * (-(p[I] - p[I-e(α)]) / Δu[α][I[α]-1])
    #         else
    #             lap +=
    #                 Ω[I] / Δ[α][I[α]] *
    #                 ((p[I+e(α)] - p[I]) / Δu[α][I[α]] - (p[I] - p[I-e(α)]) / Δu[α][I[α]-1])
    #         end
    #     end
    #     L[I] = lap
    # end
    @kernel function lapα!(L, p, I0, ::Val{α}, bc) where {α}
        I = @index(Global, Cartesian)
        I = I + I0
        # bc = boundary_conditions[α]
        if bc[1] isa PressureBC && I[α] == I0[α] + 1
            L[I] +=
                Ω[I] / Δ[α][I[α]] *
                ((p[I+e(α)] - p[I]) / Δu[α][I[α]] - (p[I]) / Δu[α][I[α]-1])
        elseif bc[2] isa PressureBC && I[α] == I0[α] + Np[α]
            L[I] +=
                Ω[I] / Δ[α][I[α]] *
                ((-p[I]) / Δu[α][I[α]] - (p[I] - p[I-e(α)]) / Δu[α][I[α]-1])
        elseif bc[1] isa DirichletBC && I[α] == I0[α] + 1
            L[I] += Ω[I] / Δ[α][I[α]] * ((p[I+e(α)] - p[I]) / Δu[α][I[α]])
        elseif bc[2] isa DirichletBC && I[α] == I0[α] + Np[α]
            L[I] += Ω[I] / Δ[α][I[α]] * (-(p[I] - p[I-e(α)]) / Δu[α][I[α]-1])
        else
            L[I] +=
                Ω[I] / Δ[α][I[α]] *
                ((p[I+e(α)] - p[I]) / Δu[α][I[α]] - (p[I] - p[I-e(α)]) / Δu[α][I[α]-1])
        end
        # L[I] = lap
    end
    # All volumes have a right velocity
    # All volumes have a left velocity except the first one
    # Start at second volume
    ndrange = Np
    I0 = first(Ip)
    I0 -= oneunit(I0)
    # lap!(get_backend(L), workgroupsize)(L, p, I0; ndrange)
    L .= 0
    for α = 1:D
        lapα!(get_backend(L), workgroupsize)(
            L,
            p,
            I0,
            Val(α),
            boundary_conditions[α];
            ndrange,
        )
    end
    L
end

"""
Compute Laplacian of pressure field.
"""
laplacian(p, setup) = laplacian!(similar(p), p, setup)

"""
Compute convective term.
"""
function convection!(F, u, setup)
    (; grid, workgroupsize) = setup
    (; dimension, Δ, Δu, Nu, Iu, A) = grid
    D = dimension()
    e = Offset{D}()
    @kernel function conv!(F, u, ::Val{α}, ::Val{βrange}, I0) where {α,βrange}
        I = @index(Global, Cartesian)
        I = I + I0
        # for β = 1:D
        KernelAbstractions.Extras.LoopInfo.@unroll for β in βrange
            Δuαβ = α == β ? Δu[β] : Δ[β]

            # Half for u[α], (reverse!) interpolation for u[β]
            # Note:
            #     In matrix version, uses
            #     1*u[α][I-e(β)] + 0*u[α][I]
            #     instead of 1/2 when u[α][I-e(β)] is at Dirichlet boundary.
            uαβ1 = (u[α][I-e(β)] + u[α][I]) / 2
            uαβ2 = (u[α][I] + u[α][I+e(β)]) / 2
            uβα1 =
                A[β][α][2][I[α]-(α==β)] * u[β][I-e(β)] +
                A[β][α][1][I[α]+(α!=β)] * u[β][I-e(β)+e(α)]
            uβα2 = A[β][α][2][I[α]] * u[β][I] + A[β][α][1][I[α]+1] * u[β][I+e(α)]
            F[α][I] -= (uαβ2 * uβα2 - uαβ1 * uβα1) / Δuαβ[I[β]]
        end
    end
    for α = 1:D
        I0 = first(Iu[α])
        I0 -= oneunit(I0)
        conv!(get_backend(F[1]), workgroupsize)(F, u, Val(α), Val(1:D), I0; ndrange = Nu[α])
    end
    F
end

function convection_adjoint!(ubar, φbar, u, setup)
    (; grid, workgroupsize) = setup
    (; dimension, Δ, Δu, N, Iu, A) = grid
    D = dimension()
    e = Offset{D}()
    T = eltype(u[1])
    h = T(1) / 2
    @kernel function adj!(ubar, φbar, u, ::Val{γ}, ::Val{looprange}) where {γ,looprange}
        J = @index(Global, Cartesian)
        KernelAbstractions.Extras.LoopInfo.@unroll for α in looprange
            KernelAbstractions.Extras.LoopInfo.@unroll for β in looprange
                Δuαβ = α == β ? Δu[β] : Δ[β]
                Aβα1 = A[β][α][1]
                Aβα2 = A[β][α][2]

                # 1
                I = J
                if α == γ && I in Iu[α]
                    uαβ2 = h
                    uβα2 = Aβα2[I[α]] * u[β][I] + Aβα1[I[α]+1] * u[β][I+e(α)]
                    dφdu = -uαβ2 * uβα2 / Δuαβ[I[β]]
                    ubar[γ][J] += φbar[α][I] * dφdu
                end

                # 2
                I = J - e(β)
                if α == γ && I in Iu[α]
                    uαβ2 = h
                    uβα2 = Aβα2[I[α]] * u[β][I] + Aβα1[I[α]+1] * u[β][I+e(α)]
                    dφdu = -uαβ2 * uβα2 / Δuαβ[I[β]]
                    ubar[γ][J] += φbar[α][I] * dφdu
                end

                # 3
                I = J
                if β == γ && I in Iu[α]
                    uαβ2 = h * u[α][I] + h * u[α][I+e(β)]
                    uβα2 = Aβα2[I[α]]
                    dφdu = -uαβ2 * uβα2 / Δuαβ[I[β]]
                    ubar[γ][J] += φbar[α][I] * dφdu
                end

                # 4
                I = J - e(α)
                if β == γ && I in Iu[α]
                    uαβ2 = h * u[α][I] + h * u[α][I+e(β)]
                    uβα2 = Aβα1[I[α]+1]
                    dφdu = -uαβ2 * uβα2 / Δuαβ[I[β]]
                    ubar[γ][J] += φbar[α][I] * dφdu
                end

                # 5
                I = J + e(β)
                if α == γ && I in Iu[α]
                    uαβ1 = h
                    uβα1 =
                        Aβα2[I[α]-(α==β)] * u[β][I-e(β)] +
                        Aβα1[I[α]+(α!=β)] * u[β][I-e(β)+e(α)]
                    dφdu = uαβ1 * uβα1 / Δuαβ[I[β]]
                    ubar[γ][J] += φbar[α][I] * dφdu
                end

                # 6
                I = J
                if α == γ && I in Iu[α]
                    uαβ1 = h
                    uβα1 =
                        Aβα2[I[α]-(α==β)] * u[β][I-e(β)] +
                        Aβα1[I[α]+(α!=β)] * u[β][I-e(β)+e(α)]
                    dφdu = uαβ1 * uβα1 / Δuαβ[I[β]]
                    ubar[γ][J] += φbar[α][I] * dφdu
                end

                # 7
                I = J + e(β)
                if β == γ && I in Iu[α]
                    uαβ1 = h * u[α][I-e(β)] + h * u[α][I]
                    uβα1 = Aβα2[I[α]-(α==β)]
                    dφdu = uαβ1 * uβα1 / Δuαβ[I[β]]
                    ubar[γ][J] += φbar[α][I] * dφdu
                end

                # 8
                I = J + e(β) - e(α)
                if β == γ && I in Iu[α]
                    uαβ1 = h * u[α][I-e(β)] + h * u[α][I]
                    uβα1 = Aβα1[I[α]+(α!=β)]
                    dφdu = uαβ1 * uβα1 / Δuαβ[I[β]]
                    ubar[γ][J] += φbar[α][I] * dφdu
                end
            end
        end
    end
    for γ = 1:D
        adj!(get_backend(u[1]), workgroupsize)(ubar, φbar, u, Val(γ), Val(1:D); ndrange = N)
    end
    ubar
end

convection(u, setup) = convection!(zero.(u), u, setup)

ChainRulesCore.rrule(::typeof(convection), u, setup) = (
    convection(u, setup),
    φ -> (
        NoTangent(),
        Tangent{typeof(u)}(convection_adjoint!(zero.(u), (φ...,), u, setup)...),
        NoTangent(),
    ),
)

"""
Compute diffusive term.
"""
function diffusion!(F, u, setup)
    (; grid, workgroupsize, Re) = setup
    (; dimension, Δ, Δu, Nu, Iu) = grid
    D = dimension()
    e = Offset{D}()
    ν = 1 / Re
    @kernel function diff!(F, u, ::Val{α}, ::Val{βrange}, I0) where {α,βrange}
        I = @index(Global, Cartesian)
        I = I + I0
        KernelAbstractions.Extras.LoopInfo.@unroll for β in βrange
            Δuαβ = α == β ? Δu[β] : Δ[β]
            Δa = β == α ? Δ[β][I[β]] : Δu[β][I[β]-1]
            Δb = β == α ? Δ[β][I[β]+1] : Δu[β][I[β]]
            ∂a = (u[α][I] - u[α][I-e(β)]) / Δa
            ∂b = (u[α][I+e(β)] - u[α][I]) / Δb
            F[α][I] += ν * (∂b - ∂a) / Δuαβ[I[β]]
        end
    end
    for α = 1:D
        I0 = first(Iu[α])
        I0 -= oneunit(I0)
        diff!(get_backend(F[1]), workgroupsize)(F, u, Val(α), Val(1:D), I0; ndrange = Nu[α])
    end
    F
end

function diffusion_adjoint!(u, φ, setup)
    (; grid, workgroupsize, Re) = setup
    (; dimension, N, Δ, Δu, Iu) = grid
    D = dimension()
    e = Offset{D}()
    ν = 1 / Re
    @kernel function adj!(u, φ, ::Val{α}, ::Val{βrange}) where {α,βrange}
        I = @index(Global, Cartesian)
        KernelAbstractions.Extras.LoopInfo.@unroll for β in βrange
            Δuαβ = α == β ? Δu[β] : Δ[β]
            # F[α][I] += ν * u[α][I+e(β)] / (β == α ? Δ[β][I[β]+1] : Δu[β][I[β]])
            # F[α][I] -= ν * u[α][I] / (β == α ? Δ[β][I[β]+1] : Δu[β][I[β]])
            # F[α][I] -= ν * u[α][I] / (β == α ? Δ[β][I[β]] : Δu[β][I[β]-1])
            # F[α][I] += ν * u[α][I-e(β)] / (β == α ? Δ[β][I[β]] : Δu[β][I[β]-1])
            val = zero(eltype(u[1]))
            if I - e(β) ∈ Iu[α]
                val +=
                    ν * φ[α][I-e(β)] / (β == α ? Δ[β][I[β]] : Δu[β][I[β]-1]) / Δuαβ[I[β]-1]
            end
            if I ∈ Iu[α]
                val -= ν * φ[α][I] / (β == α ? Δ[β][I[β]+1] : Δu[β][I[β]]) / Δuαβ[I[β]]
            end
            if I ∈ Iu[α]
                val -= ν * φ[α][I] / (β == α ? Δ[β][I[β]] : Δu[β][I[β]-1]) / Δuαβ[I[β]]
            end
            if I + e(β) ∈ Iu[α]
                val +=
                    ν * φ[α][I+e(β)] / (β == α ? Δ[β][I[β]+1] : Δu[β][I[β]]) / Δuαβ[I[β]+1]
            end
            u[α][I] += val
        end
    end
    for α = 1:D
        adj!(get_backend(u[1]), workgroupsize)(u, φ, Val(α), Val(1:D); ndrange = N)
    end
    u
end

diffusion(u, setup) = diffusion!(zero.(u), u, setup)

ChainRulesCore.rrule(::typeof(diffusion), u, setup) = (
    diffusion(u, setup),
    φ -> (
        NoTangent(),
        Tangent{typeof(u)}(diffusion_adjoint!(zero.(u), (φ...,), setup)...),
        NoTangent(),
    ),
)

function convectiondiffusion!(F, u, setup)
    (; grid, workgroupsize, Re) = setup
    (; dimension, Δ, Δu, Nu, Iu, A) = grid
    D = dimension()
    e = Offset{D}()
    ν = 1 / Re
    @kernel function cd!(F, u, ::Val{α}, ::Val{βrange}, I0) where {α,βrange}
        I = @index(Global, Cartesian)
        I = I + I0
        KernelAbstractions.Extras.LoopInfo.@unroll for β in βrange
            Δuαβ = α == β ? Δu[β] : Δ[β]
            uαβ1 = (u[α][I-e(β)] + u[α][I]) / 2
            uαβ2 = (u[α][I] + u[α][I+e(β)]) / 2
            uβα1 =
                A[β][α][2][I[α]-(α==β)] * u[β][I-e(β)] +
                A[β][α][1][I[α]+(α!=β)] * u[β][I-e(β)+e(α)]
            uβα2 = A[β][α][2][I[α]] * u[β][I] + A[β][α][1][I[α]+1] * u[β][I+e(α)]
            uαuβ1 = uαβ1 * uβα1
            uαuβ2 = uαβ2 * uβα2
            ∂βuα1 = (u[α][I] - u[α][I-e(β)]) / (β == α ? Δ[β][I[β]] : Δu[β][I[β]-1])
            ∂βuα2 = (u[α][I+e(β)] - u[α][I]) / (β == α ? Δ[β][I[β]+1] : Δu[β][I[β]])
            F[α][I] += (ν * (∂βuα2 - ∂βuα1) - (uαuβ2 - uαuβ1)) / Δuαβ[I[β]]
        end
    end
    for α = 1:D
        I0 = first(Iu[α])
        I0 -= oneunit(I0)
        cd!(get_backend(F[1]), workgroupsize)(F, u, Val(α), Val(1:D), I0; ndrange = Nu[α])
    end
    F
end

# convectiondiffusion(u, setup) = convectiondiffusion!(zero.(u), u, setup)
#
# ChainRulesCore.rrule(::typeof(convectiondiffusion), u, setup) = (
#     convection(u, setup),
#     φ -> (
#         NoTangent(),
#         Tangent{typeof(u)}(convectiondiffusion_adjoint!(similar.(u), φ, setup)...),
#         NoTangent(),
#     ),
# )

"""
Compute convection-diffusion term for the temperature equation.
Add result to `c`.
"""
function convection_diffusion_temp!(c, u, temp, setup)
    (; grid, workgroupsize, temperature) = setup
    (; dimension, Δ, Δu, Np, Ip) = grid
    (; α4) = temperature
    D = dimension()
    e = Offset{D}()
    @kernel function conv!(c, u, temp, ::Val{βrange}, I0) where {βrange}
        I = @index(Global, Cartesian)
        I = I + I0
        cI = zero(eltype(c))
        KernelAbstractions.Extras.LoopInfo.@unroll for β in βrange
            ∂T∂x1 = (temp[I] - temp[I-e(β)]) / Δu[β][I[β]-1]
            ∂T∂x2 = (temp[I+e(β)] - temp[I]) / Δu[β][I[β]]
            uT1 = u[β][I-e(β)] * avg(temp, Δ, I - e(β), β)
            uT2 = u[β][I] * avg(temp, Δ, I, β)
            cI += (-(uT2 - uT1) + α4 * (∂T∂x2 - ∂T∂x1)) / Δ[β][I[β]]
        end
        c[I] += cI
    end
    I0 = first(Ip)
    I0 -= oneunit(I0)
    conv!(get_backend(c), workgroupsize)(c, u, temp, Val(1:D), I0; ndrange = Np)
    c
end

convection_diffusion_temp(u, temp, setup) =
    convection_diffusion_temp!(zero.(temp), u, temp, setup)

function ChainRulesCore.rrule(::typeof(convection_diffusion_temp), u, temp, setup)
    conv = convection_diffusion_temp(u, temp, setup)
    convection_diffusion_temp_pullback(φ) = (NoTangent(), du, dtemp, NoTangent())
    (conv, pullback)
end

# function convection_diffusion_temp_pullback(u, temp, setup)
#     (; grid, workgroupsize, temperature) = setup
#     (; dimension, Δ, Δu, Np, Ip) = grid
#     (; α4) = temperature
#     D = dimension()
#     e = Offset{D}()
#     @kernel function pullback_u!(du, φ, u, temp, ::Val{βrange}, I0) where {βrange}
#         I = @index(Global, Cartesian)
#         I = I + I0
#         cI = zero(eltype(c))
#         KernelAbstractions.Extras.LoopInfo.@unroll for β in βrange
#             ∂T∂x1 = (temp[I] - temp[I-e(β)]) / Δu[β][I[β]-1]
#             ∂T∂x2 = (temp[I+e(β)] - temp[I]) / Δu[β][I[β]]
#             uT1 = u[β][I-e(β)] * avg(temp, Δ, I - e(β), β)
#             uT2 = u[β][I] * avg(temp, Δ, I, β)
#             cI += (-(uT2 - uT1) + α4 * (∂T∂x2 - ∂T∂x1)) / Δ[β][I[β]]
#         end
#         c[I] = cI
#     end
#     I0 = first(Ip)
#     I0 -= oneunit(I0)
#     conv!(get_backend(c), workgroupsize)(c, u, temp, Val(1:D), I0; ndrange = Np)
#     function pullback(φ)
#         (NoTangent(), du, dtemp, NoTangent())
#     end
# end

# function dissipation!(c, u, setup)
#     (; grid, workgroupsize, temperature) = setup
#     (; dimension, Δ, Np, Ip) = grid
#     D = dimension()
#     e = Offset{D}()
#     @inline ∂2(u, α, β, I) = ((u[α][I+e(β)] - u[α][I]) / Δ[β][I])^2 / 2
#     @inline Φ(u, α, β, I) = -∂2(u, α, β, I) - ∂2(u, α, β, I+e(β))
#     @kernel function diss!(d, u, ::Val{βrange}, I0) where {βrange}
#         I = @index(Global, Cartesian)
#         I = I + I0
#         cI = zero(eltype(c))
#         KernelAbstractions.Extras.LoopInfo.@unroll for β in βrange
#             cI += Φ(u, β, β, I) / Δ[β][I[β]]
#         end
#         c[I] += cI
#     end
# end

"""
Compute dissipation term for the temperature equation.
Add result to `diss`.
"""
function dissipation!(diss, diff, u, setup)
    (; grid, workgroupsize, Re, temperature) = setup
    (; dimension, Δ, Np, Ip) = grid
    (; α1, γ) = temperature
    D = dimension()
    e = Offset{D}()
    fill!.(diff, 0)
    diffusion!(diff, u, setup)
    @kernel function interpolate!(diss, diff, u, I0, ::Val{βrange}) where {βrange}
        I = @index(Global, Cartesian)
        I += I0
        d = zero(eltype(diss))
        KernelAbstractions.Extras.LoopInfo.@unroll for β in βrange
            d += Re * α1 / γ * (u[β][I-e(β)] * diff[β][I-e(β)] + u[β][I] * diff[β][I]) / 2
        end
        diss[I] += d
    end
    I0 = first(Ip)
    I0 -= oneunit(I0)
    interpolate!(get_backend(diss), workgroupsize)(
        diss,
        diff,
        u,
        I0,
        Val(1:D);
        ndrange = Np,
    )
    diss
end

dissipation(u, setup) = dissipation!(zero(u[1]), zero.(u), u, setup)

function ChainRulesCore.rrule(::typeof(dissipation), u, setup)
    (; grid, workgroupsize, Re, temperature) = setup
    (; dimension, Δ, N, Np, Ip) = grid
    (; α1, γ) = temperature
    D = dimension()
    e = Offset{D}()
    backend = get_backend(u[1])
    d, d_pb = ChainRulesCore.rrule(diffusion, u, setup)
    φ = dissipation!(zero(u[1]), d, u, setup)
    @kernel function ∂φ!(ubar, dbar, φbar, d, u, ::Val{βrange}) where {βrange}
        J = @index(Global, Cartesian)
        KernelAbstractions.Extras.LoopInfo.@unroll for β in βrange
            # Compute ubar
            a = zero(eltype(u[1]))
            # 1
            I = J + e(β)
            I ∈ Ip && (a += Re * α1 / γ * d[β][I-e(β)] / 2)
            # 2
            I = J
            I ∈ Ip && (a += Re * α1 / γ * d[β][I] / 2)
            ubar[β][J] += a

            # Compute dbar
            b = zero(eltype(u[1]))
            # 1
            I = J + e(β)
            I ∈ Ip && (b += Re * α1 / γ * u[β][I-e(β)] / 2)
            # 2
            I = J
            I ∈ Ip && (b += Re * α1 / γ * u[β][I] / 2)
            dbar[β][J] += b
        end
    end
    function dissipation_pullback(φbar)
        # Dφ/Du = ∂φ(u, d)/∂u + ∂φ(u, d)/∂d ⋅ ∂d(u)/∂u
        dbar = zero.(u)
        ubar = zero.(u)
        ∂φ!(backend, workgroupsize)(ubar, dbar, φbar, d, u, Val(1:D); ndrange = N)
        diffusion_adjoint!(ubar, dbar, setup)
        ubar = Tangent{typeof(u)}(ubar...)
        (NoTangent(), ubar, NoTangent())
    end
    φ, dissipation_pullback
end

"""
Compute dissipation term
``2 \\nu \\langle S_{i j} S_{i j} \\rangle``
from strain-rate tensor.
"""
function dissipation_from_strain!(ϵ, u, setup)
    (; grid, workgroupsize, Re) = setup
    (; dimension, Δ, Δu, Np, Ip) = grid
    D = dimension()
    e = Offset{D}()
    ν = 1 / Re
    @kernel function diss!(ϵ, u, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        S = strain(u, I, Δ, Δu)
        ϵ[I] = 2 * ν * sum(S .* S)
    end
    I0 = first(Ip)
    I0 -= oneunit(I0)
    diss!(get_backend(u[1]), workgroupsize)(ϵ, u, I0; ndrange = Np)
    ϵ
end

dissipation_from_strain(u, setup) = dissipation_from_strain!(zero(u[1]), u, setup)

"""
Compute body force.
"""
function bodyforce!(F, u, t, setup)
    (; grid, workgroupsize, bodyforce, issteadybodyforce) = setup
    (; dimension, Δ, Δu, Nu, Iu, x, xp) = grid
    isnothing(bodyforce) && return F
    D = dimension()
    e = Offset{D}()
    @assert D == 2
    @kernel function f!(F, ::Val{α}, t, I0) where {α}
        I = @index(Global, Cartesian)
        I = I + I0
        # xI = ntuple(β -> α == β ? x[β][1+I[β]] : xp[β][I[β]], D)
        xI = (
            α == 1 ? x[1][1+I[1]] : xp[1][I[1]],
            α == 2 ? x[2][1+I[2]] : xp[2][I[2]],
            # α == 3 ? x[3][1+I[3]] : xp[3][I[3]],
        )
        F[α][I] += bodyforce(Dimension(α), xI..., t)
    end
    for α = 1:D
        I0 = first(Iu[α])
        I0 -= oneunit(I0)
        if issteadybodyforce
            F[α] .+= bodyforce[α]
        else
            f!(get_backend(F[1]), workgroupsize)(F, Val(α), t, I0; ndrange = Nu[α])
        end
    end
    F
end
bodyforce(u, t, setup) = bodyforce!(zero.(u), u, t, setup)

ChainRulesCore.rrule(::typeof(bodyforce), u, t, setup) =
    (bodyforce(u, t, setup), φ -> (NoTangent(), ZeroTangent(), NoTangent(), NoTangent()))

"""
Compute gravity term (add to existing `F`).
"""
function gravity!(F, temp, setup)
    (; grid, workgroupsize, temperature) = setup
    (; dimension, Δ, Nu, Iu) = grid
    (; gdir, α2) = temperature
    D = dimension()
    e = Offset{D}()
    @kernel function g!(F, temp, ::Val{gdir}, I0) where {gdir}
        I = @index(Global, Cartesian)
        I = I + I0
        F[gdir][I] += α2 * avg(temp, Δ, I, gdir)
    end
    I0 = first(Iu[gdir])
    I0 -= oneunit(I0)
    g!(get_backend(F[1]), workgroupsize)(F, temp, Val(gdir), I0; ndrange = Nu[gdir])
    F
end

gravity(temp, setup) =
    gravity!(ntuple(α -> zero(temp), setup.grid.dimension()), temp, setup)

function ChainRulesCore.rrule(::typeof(gravity), temp, setup)
    (; grid, workgroupsize, temperature) = setup
    (; dimension, Δ, N, Iu) = grid
    (; gdir, α2) = temperature
    backend = get_backend(temp)
    D = dimension()
    e = Offset{D}()
    g = gravity(temp, setup)
    function gravity_pullback(φ)
        @kernel function g!(tempbar, φbar, ::Val{α}) where {α}
            J = @index(Global, Cartesian)
            t = zero(eltype(tempbar))
            # 1
            I = J
            I ∈ Iu[α] && (t += α2 * Δ[α][I[α]+1] * φbar[α][I] / (Δ[α][I[α]] + Δ[α][I[α]+1]))
            # 2
            I = J - e(α)
            I ∈ Iu[α] && (t += α2 * Δ[α][I[α]] * φbar[α][I] / (Δ[α][I[α]] + Δ[α][I[α]+1]))
            tempbar[J] = t
        end
        tempbar = zero(temp)
        g!(backend, workgroupsize)(tempbar, φ, Val(gdir); ndrange = N)
        (NoTangent(), tempbar, NoTangent())
    end
    g, gravity_pullback
end

"""
Right hand side of momentum equations, excluding pressure gradient.
Put the result in ``F``.
"""
function momentum!(F, u, temp, t, setup)
    (; grid, closure_model, temperature) = setup
    (; dimension) = grid
    D = dimension()
    for α = 1:D
        F[α] .= 0
    end
    # diffusion!(F, u, setup)
    # convection!(F, u, setup)
    convectiondiffusion!(F, u, setup)
    bodyforce!(F, u, t, setup)
    isnothing(temp) || gravity!(F, temp, setup)
    F
end

# monitor(u) = (@info("Forward", typeof(u)); u)
# ChainRulesCore.rrule(::typeof(monitor), u) =
#     (monitor(u), φ -> (@info("Reverse", typeof(φ)); (NoTangent(), φ)))

# tupleadd(u...) = ntuple(α -> sum(u -> u[α], u), length(u[1]))
# ChainRulesCore.rrule(::typeof(tupleadd), u...) =
#     (tupleadd(u...), φ -> (NoTangent(), map(u -> φ, u)...))

"""
Right hand side of momentum equations, excluding pressure gradient.
"""
function momentum(u, temp, t, setup)
    (; grid, closure_model) = setup
    (; dimension) = grid
    D = dimension()
    d = diffusion(u, setup)
    c = convection(u, setup)
    f = bodyforce(u, t, setup)
    # F = ntuple(D) do α
    #     d[α] .+ c[α] .+ f[α]
    # end
    F = @. d + c + f
    # F = tupleadd(d, c, f)
    if !isnothing(temp)
        g = gravity(temp, setup)
        F = @. F + g
    end
    F
end

# ChainRulesCore.rrule(::typeof(momentum), u, temp, t, setup) = (
#     (error(); momentum(u, temp, t, setup)),
#     φ -> (
#         NoTangent(),
#         Tangent{typeof(u)}(momentum_pullback!(zero.(φ), φ, u, temp, t, setup)...),
#         NoTangent(),
#         NoTangent(),
#     ),
# )

"""
Compute vorticity field.
"""
vorticity(u, setup) = vorticity!(
    length(u) == 2 ? similar(u[1], setup.grid.N) :
    ntuple(α -> similar(u[1], setup.grid.N), length(u)),
    u,
    setup,
)

"""
Compute vorticity field.
"""
vorticity!(ω, u, setup) = vorticity!(setup.grid.dimension, ω, u, setup)

function vorticity!(::Dimension{2}, ω, u, setup)
    (; grid, workgroupsize) = setup
    (; dimension, Δu, N) = grid
    D = dimension()
    e = Offset{D}()
    @kernel function ω!(ω, u, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        ω[I] =
            (u[2][I+e(1)] - u[2][I]) / Δu[1][I[1]] - (u[1][I+e(2)] - u[1][I]) / Δu[2][I[2]]
    end
    I0 = CartesianIndex(ntuple(Returns(1), D))
    I0 -= oneunit(I0)
    ω!(get_backend(ω), workgroupsize)(ω, u, I0; ndrange = N .- 1)
    ω
end

function vorticity!(::Dimension{3}, ω, u, setup)
    (; grid, workgroupsize) = setup
    (; dimension, Δu, N) = grid
    D = dimension()
    e = Offset{D}()
    @kernel function ω!(ω, u, I0)
        T = eltype(ω)
        I = @index(Global, Cartesian)
        I = I + I0
        for (α, α₊, α₋) in ((1, 2, 3), (2, 3, 1), (3, 1, 2))
            # α₊ = mod1(α + 1, D)
            # α₋ = mod1(α - 1, D)
            ω[α][I] =
                (u[α₋][I+e(α₊)] - u[α₋][I]) / Δu[α₊][I[α₊]] -
                (u[α₊][I+e(α₋)] - u[α₊][I]) / Δu[α₋][I[α₋]]
        end
    end
    I0 = CartesianIndex(ntuple(Returns(1), D))
    I0 -= oneunit(I0)
    ω!(get_backend(ω[1]), workgroupsize)(ω, u, I0; ndrange = N .- 1)
    ω
end

@inline ∂x(uα, I::CartesianIndex{D}, α, β, Δβ, Δuβ; e = Offset{D}()) where {D} =
    α == β ? (uα[I] - uα[I-e(β)]) / Δβ[I[β]] :
    (
        (uα[I+e(β)] - uα[I]) / Δuβ[I[β]] +
        (uα[I-e(α)+e(β)] - uα[I-e(α)]) / Δuβ[I[β]] +
        (uα[I] - uα[I-e(β)]) / Δuβ[I[β]-1] +
        (uα[I-e(α)] - uα[I-e(α)-e(β)]) / Δuβ[I[β]-1]
    ) / 4
@inline ∇(u, I::CartesianIndex{2}, Δ, Δu) =
    @SMatrix [∂x(u[α], I, α, β, Δ[β], Δu[β]) for α = 1:2, β = 1:2]
@inline ∇(u, I::CartesianIndex{3}, Δ, Δu) =
    @SMatrix [∂x(u[α], I, α, β, Δ[β], Δu[β]) for α = 1:3, β = 1:3]
@inline idtensor(u, I::CartesianIndex{2}) =
    @SMatrix [(α == β) * oneunit(eltype(u[1])) for α = 1:2, β = 1:2]
@inline idtensor(u, I::CartesianIndex{3}) =
    @SMatrix [(α == β) * oneunit(eltype(u[1])) for α = 1:3, β = 1:3]
@inline function strain(u, I, Δ, Δu)
    ∇u = ∇(u, I, Δ, Δu)
    (∇u + ∇u') / 2
end
@inline gridsize(Δ, I::CartesianIndex{D}) where {D} =
    sqrt(sum(ntuple(α -> Δ[α][I[α]]^2, D)))

"""
Compute Smagorinsky stress tensors `σ[I]`.
The Smagorinsky constant `θ` should be a scalar between `0` and `1`.
"""
function smagtensor!(σ, u, θ, setup)
    # TODO: Combine with normal diffusion tensor
    (; grid, workgroupsize) = setup
    (; Np, Ip, Δ, Δu) = grid
    @kernel function σ!(σ, u, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        S = strain(u, I, Δ, Δu)
        d = gridsize(Δ, I)
        νt = θ^2 * d^2 * sqrt(2 * sum(S .* S))
        σ[I] = 2 * νt * S
    end
    I0 = first(Ip)
    I0 -= oneunit(I0)
    σ!(get_backend(u[1]), workgroupsize)(σ, u, I0; ndrange = Np)
    σ
end

"""
Compute divergence of a tensor with all components in the pressure points.
The stress tensors should be precomputed and stored in `σ`.
"""
function divoftensor!(s, σ, setup)
    (; grid, workgroupsize) = setup
    (; dimension, Nu, Iu, Δ, Δu, A) = grid
    D = dimension()
    e = Offset{D}()
    @kernel function s!(s, σ, ::Val{α}, ::Val{βrange}, I0) where {α,βrange}
        I = @index(Global, Cartesian)
        I = I + I0
        s[α][I] = zero(eltype(s[1]))
        # for β = 1:D
        KernelAbstractions.Extras.LoopInfo.@unroll for β in βrange
            Δuαβ = α == β ? Δu[β] : Δ[β]
            if α == β
                σαβ2 = σ[I+e(β)][α, β]
                σαβ1 = σ[I][α, β]
            else
                # TODO: Add interpolation weights for non-uniform case
                σαβ2 =
                    (
                        σ[I][α, β] +
                        σ[I+e(β)][α, β] +
                        σ[I+e(α)+e(β)][α, β] +
                        σ[I+e(α)][α, β]
                    ) / 4
                σαβ1 =
                    (
                        σ[I-e(β)][α, β] +
                        σ[I][α, β] +
                        σ[I+e(α)-e(β)][α, β] +
                        σ[I+e(α)][α, β]
                    ) / 4
            end
            s[α][I] += (σαβ2 - σαβ1) / Δuαβ[I[β]]
        end
    end
    for α = 1:D
        I0 = first(Iu[α])
        I0 -= oneunit(I0)
        s!(get_backend(s[1]), workgroupsize)(s, σ, Val(α), Val(1:D), I0; ndrange = Nu[α])
    end
    s
end

"""
Create Smagorinsky closure model `m`.
The model is called as `m(u, θ)`, where the Smagorinsky constant
`θ` should be a scalar between `0` and `1` (for example `θ = 0.1`).
"""
function smagorinsky_closure(setup)
    (; dimension, x, N) = setup.grid
    D = dimension()
    T = eltype(x[1])
    σ = similar(x[1], SMatrix{D,D,T,D * D}, N)
    s = ntuple(α -> similar(x[1], N), D)
    # σ = zero(similar(x[1], SMatrix{D,D,T,D * D}, N))
    # s = ntuple(α -> zero(similar(x[1], N)), D)
    function closure(u, θ)
        smagtensor!(σ, u, θ, setup)
        apply_bc_p!(σ, zero(T), setup)
        divoftensor!(s, σ, setup)
    end
end

"""
Compute symmetry tensor basis `B[1]`-`B[11]` and invariants `V[1]`-`V[5]`,
as specified in [Silvis2017](@cite) in equations (9) and (11).
Note that `B[1]` corresponds to ``T_0`` in the paper, and `V` to ``I``.
"""
function tensorbasis!(B, V, u, setup)
    (; grid, workgroupsize) = setup
    (; Np, Ip, Δ, Δu, dimension) = grid
    D = dimension()
    @kernel function basis2!(B, V, u, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        ∇u = ∇(u, I, Δ, Δu)
        S = (∇u + ∇u') / 2
        R = (∇u - ∇u') / 2
        B[1][I] = idtensor(u, I)
        B[2][I] = S
        B[3][I] = S * R - R * S
        V[1][I] = tr(S * S)
        V[2][I] = tr(R * R)
    end
    @kernel function basis3!(B, V, u, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        ∇u = ∇(u, I, Δ, Δu)
        S = (∇u + ∇u') / 2
        R = (∇u - ∇u') / 2
        B[1][I] = idtensor(u, I)
        B[2][I] = S
        B[3][I] = S * R - R * S
        B[4][I] = S * S
        B[5][I] = R * R
        B[6][I] = S * S * R - R * S * S
        B[7][I] = S * R * R + R * R * S
        B[8][I] = R * S * R * R - R * R * S * R
        B[9][I] = S * R * S * S - S * S * R * S
        B[10][I] = S * S * R * R + R * R * S * S
        B[11][I] = R * S * S * R * R - R * R * S * S * R
        V[1][I] = tr(S * S)
        V[2][I] = tr(R * R)
        V[3][I] = tr(S * S * S)
        V[4][I] = tr(S * R * R)
        V[5][I] = tr(S * S * R * R)
    end
    I0 = first(Ip)
    I0 -= oneunit(I0)
    basis! = D == 2 ? basis2! : basis3!
    basis!(get_backend(u[1]), workgroupsize)(B, V, u, I0; ndrange = Np)
    B, V
end

"""
Compute symmetry tensor basis `T[1]`-`T[11]` and invariants `V[1]`-`V[5]`.
"""
function tensorbasis(u, setup)
    T = eltype(u[1])
    D = setup.grid.dimension()
    tensorbasis!(
        ntuple(α -> similar(u[1], SMatrix{D,D,T,D * D}, setup.grid.N), D == 2 ? 3 : 11),
        ntuple(α -> similar(u[1], setup.grid.N), D == 2 ? 2 : 5),
        u,
        setup,
    )
end

"""
Interpolate velocity to pressure points.
"""
interpolate_u_p(u, setup) =
    interpolate_u_p!(ntuple(α -> similar(u[1], setup.grid.N), length(u)), u, setup)

"""
Interpolate velocity to pressure points.
"""
function interpolate_u_p!(up, u, setup)
    (; grid, workgroupsize) = setup
    (; dimension, Np, Ip) = grid
    D = dimension()
    e = Offset{D}()
    @kernel function int!(up, u, ::Val{α}, I0) where {α}
        I = @index(Global, Cartesian)
        I = I + I0
        up[α][I] = (u[α][I-e(α)] + u[α][I]) / 2
    end
    for α = 1:D
        I0 = first(Ip)
        I0 -= oneunit(I0)
        int!(get_backend(up[1]), workgroupsize)(up, u, Val(α), I0; ndrange = Np)
    end
    up
end

"""
Interpolate vorticity to pressure points.
"""
interpolate_ω_p(ω, setup) = interpolate_ω_p!(
    setup.grid.dimension() == 2 ? similar(ω, setup.grid.N) :
    ntuple(α -> similar(ω[1], setup.grid.N), length(ω)),
    ω,
    setup,
)

"""
Interpolate vorticity to pressure points.
"""
interpolate_ω_p!(ωp, ω, setup) = interpolate_ω_p!(setup.grid.dimension, ωp, ω, setup)

function interpolate_ω_p!(::Dimension{2}, ωp, ω, setup)
    (; grid, workgroupsize) = setup
    (; dimension, Np, Ip) = grid
    D = dimension()
    e = Offset{D}()
    @kernel function int!(ωp, ω, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        ωp[I] = (ω[I-e(1)-e(2)] + ω[I]) / 2
    end
    I0 = first(Ip)
    I0 -= oneunit(I0)
    int!(get_backend(ωp), workgroupsize)(ωp, ω, I0; ndrange = Np)
    ωp
end

function interpolate_ω_p!(::Dimension{3}, ωp, ω, setup)
    (; grid, workgroupsize) = setup
    (; dimension, Np, Ip) = grid
    D = dimension()
    e = Offset{D}()
    @kernel function int!(ωp, ω, ::Val{α}, I0) where {α}
        I = @index(Global, Cartesian)
        I = I + I0
        α₊ = mod1(α + 1, D)
        α₋ = mod1(α - 1, D)
        ωp[α][I] = (ω[α][I-e(α₊)-e(α₋)] + ω[α][I]) / 2
    end
    I0 = first(Ip)
    I0 -= oneunit(I0)
    for α = 1:D
        int!(get_backend(ωp[1]), workgroupsize)(ωp, ω, Val(α), I0; ndrange = Np)
    end
    ωp
end

"""
Compute the ``D``-field [LiJiajia2019](@cite) given by

```math
D = \\frac{2 | \\nabla p |}{\\nabla^2 p}.
```
"""
function Dfield!(d, G, p, setup; ϵ = eps(eltype(p)))
    (; grid, workgroupsize) = setup
    (; dimension, Np, Ip, Δ) = grid
    T = eltype(p)
    D = dimension()
    e = Offset{D}()
    @kernel function D!(d, G, p, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        g = zero(eltype(p))
        for α = 1:D
            g += (G[α][I-e(α)] + G[α][I])^2
        end
        lap = zero(eltype(p))
        # for α = 1:D
        #     lap += (G[α][I] - G[α][I-e(α)]) / Δ[α][I[α]]
        # end
        if D == 2
            lap += (G[1][I] - G[1][I-e(1)]) / Δ[1][I[1]]
            lap += (G[2][I] - G[2][I-e(2)]) / Δ[2][I[2]]
        elseif D == 3
            lap += (G[1][I] - G[1][I-e(1)]) / Δ[1][I[1]]
            lap += (G[2][I] - G[2][I-e(2)]) / Δ[2][I[2]]
            lap += (G[3][I] - G[3][I-e(3)]) / Δ[3][I[3]]
        end
        lap = lap > 0 ? max(lap, ϵ) : min(lap, -ϵ)
        # lap = abs(lap)
        d[I] = sqrt(g) / 2 / lap
    end
    pressuregradient!(G, p, setup)
    I0 = first(Ip)
    I0 -= oneunit(I0)
    D!(get_backend(p), workgroupsize)(d, G, p, I0; ndrange = Np)
    d
end

"""
Compute the ``D``-field.
"""
Dfield(p, setup; kwargs...) = Dfield!(
    zero(p),
    ntuple(α -> similar(p, setup.grid.N), setup.grid.dimension()),
    p,
    setup;
    kwargs...,
)

"""
Compute ``Q``-field [Jeong1995](@cite) given by

```math
Q = - \\frac{1}{2} \\sum_{α, β} \\frac{\\partial u^α}{\\partial x^β}
\\frac{\\partial u^β}{\\partial x^α}.
```
"""
function Qfield!(Q, u, setup)
    (; grid, workgroupsize) = setup
    (; dimension, Np, Ip, Δ) = grid
    D = dimension()
    e = Offset{D}()
    @kernel function Q!(Q, u, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        q = zero(eltype(Q))
        for α = 1:D, β = 1:D
            q -=
                (u[α][I] - u[α][I-e(β)]) / Δ[β][I[β]] * (u[β][I] - u[β][I-e(α)]) /
                Δ[α][I[α]] / 2
        end
        Q[I] = q
    end
    I0 = first(Ip)
    I0 -= oneunit(I0)
    Q!(get_backend(u[1]), workgroupsize)(Q, u, I0; ndrange = Np)
    Q
end

"""
Compute the ``Q``-field.
"""
Qfield(u, setup) = Qfield!(similar(u[1], setup.grid.N), u, setup)

"""
Compute the second eigenvalue of ``S^2 + R^2``,
as proposed by Jeong and Hussain [Jeong1995](@cite).
"""
function eig2field!(λ, u, setup)
    (; grid, workgroupsize) = setup
    (; dimension, Np, Ip, Δ, Δu) = grid
    D = dimension()
    @assert D == 3 "eig2 only implemented in 3D"
    @kernel function λ!(λ, u, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        ∇u = ∇(u, I, Δ, Δu)
        S = @. (∇u + ∇u') / 2
        R = @. (∇u - ∇u') / 2
        # FIXME: Is not recognized as hermitian with Float64 on CPU
        λ[I] = eigvals(S^2 + R^2)[2]
    end
    I0 = first(Ip)
    I0 -= oneunit(I0)
    λ!(get_backend(u[1]), workgroupsize)(λ, u, I0; ndrange = Np)
    λ
end

"""
Compute the second eigenvalue of ``S^2 + \\Omega^2``,
as proposed by Jeong and Hussain [Jeong1995](@cite).
"""
eig2field(u, setup) = eig2field!(similar(u[1], setup.grid.N), u, setup)

"""
Compute kinetic energy field ``k`` (in-place version).
If `interpolate_first` is true, it is given by

```math
k_I = \\frac{1}{8} \\sum_\\alpha (u^\\alpha_{I + h_\\alpha} + u^\\alpha_{I - h_\\alpha})^2.
```

Otherwise, it is given by

```math
k_I = \\frac{1}{4} \\sum_\\alpha ((u^\\alpha_{I + h_\\alpha})^2 + (u^\\alpha_{I - h_\\alpha})^2),
```

as in [Sanderse2023](@cite).
"""
function kinetic_energy!(ke, u, setup; interpolate_first = false)
    (; grid, workgroupsize) = setup
    (; dimension, Np, Ip) = grid
    D = dimension()
    e = Offset{D}()
    @kernel function efirst!(ke, u, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        k = zero(eltype(ke))
        for α = 1:D
            k += (u[α][I] + u[α][I-e(α)])^2
        end
        k = k / 8
        ke[I] = k
    end
    @kernel function elast!(ke, u, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        k = zero(eltype(ke))
        for α = 1:D
            k += u[α][I]^2 + u[α][I-e(α)]^2
        end
        k = k / 4
        ke[I] = k
    end
    ke! = interpolate_first ? efirst! : elast!
    I0 = first(Ip)
    I0 -= oneunit(I0)
    ke!(get_backend(u[1]), workgroupsize)(ke, u, I0; ndrange = Np)
    ke
end

"""
Compute kinetic energy field ``e`` (out-of-place version).
"""
kinetic_energy(u, setup; kwargs...) = kinetic_energy!(similar(u[1]), u, setup; kwargs...)

"""
Compute total kinetic energy. The velocity components are interpolated to the
volume centers and squared.
"""
function total_kinetic_energy(u, setup; kwargs...)
    (; Ω, Ip) = setup.grid
    k = kinetic_energy(u, setup; kwargs...)
    k .*= Ω
    sum(view(k, Ip))
end

"""
Get the following dimensional scale numbers [Pope2000](@cite):

- Velocity ``u_\\text{avg} = \\langle u_i u_i \\rangle^{1/2}``
- Dissipation rate ``\\epsilon = 2 \\nu \\langle S_{ij} S_{ij} \\rangle``
- Kolmolgorov length scale ``\\eta = (\\frac{\\nu^3}{\\epsilon})^{1/4}``
- Taylor length scale ``\\lambda = (\\frac{5 \\nu}{\\epsilon})^{1/2} u_\\text{avg}``
- Taylor-scale Reynolds number ``Re_\\lambda = \\frac{\\lambda u_\\text{avg}}{\\sqrt{3} \\nu}``
- Integral length scale ``L = \\frac{3 \\pi}{2 u_\\text{avg}^2} \\int_0^\\infty \\frac{E(k)}{k} \\, \\mathrm{d} k``
- Large-eddy turnover time ``\\tau = \\frac{L}{u_\\text{avg}}``
"""
function get_scale_numbers(u, setup)
    (; grid, Re) = setup
    (; dimension, Iu, Ip, Δ, Δu, Ω) = grid
    D = dimension()
    T = eltype(u[1])
    ν = 1 / Re
    uavg =
        sum(1:D) do α
            Δα = ntuple(
                β -> reshape(α == β ? Δu[β] : Δ[β], ntuple(Returns(1), β - 1)..., :),
                D,
            )
            Ωu = .*(Δα...)
            field = @. u[α]^2 * Ωu
            sum(field[Iu[α]]) / sum(Ωu[Iu[α]])
        end |> sqrt
    ϵ = dissipation_from_strain(u, setup)
    ϵ = sum((Ω.*ϵ)[Ip]) / sum(Ω[Ip])
    η = (ν^3 / ϵ)^T(1 / 4)
    λ = sqrt(5ν / ϵ) * uavg
    Reλ = λ * uavg / sqrt(T(3)) / ν
    # TODO: L and τ
    L = nothing
    τ = nothing
    (; uavg, ϵ, η, λ, Reλ, L, τ)
end
