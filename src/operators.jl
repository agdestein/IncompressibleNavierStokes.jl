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
Calling `Offset(D)(α)` returns a Cartesian index with `1` in the dimension `α` and zeros
elsewhere.

See <https://b-fg.github.io/research/2023-07-05-waterlily-on-gpu.html>
for writing kernel loops using Cartesian indices.
"""
struct Offset{D} end

Offset(D) = Offset{D}()

@inline (::Offset{D})(α) where {D} = CartesianIndex(ntuple(β -> β == α ? 1 : 0, D))

"Get tuple of all unit vectors as Cartesian indices."
unit_cartesian_indices(D) = ntuple(α -> Offset(D)(α), D)

"Left index `n` times away in direction `i`."
@inline left(I::CartesianIndex{D}, i, n = 1) where {D} =
    CartesianIndex(ntuple(j -> j == i ? I[j] - n : I[j], D))

"Right index `n` times away in direction `i`."
@inline right(I::CartesianIndex{D}, i, n = 1) where {D} =
    CartesianIndex(ntuple(j -> j == i ? I[j] + n : I[j], D))

"""
Apply kernel to args with offset `O`.
By default, it is applied over `setup.Ip`.
"""
function apply!(
    kernel,
    setup,
    args...;
    ndrange = map(n -> n - 2, setup.N),
    offset = oneunit(setup.Ip[1]),
)
    (; backend, workgroupsize) = setup
    kernel(backend, workgroupsize)(offset, args...; ndrange)
    KernelAbstractions.synchronize(setup.backend)
end

@kernel function contract_vector!(O::CartesianIndex{2}, f, p, args)
    I = @index(Global, Cartesian)
    I = I + O
    p[I] = f(args..., 1, I) + f(args..., 2, I)
end
@kernel function contract_vector!(O::CartesianIndex{3}, f, p, args)
    I = @index(Global, Cartesian)
    I = I + O
    p[I] = f(args..., 1, I) + f(args..., 2, I) + f(args..., 3, I)
end
@kernel function contract_vector_add!(O::CartesianIndex{2}, f, p, args)
    I = @index(Global, Cartesian)
    I = I + O
    p[I] += f(args..., 1, I) + f(args..., 2, I)
end
@kernel function contract_vector_add!(O::CartesianIndex{3}, f, p, args)
    I = @index(Global, Cartesian)
    I = I + O
    p[I] += f(args..., 1, I) + f(args..., 2, I) + f(args..., 3, I)
end

@kernel function contract_tensor!(O::CartesianIndex{2}, f, u, args)
    I = @index(Global, Cartesian)
    I = I + O
    u[I, 1] = f(args..., 1, 1, I) + f(args..., 1, 2, I)
    u[I, 2] = f(args..., 2, 1, I) + f(args..., 2, 2, I)
end
@kernel function contract_tensor!(O::CartesianIndex{3}, f, u, args)
    I = @index(Global, Cartesian)
    I = I + O
    u[I, 1] = f(args..., 1, 1, I) + f(args..., 1, 2, I) + f(args..., 1, 3, I)
    u[I, 2] = f(args..., 2, 1, I) + f(args..., 2, 2, I) + f(args..., 2, 3, I)
    u[I, 3] = f(args..., 3, 1, I) + f(args..., 3, 2, I) + f(args..., 3, 3, I)
end
@kernel function contract_tensor_add!(O::CartesianIndex{2}, f, u, args)
    I = @index(Global, Cartesian)
    I = I + O
    u[I, 1] += f(args..., 1, 1, I) + f(args..., 1, 2, I)
    u[I, 2] += f(args..., 2, 1, I) + f(args..., 2, 2, I)
end
@kernel function contract_tensor_add!(O::CartesianIndex{3}, f, u, args)
    I = @index(Global, Cartesian)
    I = I + O
    u[I, 1] += f(args..., 1, 1, I) + f(args..., 1, 2, I) + f(args..., 1, 3, I)
    u[I, 2] += f(args..., 2, 1, I) + f(args..., 2, 2, I) + f(args..., 2, 3, I)
    u[I, 3] += f(args..., 3, 1, I) + f(args..., 3, 2, I) + f(args..., 3, 3, I)
end

@kernel function expand_scalar!(O::CartesianIndex{2}, f, u, args)
    I = @index(Global, Cartesian)
    I = I + O
    u[I, 1] = f(args..., 1, I)
    u[I, 2] = f(args..., 2, I)
end
@kernel function expand_scalar!(O::CartesianIndex{3}, f, u, args)
    I = @index(Global, Cartesian)
    I = I + O
    u[I, 1] = f(args..., 1, I)
    u[I, 2] = f(args..., 2, I)
    u[I, 3] = f(args..., 3, I)
end
@kernel function expand_scalar_add!(O::CartesianIndex{2}, f, u, args)
    I = @index(Global, Cartesian)
    I = I + O
    u[I, 1] += f(args..., 1, I)
    u[I, 2] += f(args..., 2, I)
end
@kernel function expand_scalar_add!(O::CartesianIndex{3}, f, u, args)
    I = @index(Global, Cartesian)
    I = I + O
    u[I, 1] += f(args..., 1, I)
    u[I, 2] += f(args..., 2, I)
    u[I, 3] += f(args..., 3, I)
end

"""
Differentiate vector ``u_i`` in direction ``e_j``.
Make sure that `i` and `j` are known at compile-time
to remove the `if`-statement.
"""
@inline function δ(setup, u, i, j, I)
    (; Δ, Δu) = setup
    Δij = i == j ? Δ[j][I[j]] : Δu[j][I[j]]
    δu = (u[right(I, j, i != j), i] - u[left(I, j, i == j), i]) / Δij
    # For some Neumann BC, Δa or Δb are zero (eps),
    # and (right - left) / Δa blows up even if right = left according to BC.
    # Here we manually set the derivatives to zero in such cases.
    ifelse(Δij > 2 * eps(typeof(δu)), δu, zero(δu))
end

# Land in volume face
"Differentiate scalar ``p`` in direction ``e_i``."
@inline δ(setup, p, i, I) = (p[right(I, i)] - p[I]) / setup.Δu[i][I[i]]

"""
Average scalar field `ϕ` in the `α`-direction.
"""
@inline function avg(ϕ, Δ, I, α)
    e = Offset(length(I.I))
    (Δ[α][I[α]+1] * ϕ[I] + Δ[α][I[α]] * ϕ[I+e(α)]) / (Δ[α][I[α]] + Δ[α][I[α]+1])
end

"Scale scalar field `p` with volume sizes (differentiable version)."
function scalewithvolume(p, setup)
    (; dimension, Δ) = setup
    if dimension() == 2
        Δx = reshape(Δ[1], :)
        Δy = reshape(Δ[2], 1, :)
        @. p * Δx * Δy
    else
        Δx = reshape(Δ[1], :)
        Δy = reshape(Δ[2], 1, :)
        Δz = reshape(Δ[3], 1, 1, :)
        @. p * Δx * Δy * Δz
    end
end

"Scale scalar field with volume sizes (in-place version)."
function scalewithvolume!(p, setup)
    (; dimension, Δ) = setup
    if dimension() == 2
        Δx = reshape(Δ[1], :)
        Δy = reshape(Δ[2], 1, :)
        @. p *= Δx * Δy
    elseif dimension() == 3
        Δx = reshape(Δ[1], :)
        Δy = reshape(Δ[2], 1, :)
        Δz = reshape(Δ[3], 1, 1, :)
        @. p *= Δx * Δy * Δz
    end
    p
end

"Compute divergence of velocity field (differentiable version)."
divergence(u, setup) = divergence!(scalarfield(setup), u, setup)

ChainRulesCore.rrule(::typeof(divergence), u, setup) = (
    divergence(u, setup),
    φ -> (NoTangent(), divergence_adjoint!(vectorfield(setup), φ, setup), NoTangent()),
)

"Compute divergence of velocity field (in-place version)."
function divergence!(div, u, setup)
    @inline divfunc(setup, u, i, I) = δ(setup, u, i, i, I)
    apply!(contract_vector!, setup, divfunc, div, (setup, u))
    div
end

function divergence_adjoint!(u, φ, setup)
    (; Δ, N, Ip, backend, workgroupsize) = setup
    D = length(Δ)
    e = Offset(D)
    divergence_adjoint_kernel!(backend, workgroupsize)(u, φ, Δ, Ip, e; ndrange = N)
    u
end

@kernel function divergence_adjoint_kernel!(u, φ, Δ, Ip, e)
    I = @index(Global, Cartesian)
    for α in eachindex(Δ)
        adjoint = zero(eltype(u))
        I ∈ Ip && (adjoint += φ[I] / Δ[α][I[α]])
        I + e(α) ∈ Ip && (adjoint -= φ[I+e(α)] / Δ[α][I[α]+1])
        u[I, α] += adjoint
    end
end

"Compute pressure gradient (differentiable version)."
pressuregradient(p, setup) = pressuregradient!(vectorfield(setup), p, setup)

ChainRulesCore.rrule(::typeof(pressuregradient), p, setup) = (
    pressuregradient(p, setup),
    φ -> (
        NoTangent(),
        pressuregradient_adjoint!(scalarfield(setup), φ, setup),
        NoTangent(),
    ),
)

"Compute pressure gradient (in-place version)."
function pressuregradient!(G, p, setup)
    (; Ip, N) = setup
    apply!(expand_scalar!, setup, δ, G, (setup, p))
    G
end

function pressuregradient_adjoint!(pbar, φ, setup)
    (; dimension, Δu, N, Iu, backend, workgroupsize) = setup
    D = dimension()
    e = Offset(D)
    kernel = pressuregradient_adjoint_kernel!(backend, workgroupsize)
    kernel(pbar, φ, Δu, Iu, e, Val(1:D); ndrange = N)
    pbar
end

@kernel function pressuregradient_adjoint_kernel!(p, φ, Δu, Iu, e, valdims)
    I = @index(Global, Cartesian)
    adjoint = zero(eltype(p))
    @unroll for α in getval(valdims)
        I - e(α) ∈ Iu[α] && (adjoint += φ[I-e(α), α] / Δu[α][I[α]-1])
        I ∈ Iu[α] && (adjoint -= φ[I, α] / Δu[α][I[α]])
    end
    p[I] += adjoint
end

"Subtract pressure gradient (in-place version)."
function applypressure!(u, p, setup)
    @inline mδ(setup, p, i, I) = -δ(setup, p, i, I)
    apply!(expand_scalar_add!, setup, mδ, u, (setup, p))
    u
end

"Compute Laplacian of pressure field (differentiable version)."
laplacian(p, setup) = laplacian!(scalarfield(setup), p, setup)

ChainRulesCore.rrule(::typeof(laplacian), p, setup) =
    (laplacian(p, setup), φ -> error("Pullback for `laplacian` not yet implemented."))

"Compute Laplacian of pressure field (in-place version)."
function laplacian!(L, p, setup)
    (; dimension, Δ, Δu, N, Np, Ip, backend, workgroupsize, boundary_conditions) = setup
    D = dimension()
    e = Offset(D)
    # @kernel function lap!(L, p, I0)
    #     I = @index(Global, Cartesian)
    #     I = I + I0
    #     lap = zero(eltype(p))
    #     for α = 1:D
    #         # bc = boundary_conditions[α]
    #         if bc[1] isa PressureBC && I[α] == I0[α] + 1
    #             lap +=
    #                 ΩI / Δ[α][I[α]] *
    #                 ((p[I+e(α)] - p[I]) / Δu[α][I[α]] - (p[I]) / Δu[α][I[α]-1])
    #         elseif bc[2] isa PressureBC && I[α] == I0[α] + Np[α]
    #             lap +=
    #                 ΩI / Δ[α][I[α]] *
    #                 ((-p[I]) / Δu[α][I[α]] - (p[I] - p[I-e(α)]) / Δu[α][I[α]-1])
    #         elseif bc[1] isa DirichletBC && I[α] == I0[α] + 1
    #             lap += ΩI / Δ[α][I[α]] * ((p[I+e(α)] - p[I]) / Δu[α][I[α]])
    #         elseif bc[2] isa DirichletBC && I[α] == I0[α] + Np[α]
    #             lap += ΩI / Δ[α][I[α]] * (-(p[I] - p[I-e(α)]) / Δu[α][I[α]-1])
    #         else
    #             lap +=
    #                 ΩI / Δ[α][I[α]] *
    #                 ((p[I+e(α)] - p[I]) / Δu[α][I[α]] - (p[I] - p[I-e(α)]) / Δu[α][I[α]-1])
    #         end
    #     end
    #     L[I] = lap
    # end
    @kernel function lapα!(L, p, I0, ::Val{α}, bc) where {α}
        I = @index(Global, Cartesian)
        I = I + I0
        ΔI = getindex.(Δ, I.I)
        ΩI = prod(ΔI)
        # bc = boundary_conditions[α]
        if bc[1] isa PressureBC && I[α] == I0[α] + 1
            L[I] +=
                ΩI / Δ[α][I[α]] *
                ((p[I+e(α)] - p[I]) / Δu[α][I[α]] - (p[I]) / Δu[α][I[α]-1])
        elseif bc[2] isa PressureBC && I[α] == I0[α] + Np[α]
            L[I] +=
                ΩI / Δ[α][I[α]] *
                ((-p[I]) / Δu[α][I[α]] - (p[I] - p[I-e(α)]) / Δu[α][I[α]-1])
        elseif bc[1] isa DirichletBC && I[α] == I0[α] + 1
            L[I] += ΩI / Δ[α][I[α]] * ((p[I+e(α)] - p[I]) / Δu[α][I[α]])
        elseif bc[2] isa DirichletBC && I[α] == I0[α] + Np[α]
            L[I] += ΩI / Δ[α][I[α]] * (-(p[I] - p[I-e(α)]) / Δu[α][I[α]-1])
        else
            L[I] +=
                ΩI / Δ[α][I[α]] *
                ((p[I+e(α)] - p[I]) / Δu[α][I[α]] - (p[I] - p[I-e(α)]) / Δu[α][I[α]-1])
        end
        # L[I] = lap
    end
    # All volumes have a right velocity
    # All volumes have a left velocity except the first one
    # Start at second volume
    ndrange = Np
    I0 = getoffset(Ip)
    # lap!(backend, workgroupsize)(L, p, I0; ndrange)
    L .= 0
    for α = 1:D
        lapα!(backend, workgroupsize)(L, p, I0, Val(α), boundary_conditions.u[α]; ndrange)
    end
    L
end

"Compute convective term (differentiable version)."
convection(u, setup) = convection!(zero(u), u, setup)

ChainRulesCore.rrule(::typeof(convection), u, setup) = (
    convection(u, setup),
    φ -> (NoTangent(), convection_adjoint!(zero(u), φ, u, setup), NoTangent()),
)

@inline function interpolate_reverse(setup, u, i, j, I)
    (; A_coll, A_stag) = setup
    if i == j
        u[left(I, j), i] / 2 + u[I, i] / 2
    else
        A_stag[j][2][I[j]] * u[I, i] + A_stag[j][1][I[j]+1] * u[right(I, j), i]
    end
end

@inline function conv(setup, u, i, j, I)
    (; Δu, Δ) = setup
    Δuij = i == j ? Δu[j] : Δ[j]

    # Half for u[i], (reverse!) interpolation for u[j]
    # Note:
    #     In matrix version, uses
    #     1*u[i][I-e(j)] + 0*u[i][I]
    #     instead of 1/2 when u[i][I-e(j)] is at Dirichlet boundary.
    uij1 = (u[left(I, j), i] + u[I, i]) / 2
    uij2 = (u[I, i] + u[right(I, j), i]) / 2
    uji1 = interpolate_reverse(setup, u, j, i, left(I, j, i != j))
    uji2 = interpolate_reverse(setup, u, j, i, right(I, j, i == j))
    -(uij2 * uji2 - uij1 * uji1) / Δuij[I[j]]
end

@inline function convstress(setup, u, i, j, I)
    (; A_coll, A_stag) = setup
    # Half for u[i], (reverse!) interpolation for u[j]
    # Note:
    #     In matrix version, uses
    #     1*u[i][I-e(j)] + 0*u[i][I]
    #     instead of 1/2 when u[i][I-e(j)] is at Dirichlet boundary.
    if i == j
        uij = (u[left(I, j), i] + u[I, i]) / 2
        uji = uij
    else
        uij = (u[I, i] + u[right(I, j), i]) / 2
        uji = A_stag[i][2][I[i]] * u[I, j] + A_stag[i][1][I[i]+1] * u[right(I, i), j]
    end
    uij * uji
end

@inline function tensordivergence(setup, σ, args, i, j, I)
    (; Δu, Δ) = setup
    Δuij = i == j ? Δu[j] : Δ[j]
    σb = σ(args..., i, j, right(I, j, i == j))
    σa = σ(args..., i, j, left(I, j, i != j))
    -(σb - σa) / Δuij[I[j]]
end

"""
Compute convective term (in-place version).
Add the result to `F`.
"""
function convection!(f, u, setup)
    apply!(contract_tensor_add!, setup, conv, f, (setup, u))
    f
end

"""
Compute convective term (in-place version).
Add the result to `F`.
"""
function convection2!(f, u, setup)
    apply!(
        contract_tensor_add!,
        setup,
        tensordivergence,
        f,
        (setup, convstress, (setup, u)),
    )
    f
end

function convection_adjoint!(ubar, φbar, u, setup)
    (; dimension, N, backend, workgroupsize) = setup
    D = dimension()
    e = Offset(D)
    T = eltype(u)
    kernel = convection_adjoint_kernel!(backend, workgroupsize)
    kernel(ubar, φbar, u, setup, Val(1:D); ndrange = N)
    ubar
end

@kernel function convection_adjoint_kernel!(ubar, φbar, u, setup, valdims)
    h = eltype(u)(1 / 2)
    e = Offset(setup.dimension())
    dims = getval(valdims)
    (; Δ, Δu, Iu, A_coll, A_stag) = setup
    J = @index(Global, Cartesian)
    @unroll for γ in dims
        adjoint = zero(eltype(u))
        @unroll for α in dims
            @unroll for β in dims
                Δuαβ = α == β ? Δu[β] : Δ[β]
                AAβα1 = α == β ? A_coll[α][1] : A_stag[α][1]
                AAβα2 = α == β ? A_coll[α][2] : A_stag[α][2]

                # 1
                I = J
                if α == γ && I in Iu[α]
                    uαβ2 = h
                    uβα2 = interpolate_reverse(setup, u, β, α, I + (α == β) * e(β))
                    dφdu = -uαβ2 * uβα2 / Δuαβ[I[β]]
                    adjoint += φbar[I, α] * dφdu
                end

                # 2
                I = J - e(β)
                if α == γ && I in Iu[α]
                    uαβ2 = h
                    uβα2 = interpolate_reverse(setup, u, β, α, I + (α == β) * e(β))
                    dφdu = -uαβ2 * uβα2 / Δuαβ[I[β]]
                    adjoint += φbar[I, α] * dφdu
                end

                # 3
                I = J
                if β == γ && I in Iu[α]
                    uαβ2 = h * u[I, α] + h * u[I+e(β), α]
                    uβα2 = AAβα2[I[α]+(α==β)]
                    dφdu = -uαβ2 * uβα2 / Δuαβ[I[β]]
                    adjoint += φbar[I, α] * dφdu
                end

                # 4
                I = J - e(α)
                if β == γ && I in Iu[α]
                    uαβ2 = h * u[I, α] + h * u[I+e(β), α]
                    uβα2 = AAβα1[I[α]+1]
                    dφdu = -uαβ2 * uβα2 / Δuαβ[I[β]]
                    adjoint += φbar[I, α] * dφdu
                end

                # 5
                I = J + e(β)
                if α == γ && I in Iu[α]
                    uαβ1 = h
                    uβα1 = interpolate_reverse(setup, u, β, α, I - (α != β) * e(β))
                    dφdu = uαβ1 * uβα1 / Δuαβ[I[β]]
                    adjoint += φbar[I, α] * dφdu
                end

                # 6
                I = J
                if α == γ && I in Iu[α]
                    uαβ1 = h
                    uβα1 = interpolate_reverse(setup, u, β, α, I - (α != β) * e(β))
                    dφdu = uαβ1 * uβα1 / Δuαβ[I[β]]
                    adjoint += φbar[I, α] * dφdu
                end

                # 7
                I = J + e(β)
                if β == γ && I in Iu[α]
                    uαβ1 = h * u[I-e(β), α] + h * u[I, α]
                    uβα1 = AAβα2[I[α]-(α==β)]
                    dφdu = uαβ1 * uβα1 / Δuαβ[I[β]]
                    adjoint += φbar[I, α] * dφdu
                end

                # 8
                I = J + e(β) - e(α)
                if β == γ && I in Iu[α]
                    uαβ1 = h * u[I-e(β), α] + h * u[I, α]
                    uβα1 = AAβα1[I[α]+(α!=β)]
                    dφdu = uαβ1 * uβα1 / Δuαβ[I[β]]
                    adjoint += φbar[I, α] * dφdu
                end
            end
        end
        ubar[J, γ] += adjoint
    end
end

"Compute diffusive term (differentiable version)."
diffusion(u, setup, viscosity) = diffusion!(zero.(u), u, setup, viscosity)

ChainRulesCore.rrule(::typeof(diffusion), u, setup, viscosity) = (
    diffusion(u, setup, viscosity),
    φ -> (
        NoTangent(),
        diffusion_adjoint!(zero(u), φ, setup, viscosity),
        NoTangent(),
        NoTangent(),
    ),
)

"""
Compute diffusive term (in-place version).
Add the result to `F`.
"""
function diffusion!(f, u, setup, viscosity)
    apply!(
        contract_tensor_add!,
        setup,
        tensordivergence,
        f,
        (setup, diffstress, (setup, u, viscosity)),
    )
    f
end

@inline diffstress(setup, u, viscosity, i, j, I) = -viscosity * δ(setup, u, i, j, I)

function diffusion_adjoint!(u, φ, setup, viscosity)
    (; dimension, N, Δ, Δu, Iu, backend, workgroupsize) = setup
    D = dimension()
    e = Offset(D)
    kernel = diffusion_adjoint_kernel!(backend, workgroupsize)
    kernel(u, φ, viscosity, e, Δ, Δu, Iu, Val(1:D); ndrange = N)
    u
end

@kernel function diffusion_adjoint_kernel!(u, φ, visc, e, Δ, Δu, Iu, valdims)
    dims = getval(valdims)
    I = @index(Global, Cartesian)
    @unroll for α in dims
        val = zero(eltype(u))
        @unroll for β in dims
            Δuαβ = α == β ? Δu[β] : Δ[β]
            # F[α][I] += visc * u[I+e(β), α] / (β == α ? Δ[β][I[β]+1] : Δu[β][I[β]])
            # F[α][I] -= visc * u[I, α] / (β == α ? Δ[β][I[β]+1] : Δu[β][I[β]])
            # F[α][I] -= visc * u[I, α] / (β == α ? Δ[β][I[β]] : Δu[β][I[β]-1])
            # F[α][I] += visc * u[I-e(β), α] / (β == α ? Δ[β][I[β]] : Δu[β][I[β]-1])
            if I - e(β) ∈ Iu[α]
                val +=
                    visc * φ[I-e(β), α] / (β == α ? Δ[β][I[β]] : Δu[β][I[β]-1]) /
                    Δuαβ[I[β]-1]
            end
            if I ∈ Iu[α]
                val -= visc * φ[I, α] / (β == α ? Δ[β][I[β]+1] : Δu[β][I[β]]) / Δuαβ[I[β]]
            end
            if I ∈ Iu[α]
                val -= visc * φ[I, α] / (β == α ? Δ[β][I[β]] : Δu[β][I[β]-1]) / Δuαβ[I[β]]
            end
            if I + e(β) ∈ Iu[α]
                val +=
                    visc * φ[I+e(β), α] / (β == α ? Δ[β][I[β]+1] : Δu[β][I[β]]) /
                    Δuαβ[I[β]+1]
            end
        end
        u[I, α] += val
    end
end

"""
Compute diffusive term (in-place version).
Add the result to `F`.
"""
function convectiondiffusion!(f, u, setup, viscosity)
    @inline convdiffstress(setup, u, viscosity, i, j, I) =
        convstress(setup, u, i, j, I) + diffstress(setup, u, viscosity, i, j, I)
    apply!(
        contract_tensor_add!,
        setup,
        tensordivergence,
        f,
        (setup, convdiffstress, (setup, u, viscosity)),
    )
    f
end

"""
Compute convection-diffusion term for the temperature equation.
(differentiable version).
"""
convection_diffusion_temp(u, temp, setup) =
    convection_diffusion_temp!(zero(temp), u, temp, setup)

function ChainRulesCore.rrule(::typeof(convection_diffusion_temp), u, temp, setup)
    conv = convection_diffusion_temp(u, temp, setup)
    convection_diffusion_temp_pullback(φ) = (NoTangent(), du, dtemp, NoTangent())
    @warn "Check if convection_diffusion_temp pullback behaves as expected"
    (conv, pullback)
end

"""
Compute convection-diffusion term for the temperature equation.
(in-place version).
Add result to `c`.
"""
function convection_diffusion_temp!(c, u, temp, setup, conductivity)
    (; dimension, Δ, Δu, Np, Ip, backend, workgroupsize) = setup
    D = dimension()
    e = Offset(D)
    I0 = getoffset(Ip)
    kernel = convection_diffusion_temp_kernel!(backend, workgroupsize)
    kernel(c, u, temp, conductivity, Δ, Δu, e, Val(1:D), I0; ndrange = Np)
    c
end

@kernel function convection_diffusion_temp_kernel!(c, u, temp, cond, Δ, Δu, e, valdims, I0)
    I = @index(Global, Cartesian)
    I = I + I0
    cI = zero(eltype(c))
    @unroll for β in getval(valdims)
        ∂T∂x1 = (temp[I] - temp[I-e(β)]) / Δu[β][I[β]-1]
        ∂T∂x2 = (temp[I+e(β)] - temp[I]) / Δu[β][I[β]]
        uT1 = u[I-e(β), β] * avg(temp, Δ, I - e(β), β)
        uT2 = u[I, β] * avg(temp, Δ, I, β)
        cI += (-(uT2 - uT1) + cond * (∂T∂x2 - ∂T∂x1)) / Δ[β][I[β]]
    end
    c[I] += cI
end

"Compute dissipation term for the temperature equation (differentiable version)."
dissipation(u, setup, coeff) = dissipation!(scalarfield(setup), u, setup, coeff)

function ChainRulesCore.rrule(::typeof(dissipation), u, setup, coeff)
    error("Not imlemented yet")
    φ, dissipation_pullback
end

"""
Compute dissipation term for the temperature equation (in-place version).
Add result to `diss`.
"""
function dissipation!(diss, u, setup, coeff)
    (; dimension, Δ, Δu, Np, Ip, backend, workgroupsize) = setup
    D = dimension()
    e = Offset(D)
    @kernel function dissipartion_kernel!(diss, u, coeff, I0)
        I = @index(Global, Cartesian)
        I += I0
        G = ∇(u, I, Δ, Δu)
        diss[I] += coeff * dot(G, G)
    end
    I0 = first(Ip)
    I0 -= oneunit(I0)
    dissipartion_kernel!(backend, workgroupsize)(diss, u, coeff, I0; ndrange = Np)
    diss
end

"Compute gravity term (differentiable version)."
applygravity(temp, setup) = applygravity!(vectorfield(setup), temp, setup)

function ChainRulesCore.rrule(::typeof(applygravity), temp, setup, gdir, gravity)
    (; dimension, Δ, N, Iu, backend, workgroupsize) = setup
    backend = get_backend(temp)
    D = dimension()
    e = Offset(D)
    g = applygravity(temp, setup, gdir, gravity)
    function gravity_pullback(φ)
        @kernel function g!(tempbar, φbar, valα)
            α = getval(valα)
            J = @index(Global, Cartesian)
            t = zero(eltype(tempbar))
            # 1
            I = J
            I ∈ Iu[α] &&
                (t += gravity * Δ[α][I[α]+1] * φbar[I, α] / (Δ[α][I[α]] + Δ[α][I[α]+1]))
            # 2
            I = J - e(α)
            I ∈ Iu[α] &&
                (t += gravity * Δ[α][I[α]] * φbar[I, α] / (Δ[α][I[α]] + Δ[α][I[α]+1]))
            tempbar[J] = t
        end
        tempbar = zero(temp)
        g!(backend, workgroupsize)(tempbar, φ, Val(gdir); ndrange = N)
        (NoTangent(), tempbar, NoTangent())
    end
    g, gravity_pullback
end

"""
Compute gravity term (in-place version).
add the result to `F`.
"""
function applygravity!(F, temp, setup, gdir, gravity)
    (; dimension, Δ, Nu, Iu, backend, workgroupsize) = setup
    @kernel function g!(F, temp, gravity, ::Val{gdir}, I0) where {gdir}
        I = @index(Global, Cartesian)
        I = I + I0
        F[I, gdir] += gravity * avg(temp, Δ, I, gdir)
    end
    I0 = first(Iu[gdir])
    I0 -= oneunit(I0)
    g!(backend, workgroupsize)(F, temp, gravity, Val(gdir), I0; ndrange = Nu[gdir])
    F
end

"Compute vorticity field (differentiable version)."
vorticity(u, setup) =
    vorticity!(setup.dimension() == 2 ? scalarfield(setup) : vectorfield(setup), u, setup)

"Compute vorticity field (in-place version)."
vorticity!(ω, u, setup) = vorticity!(setup.dimension, ω, u, setup)

# 2D version
function vorticity!(::Dimension{2}, ω, u, setup)
    (; dimension, Δu, N, backend, workgroupsize) = setup
    D = dimension()
    e = Offset(D)
    @kernel function ω!(ω, u)
        I = @index(Global, Cartesian)
        ω[I] =
            (u[I+e(1), 2] - u[I, 2]) / Δu[1][I[1]] - (u[I+e(2), 1] - u[I, 1]) / Δu[2][I[2]]
    end
    ω!(backend, workgroupsize)(ω, u; ndrange = N .- 1)
    ω
end

# 3D version
function vorticity!(::Dimension{3}, ω, u, setup)
    (; dimension, Δu, N, backend, workgroupsize) = setup
    D = dimension()
    e = Offset(D)
    @kernel function ω!(ω, u)
        I = @index(Global, Cartesian)
        for (α, α₊, α₋) in ((1, 2, 3), (2, 3, 1), (3, 1, 2))
            # α₊ = mod1(α + 1, D)
            # α₋ = mod1(α - 1, D)
            ω[I, α] =
                (u[I+e(α₊), α₋] - u[I, α₋]) / Δu[α₊][I[α₊]] -
                (u[I+e(α₋), α₊] - u[I, α₊]) / Δu[α₋][I[α₋]]
        end
    end
    ω!(backend, workgroupsize)(ω, u; ndrange = N .- 1)
    ω
end

@inline ∂x(u, I::CartesianIndex{D}, α, β, Δβ, Δuβ; e = Offset(D)) where {D} =
    if α == β
        (u[I, α] - u[I-e(β), α]) / Δβ[I[β]]
    else
        (
            (u[I+e(β), α] - u[I, α]) / Δuβ[I[β]] +
            (u[I-e(α)+e(β), α] - u[I-e(α), α]) / Δuβ[I[β]] +
            (u[I, α] - u[I-e(β), α]) / Δuβ[I[β]-1] +
            (u[I-e(α), α] - u[I-e(α)-e(β), α]) / Δuβ[I[β]-1]
        ) / 4
    end
@inline function ∂x_adjoint!(
    φ,
    u,
    I::CartesianIndex{D},
    α,
    β,
    Δβ,
    Δuβ;
    e = Offset(D),
) where {D}
    # TODO:
    # - Invert input/output indices
    # - combine all output indices into one
    # - Get rid of @atomic
    if α == β
        # φ = (u[I, α] - u[I-e(β), α]) / Δβ[I[β]]
        val = φ / Δβ[I[β]]
        @atomic u[I, α] += val
        @atomic u[I-e(β), α] -= val
    else
        # φ =
        #     (u[I+e(β), α] - u[I, α]) / 4Δuβ[I[β]] +
        #     (u[I-e(α)+e(β), α] - u[I-e(α), α]) / 4Δuβ[I[β]] +
        #     (u[I, α] - u[I-e(β), α]) / 4Δuβ[I[β]-1] +
        #     (u[I-e(α), α] - u[I-e(α)-e(β), α]) / 4Δuβ[I[β]-1]
        val = φ / 4Δuβ[I[β]]
        @atomic u[I+e(β), α] += val
        @atomic u[I, α] -= val
        @atomic u[I-e(α)+e(β), α] += val
        @atomic u[I-e(α), α] -= val
        val = φ / 4Δuβ[I[β]-1]
        @atomic u[I, α] += val
        @atomic u[I-e(β), α] -= val
        @atomic u[I-e(α), α] += val
        @atomic u[I-e(α)-e(β), α] -= val
    end
    u
end
@inline ∇(u, I::CartesianIndex{2}, Δ, Δu) = SMatrix{2,2,eltype(u),4}(
    ∂x(u, I, 1, 1, Δ[1], Δu[1]),
    ∂x(u, I, 2, 1, Δ[1], Δu[1]),
    ∂x(u, I, 1, 2, Δ[2], Δu[2]),
    ∂x(u, I, 2, 2, Δ[2], Δu[2]),
)
@inline ∇(u, I::CartesianIndex{3}, Δ, Δu) = SMatrix{3,3,eltype(u),9}(
    ∂x(u, I, 1, 1, Δ[1], Δu[1]),
    ∂x(u, I, 2, 1, Δ[1], Δu[1]),
    ∂x(u, I, 3, 1, Δ[1], Δu[1]),
    ∂x(u, I, 1, 2, Δ[2], Δu[2]),
    ∂x(u, I, 2, 2, Δ[2], Δu[2]),
    ∂x(u, I, 3, 2, Δ[2], Δu[2]),
    ∂x(u, I, 1, 3, Δ[3], Δu[3]),
    ∂x(u, I, 2, 3, Δ[3], Δu[3]),
    ∂x(u, I, 3, 3, Δ[3], Δu[3]),
)
@inline function ∇_adjoint!(∇u, u, I::CartesianIndex{2}, Δ, Δu)
    ∂x_adjoint!(∇u[1, 1], u, I, 1, 1, Δ[1], Δu[1])
    ∂x_adjoint!(∇u[2, 1], u, I, 2, 1, Δ[1], Δu[1])
    ∂x_adjoint!(∇u[1, 2], u, I, 1, 2, Δ[2], Δu[2])
    ∂x_adjoint!(∇u[2, 2], u, I, 2, 2, Δ[2], Δu[2])
    u
end
@inline function ∇_adjoint!(∇u, u, I::CartesianIndex{3}, Δ, Δu)
    ∂x_adjoint!(∇u[1, 1], u, I, 1, 1, Δ[1], Δu[1])
    ∂x_adjoint!(∇u[2, 1], u, I, 2, 1, Δ[1], Δu[1])
    ∂x_adjoint!(∇u[3, 1], u, I, 3, 1, Δ[1], Δu[1])
    ∂x_adjoint!(∇u[1, 2], u, I, 1, 2, Δ[2], Δu[2])
    ∂x_adjoint!(∇u[2, 2], u, I, 2, 2, Δ[2], Δu[2])
    ∂x_adjoint!(∇u[3, 2], u, I, 3, 2, Δ[2], Δu[2])
    ∂x_adjoint!(∇u[1, 3], u, I, 1, 3, Δ[3], Δu[3])
    ∂x_adjoint!(∇u[2, 3], u, I, 2, 3, Δ[3], Δu[3])
    ∂x_adjoint!(∇u[3, 3], u, I, 3, 3, Δ[3], Δu[3])
    u
end
@inline idtensor(u, ::CartesianIndex{2}) = SMatrix{2,2,eltype(u),4}(1, 0, 0, 1)
@inline idtensor(u, ::CartesianIndex{3}) =
    SMatrix{3,3,eltype(u),9}(1, 0, 0, 0, 1, 0, 0, 0, 1)
@inline unittensor(u, ::CartesianIndex{2}, α, β) = SMatrix{2,2,eltype(u),4}(
    (α, β) == (1, 1),
    (α, β) == (2, 1),
    (α, β) == (1, 2),
    (α, β) == (2, 2),
)
@inline unittensor(u, ::CartesianIndex{3}, α, β) = SMatrix{3,3,eltype(u),9}(
    (α, β) == (1, 1),
    (α, β) == (2, 1),
    (α, β) == (3, 1),
    (α, β) == (1, 2),
    (α, β) == (2, 2),
    (α, β) == (3, 2),
    (α, β) == (1, 3),
    (α, β) == (2, 3),
    (α, β) == (3, 3),
)

"Gridsize based on the length of the diagonal of the cell."
@inline gridsize(setup, I::CartesianIndex{2}) = sqrt(setup[1][I[1]]^2 + setup[2][I[2]]^2)
@inline gridsize(setup, I::CartesianIndex{3}) =
    cbrt(setup[1][I[1]]^2 + setup[2][I[2]]^2 + setup[3][I[3]]^2)

"Grid size based on the volume of the cell."
@inline gridsize_vol(setup, I::CartesianIndex{2}) =
    sqrt(setup.Δ[1][I[1]] * setup.Δ[2][I[2]])
@inline gridsize_vol(setup, I::CartesianIndex{3}) =
    cbrt(setup.Δ[1][I[1]] * setup.Δ[2][I[2]] * setup.Δ[3][I[3]])

"Interpolate velocity to pressure points (differentiable version)."
interpolate_u_p(u, setup) = interpolate_u_p!(vectorfield(setup), u, setup)

"Interpolate velocity to pressure points (in-place version)."
function interpolate_u_p!(up, u, setup)
    (; dimension, Np, Ip, backend, workgroupsize) = setup
    D = dimension()
    e = Offset(D)
    @kernel function int!(up, u, ::Val{α}, I0) where {α}
        I = @index(Global, Cartesian)
        I = I + I0
        up[I, α] = (u[I-e(α), α] + u[I, α]) / 2
    end
    for α = 1:D
        I0 = getoffset(Ip)
        int!(backend, workgroupsize)(up, u, Val(α), I0; ndrange = Np)
    end
    up
end

"Interpolate vorticity to pressure points (differentiable version)."
interpolate_ω_p(ω, setup) = interpolate_ω_p!(
    setup.dimension() == 2 ? scalarfield(setup) : vectorfield(setup),
    ω,
    setup,
)

"Interpolate vorticity to pressure points (in-place version)."
interpolate_ω_p!(ωp, ω, setup) = interpolate_ω_p!(setup.dimension, ωp, ω, setup)

# 2D version
function interpolate_ω_p!(::Dimension{2}, ωp, ω, setup)
    (; dimension, Np, Ip, backend, workgroupsize) = setup
    D = dimension()
    e = Offset(D)
    @kernel function int!(ωp, ω, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        ωp[I] = (ω[I-e(1)-e(2)] + ω[I]) / 2
    end
    I0 = getoffset(Ip)
    int!(backend, workgroupsize)(ωp, ω, I0; ndrange = Np)
    ωp
end

# 3D version
function interpolate_ω_p!(::Dimension{3}, ωp, ω, setup)
    (; dimension, Np, Ip, backend, workgroupsize) = setup
    D = dimension()
    e = Offset(D)
    @kernel function int!(ωp, ω, ::Val{α}, I0) where {α}
        I = @index(Global, Cartesian)
        I = I + I0
        α₊ = mod1(α + 1, D)
        α₋ = mod1(α - 1, D)
        ωp[I, α] = (ω[I-e(α₊)-e(α₋), α] + ω[I, α]) / 2
    end
    I0 = getoffset(Ip)
    for α = 1:D
        int!(backend, workgroupsize)(ωp, ω, Val(α), I0; ndrange = Np)
    end
    ωp
end

"Compute ``Q``, the second invariant of the velocity gradient tensor."
qcrit(u, setup) = qcrit!(scalarfield(setup), u, setup)

function qcrit!(q, u, setup)
    (; Np, Δ, Δu, backend, workgroupsize) = setup
    @kernel function qcrit_kernel!(q, u)
        I = @index(Global, Cartesian)
        I += oneunit(I)
        G = ∇(u, I, Δ, Δu)
        q[I] = -tr(G * G) / 2
    end
    kernel! = qcrit_kernel!(backend, workgroupsize)
    kernel!(q, u; ndrange = Np)
    q
end

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

Differentiable version.
"""
kinetic_energy(u, setup; kwargs...) =
    kinetic_energy!(scalarfield(setup), u, setup; kwargs...)

ChainRulesCore.rrule(::typeof(kinetic_energy), u, setup; kwargs...) =
    (kinetic_energy(u, setup; kwargs...), φ -> error("Not yet implemented"))

"Compute kinetic energy field (in-place version)."
function kinetic_energy!(ke, u, setup; interpolate_first = false)
    (; dimension, Np, Ip, backend, workgroupsize) = setup
    D = dimension()
    e = Offset(D)
    @kernel function efirst!(ke, u, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        k = zero(eltype(ke))
        for α = 1:D
            k += (u[I, α] + u[I-e(α), α])^2
        end
        k = k / 8
        ke[I] = k
    end
    @kernel function elast!(ke, u, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        k = zero(eltype(ke))
        for α = 1:D
            k += u[I, α]^2 + u[I-e(α), α]^2
        end
        k = k / 4
        ke[I] = k
    end
    ke! = interpolate_first ? efirst! : elast!
    I0 = getoffset(Ip)
    ke!(backend, workgroupsize)(ke, u, I0; ndrange = Np)
    ke
end

"""
Compute total kinetic energy. The velocity components are interpolated to the
volume centers and squared.
"""
function total_kinetic_energy(u, setup; kwargs...)
    (; Ip) = setup
    k = kinetic_energy(u, setup; kwargs...)
    k = scalewithvolume(k, setup)
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
function get_scale_numbers(u, setup, viscosity)
    (; dimension, Iu, Ip, Δ, Δu, Np) = setup
    D = dimension()
    T = eltype(u)
    Ω = scalewithvolume!(fill!(scalarfield(setup), 1), setup)
    uavg =
        sum(1:D) do α
            Δα = ntuple(
                β -> reshape(α == β ? Δu[β] : Δ[β], ntuple(Returns(1), β - 1)..., :),
                D,
            )
            Ωu = .*(Δα...)
            uα = eachslice(u; dims = ndims(u))
            field = @. u^2 * Ωu
            sum(field[Iu[1], :]) / sum(Ωu[Iu[1]])
        end |> sqrt
    ϵ = dissipation(u, setup, viscosity)
    ϵ = sum((Ω .* ϵ)[Ip]) / sum(Ω[Ip])
    η = (viscosity^3 / ϵ)^T(1 / 4)
    λ = sqrt(5 * viscosity / ϵ) * uavg
    Reλ = λ * uavg / sqrt(T(3)) / viscosity
    L = let
        assert_uniform_periodic(setup, "Scale numbers")
        K = div.(Np, 2)
        up = view(u, Ip, :)
        uhat = fft(up, 1:D)
        uhat = uhat[ntuple(α->1:K[α], D)..., :]
        e = abs2.(uhat) ./ (2 * prod(Np)^2)
        if D == 2
            kx = reshape(0:(K[1]-1), :)
            ky = reshape(0:(K[2]-1), 1, :)
            @. e = e / sqrt(kx^2 + ky^2)
        elseif D == 3
            kx = reshape(0:(K[1]-1), :)
            ky = reshape(0:(K[2]-1), 1, :)
            kz = reshape(0:(K[3]-1), 1, 1, :)
            @. e = e / sqrt(kx^2 + ky^2 + kz^2)
        end
        e = sum(e; dims = D + 1)
        # Remove k=(0,...,0) component
        # Note use of singleton range 1:1 instead of scalar index 1
        # (otherwise CUDA gets annoyed)
        e[1:1] .= 0
        T(3π) / 2 / uavg^2 * sum(e)
    end
    τ = L / uavg
    Re_int = L * uavg / viscosity
    (; uavg, ϵ, η, λ, Reλ, L, τ, Re_int)
end
