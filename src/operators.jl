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

See <https://b-fg.github.io/2023/05/07/waterlily-on-gpu.html>
for writing kernel loops using Cartesian indices.
"""
struct Offset{D} end

Offset(D) = Offset{D}()

@inline (::Offset{D})(α) where {D} = CartesianIndex(ntuple(β -> β == α ? 1 : 0, D))

"Get tuple of all unit vectors as Cartesian indices."
unit_cartesian_indices(D) = ntuple(α -> Offset(D)(α), D)

"""
Average scalar field `ϕ` in the `α`-direction.
"""
@inline function avg(ϕ, Δ, I, α)
    e = Offset(length(I.I))
    (Δ[α][I[α]+1] * ϕ[I] + Δ[α][I[α]] * ϕ[I+e(α)]) / (Δ[α][I[α]] + Δ[α][I[α]+1])
end

"Scale scalar field `p` with volume sizes (differentiable version)."
function scalewithvolume(p, setup)
    (; grid) = setup
    (; dimension, Δ) = grid
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
    (; grid) = setup
    (; dimension, Δ) = grid
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
    (; grid, backend, workgroupsize) = setup
    (; Δ, Ip, Np) = grid
    D = length(Δ)
    e = Offset(D)
    I0 = getoffset(Ip)
    kernel = divergence_kernel!(backend, workgroupsize)
    kernel(div, u, Δ, e, I0; ndrange = Np)
    div
end

@kernel inbounds = true function divergence_kernel!(div, u, Δ, e, I0)
    I = @index(Global, Cartesian)
    I = I + I0
    d = zero(eltype(div))
    for α in eachindex(Δ)
        d += (u[I, α] - u[I-e(α), α]) / Δ[α][I[α]]
    end
    div[I] = d
end

function divergence_adjoint!(u, φ, setup)
    (; grid, backend, workgroupsize) = setup
    (; Δ, N, Ip) = grid
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
    (; grid, backend, workgroupsize) = setup
    (; dimension, Δu, N, Iu) = grid
    D = dimension()
    e = Offset(D)
    kernel = pressuregradient_kernel!(backend, workgroupsize)
    I0 = oneunit(CartesianIndex{D})
    kernel(G, p, Δu, Iu, e, Val(1:D), I0; ndrange = N .- 2)
    G
end

@kernel function pressuregradient_kernel!(G, p, Δu, Iu, e, valdims, I0)
    I = @index(Global, Cartesian)
    I = I0 + I
    @unroll for α in getval(valdims)
        if I ∈ Iu[α]
            G[I, α] = (p[I+e(α)] - p[I]) / Δu[α][I[α]]
        end
    end
end

function pressuregradient_adjoint!(pbar, φ, setup)
    (; grid, backend, workgroupsize) = setup
    (; dimension, Δu, N, Iu) = grid
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

# "Subtract pressure gradient (differentiable version)."
applypressure(u, p, setup) = applypressure!(copy.(u), p, setup)

ChainRulesCore.rrule(::typeof(applypressure), u, p, setup) = (
    applypressure(u, p, setup),
    φ -> (
        NoTangent(),
        NoTangent(),
        applypressure_adjoint!(scalarfield(setup), φ, nothing, setup),
        NoTangent(),
    ),
)

"Subtract pressure gradient (in-place version)."
function applypressure!(u, p, setup)
    (; grid, backend, workgroupsize) = setup
    (; dimension, Δu, N) = grid
    D = dimension()
    e = Offset(D)
    kernel = applypressure_kernel!(backend, workgroupsize)
    I0 = oneunit(CartesianIndex{D})
    kernel(u, p, Δu, e, Val(1:D), I0; ndrange = N .- 2)
    u
end

@kernel function applypressure_kernel!(u, p, Δu, e, valdims, I0)
    I = @index(Global, Cartesian)
    I = I0 + I
    @unroll for α in getval(valdims)
        u[I, α] -= (p[I+e(α)] - p[I]) / Δu[α][I[α]]
    end
end

#function applypressure_adjoint!(pbar, φ, u, setup)
#    (; grid, backend, workgroupsize) = setup
#    (; dimension, Δu, N, Iu) = grid
#    D = dimension()
#    e = Offset(D)
#    @kernel function applypressure_adjoint_kernel!(p, φ)
#        I = @index(Global, Cartesian)
#        p[I] = zero(eltype(p))
#        for α = 1:D
#            I - e(α) ∈ Iu[α] && (p[I] += φ[I-e(α),α] / Δu[α][I[α]-1])
#            I ∈ Iu[α] && (p[I] -= φ[I,α] / Δu[α][I[α]])
#        end
#    end
#    applypressure_adjoint_kernel!(backend, workgroupsize)(pbar, φ; ndrange = N)
#    pbar
#end
function applypressure_adjoint!(pbar, φ, u, setup)
    # Extract necessary components from the setup structure
    (; grid, backend, workgroupsize) = setup
    (; dimension, Δu, N, Iu) = grid
    D = dimension()  # Get the spatial dimension
    e = Offset(D)    # Offset function for indexing neighbors

    # Kernel definition for computing the adjoint
    @kernel function applypressure_adjoint_kernel!(p, φ)
        # Get the global index for the current thread
        I = @index(Global, Cartesian)

        # Initialize the adjoint value at the current index to zero
        local p_I = zero(eltype(p))

        # Loop over each dimension to compute adjoint contributions
        for α = 1:D
            # Contribution from φ[I - e(α)] / Δu[α][I[α] - 1]
            if I - e(α) ∈ Iu[α]
                p_I += φ[I-e(α), α] / Δu[α][I[α]-1]
            end

            # Contribution from -φ[I, α] / Δu[α][I[α]]
            if I ∈ Iu[α]
                p_I -= φ[I, α] / Δu[α][I[α]]
            end
        end

        # Assign the computed value back to p
        p[I] = p_I
    end

    # Run the adjoint kernel on the backend, with specified workgroup size
    applypressure_adjoint_kernel!(backend, workgroupsize)(pbar, φ; ndrange = N)

    # Return the adjoint result for p
    return pbar
end

"Compute Laplacian of pressure field (differentiable version)."
laplacian(p, setup) = laplacian!(scalarfield(setup), p, setup)

ChainRulesCore.rrule(::typeof(laplacian), p, setup) =
    (laplacian(p, setup), φ -> error("Pullback for `laplacian` not yet implemented."))

"Compute Laplacian of pressure field (in-place version)."
function laplacian!(L, p, setup)
    (; grid, backend, workgroupsize, boundary_conditions) = setup
    (; dimension, Δ, Δu, N, Np, Ip) = grid
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
        lapα!(backend, workgroupsize)(L, p, I0, Val(α), boundary_conditions[α]; ndrange)
    end
    L
end

"Compute convective term (differentiable version)."
convection(u, setup) = convection!(zero(u), u, setup)

ChainRulesCore.rrule(::typeof(convection), u, setup) = (
    convection(u, setup),
    φ -> (NoTangent(), convection_adjoint!(zero(u), φ, u, setup), NoTangent()),
)

"""
Compute convective term (in-place version).
Add the result to `F`.
"""
function convection!(F, u, setup)
    (; grid, backend, workgroupsize) = setup
    (; dimension, Δ, Δu, N, A, Iu) = grid
    D = dimension()
    e = Offset(D)
    kernel = convection_kernel!(backend, workgroupsize)
    I0 = oneunit(CartesianIndex{D})
    kernel(F, u, Δ, Δu, A, Iu, e, Val(1:D), I0; ndrange = N .- 2)
    F
end

@kernel function convection_kernel!(F, u, Δ, Δu, A, Iu, e, valdims, I0)
    dims = getval(valdims)
    I = @index(Global, Cartesian)
    I = I + I0
    @unroll for α in dims
        f = F[I, α]
        if I ∈ Iu[α]
            @unroll for β in dims
                Δuαβ = α == β ? Δu[β] : Δ[β]

                # Half for u[α], (reverse!) interpolation for u[β]
                # Note:
                #     In matrix version, uses
                #     1*u[α][I-e(β)] + 0*u[α][I]
                #     instead of 1/2 when u[α][I-e(β)] is at Dirichlet boundary.
                uαβ1 = (u[I-e(β), α] + u[I, α]) / 2
                uαβ2 = (u[I, α] + u[I+e(β), α]) / 2
                uβα1 =
                    A[β][α][2][I[α]-(α==β)] * u[I-e(β), β] +
                    A[β][α][1][I[α]+(α!=β)] * u[I-e(β)+e(α), β]
                uβα2 = A[β][α][2][I[α]] * u[I, β] + A[β][α][1][I[α]+1] * u[I+e(α), β]
                f -= (uαβ2 * uβα2 - uαβ1 * uβα1) / Δuαβ[I[β]]
            end
        end
        F[I, α] = f
    end
end

function convection_adjoint!(ubar, φbar, u, setup)
    (; grid, backend, workgroupsize) = setup
    (; dimension, Δ, Δu, N, Iu, A) = grid
    D = dimension()
    e = Offset(D)
    T = eltype(u)
    h = T(1) / 2
    kernel = convection_adjoint_kernel!(backend, workgroupsize)
    kernel(ubar, φbar, u, Δ, Δu, Iu, A, h, e, Val(1:D); ndrange = N)
    ubar
end

@kernel function convection_adjoint_kernel!(ubar, φbar, u, Δ, Δu, Iu, A, h, e, valdims)
    dims = getval(valdims)
    J = @index(Global, Cartesian)
    @unroll for γ in dims
        adjoint = zero(eltype(u))
        @unroll for α in dims
            @unroll for β in dims
                Δuαβ = α == β ? Δu[β] : Δ[β]
                Aβα1 = A[β][α][1]
                Aβα2 = A[β][α][2]

                # 1
                I = J
                if α == γ && I in Iu[α]
                    uαβ2 = h
                    uβα2 = Aβα2[I[α]] * u[I, β] + Aβα1[I[α]+1] * u[I+e(α), β]
                    dφdu = -uαβ2 * uβα2 / Δuαβ[I[β]]
                    adjoint += φbar[I, α] * dφdu
                end

                # 2
                I = J - e(β)
                if α == γ && I in Iu[α]
                    uαβ2 = h
                    uβα2 = Aβα2[I[α]] * u[I, β] + Aβα1[I[α]+1] * u[I+e(α), β]
                    dφdu = -uαβ2 * uβα2 / Δuαβ[I[β]]
                    adjoint += φbar[I, α] * dφdu
                end

                # 3
                I = J
                if β == γ && I in Iu[α]
                    uαβ2 = h * u[I, α] + h * u[I+e(β), α]
                    uβα2 = Aβα2[I[α]]
                    dφdu = -uαβ2 * uβα2 / Δuαβ[I[β]]
                    adjoint += φbar[I, α] * dφdu
                end

                # 4
                I = J - e(α)
                if β == γ && I in Iu[α]
                    uαβ2 = h * u[I, α] + h * u[I+e(β), α]
                    uβα2 = Aβα1[I[α]+1]
                    dφdu = -uαβ2 * uβα2 / Δuαβ[I[β]]
                    adjoint += φbar[I, α] * dφdu
                end

                # 5
                I = J + e(β)
                if α == γ && I in Iu[α]
                    uαβ1 = h
                    uβα1 =
                        Aβα2[I[α]-(α==β)] * u[I-e(β), β] +
                        Aβα1[I[α]+(α!=β)] * u[I-e(β)+e(α), β]
                    dφdu = uαβ1 * uβα1 / Δuαβ[I[β]]
                    adjoint += φbar[I, α] * dφdu
                end

                # 6
                I = J
                if α == γ && I in Iu[α]
                    uαβ1 = h
                    uβα1 =
                        Aβα2[I[α]-(α==β)] * u[I-e(β), β] +
                        Aβα1[I[α]+(α!=β)] * u[I-e(β)+e(α), β]
                    dφdu = uαβ1 * uβα1 / Δuαβ[I[β]]
                    adjoint += φbar[I, α] * dφdu
                end

                # 7
                I = J + e(β)
                if β == γ && I in Iu[α]
                    uαβ1 = h * u[I-e(β), α] + h * u[I, α]
                    uβα1 = Aβα2[I[α]-(α==β)]
                    dφdu = uαβ1 * uβα1 / Δuαβ[I[β]]
                    adjoint += φbar[I, α] * dφdu
                end

                # 8
                I = J + e(β) - e(α)
                if β == γ && I in Iu[α]
                    uαβ1 = h * u[I-e(β), α] + h * u[I, α]
                    uβα1 = Aβα1[I[α]+(α!=β)]
                    dφdu = uαβ1 * uβα1 / Δuαβ[I[β]]
                    adjoint += φbar[I, α] * dφdu
                end
            end
        end
        ubar[J, γ] += adjoint
    end
end

"Compute diffusive term (differentiable version)."
diffusion(u, setup) = diffusion!(zero.(u), u, setup)

ChainRulesCore.rrule(::typeof(diffusion), u, setup) = (
    diffusion(u, setup),
    φ -> (NoTangent(), diffusion_adjoint!(zero(u), φ, setup), NoTangent()),
)

"""
Compute diffusive term (in-place version).
Add the result to `F`.
"""
function diffusion!(F, u, setup)
    (; grid, backend, workgroupsize, Re) = setup
    (; dimension, Δ, Δu, N, Iu) = grid
    D = dimension()
    e = Offset(D)
    visc = 1 / Re
    kernel = diffusion_kernel!(backend, workgroupsize)
    I0 = oneunit(CartesianIndex{D})
    kernel(F, u, visc, e, Δ, Δu, Iu, Val(1:D), I0; ndrange = N .- 2)
    F
end

@kernel function diffusion_kernel!(F, u, visc, e, Δ, Δu, Iu, valdims, I0)
    dims = getval(valdims)
    I = @index(Global, Cartesian)
    I = I + I0
    @unroll for α in dims
        f = F[I, α]
        if I ∈ Iu[α]
            @unroll for β in dims
                Δuαβ = α == β ? Δu[β] : Δ[β]
                Δa = β == α ? Δ[β][I[β]] : Δu[β][I[β]-1]
                Δb = β == α ? Δ[β][I[β]+1] : Δu[β][I[β]]
                ∂a = (u[I, α] - u[I-e(β), α]) / Δa
                ∂b = (u[I+e(β), α] - u[I, α]) / Δb
                f += visc * (∂b - ∂a) / Δuαβ[I[β]]
            end
        end
        F[I, α] = f
    end
end

function diffusion_adjoint!(u, φ, setup)
    (; grid, backend, workgroupsize, Re) = setup
    (; dimension, N, Δ, Δu, Iu) = grid
    D = dimension()
    e = Offset(D)
    visc = 1 / Re
    kernel = diffusion_adjoint_kernel!(backend, workgroupsize)
    kernel(u, φ, visc, e, Δ, Δu, Iu, Val(1:D); ndrange = N)
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

# "Compute convective and diffusive terms (differentiable version)."
# convectiondiffusion(u, setup) = convectiondiffusion!(zero.(u), u, setup)
#
# ChainRulesCore.rrule(::typeof(convectiondiffusion), u, setup) = (
#     convection(u, setup),
#     φ -> (
#         NoTangent(),
#         convectiondiffusion_adjoint!(vectorfield(setup), φ, setup),
#         NoTangent(),
#     ),
# )

"""
Compute convective and diffusive terms (in-place version).
Add the result to `F`.
"""
function convectiondiffusion!(F, u, setup)
    (; grid, backend, workgroupsize, Re) = setup
    (; dimension, Δ, Δu, N, A, Iu) = grid
    D = dimension()
    e = Offset(D)
    @assert size(u) == size(F) == (N..., D)
    visc = 1 / Re
    I0 = oneunit(CartesianIndex{D})
    kernel = convection_diffusion_kernel!(backend, workgroupsize)
    kernel(F, u, visc, Δ, Δu, A, Iu, e, Val(1:D), I0; ndrange = N .- 2)
    F
end

@kernel inbounds = true function convection_diffusion_kernel!(
    F,
    u,
    visc,
    Δ,
    Δu,
    A,
    Iu,
    e,
    valdims,
    I0,
)
    I = @index(Global, Cartesian)
    I = I + I0
    dims = getval(valdims)
    @unroll for α in dims
        f = F[I, α]
        if I ∈ Iu[α]
            @unroll for β in dims
                Δuαβ = α == β ? Δu[β] : Δ[β]
                uαβ1 = (u[I-e(β), α] + u[I, α]) / 2
                uαβ2 = (u[I, α] + u[I+e(β), α]) / 2
                uβα1 =
                    A[β][α][2][I[α]-(α==β)] * u[I-e(β), β] +
                    A[β][α][1][I[α]+(α!=β)] * u[I-e(β)+e(α), β]
                uβα2 = A[β][α][2][I[α]] * u[I, β] + A[β][α][1][I[α]+1] * u[I+e(α), β]
                uαuβ1 = uαβ1 * uβα1
                uαuβ2 = uαβ2 * uβα2
                ∂βuα1 = (u[I, α] - u[I-e(β), α]) / (β == α ? Δ[β][I[β]] : Δu[β][I[β]-1])
                ∂βuα2 = (u[I+e(β), α] - u[I, α]) / (β == α ? Δ[β][I[β]+1] : Δu[β][I[β]])
                f += (visc * (∂βuα2 - ∂βuα1) - (uαuβ2 - uαuβ1)) / Δuαβ[I[β]]
            end
        end
        F[I, α] = f
    end
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
function convection_diffusion_temp!(c, u, temp, setup)
    (; grid, backend, workgroupsize, temperature) = setup
    (; dimension, Δ, Δu, Np, Ip) = grid
    (; α4) = temperature
    D = dimension()
    e = Offset(D)
    I0 = getoffset(Ip)
    kernel = convection_diffusion_temp_kernel!(backend, workgroupsize)
    kernel(c, u, temp, α4, Δ, Δu, e, Val(1:D), I0; ndrange = Np)
    c
end

@kernel function convection_diffusion_temp_kernel!(c, u, temp, α4, Δ, Δu, e, valdims, I0)
    I = @index(Global, Cartesian)
    I = I + I0
    cI = zero(eltype(c))
    @unroll for β in getval(valdims)
        ∂T∂x1 = (temp[I] - temp[I-e(β)]) / Δu[β][I[β]-1]
        ∂T∂x2 = (temp[I+e(β)] - temp[I]) / Δu[β][I[β]]
        uT1 = u[I-e(β), β] * avg(temp, Δ, I - e(β), β)
        uT2 = u[I, β] * avg(temp, Δ, I, β)
        cI += (-(uT2 - uT1) + α4 * (∂T∂x2 - ∂T∂x1)) / Δ[β][I[β]]
    end
    c[I] += cI
end

"Compute dissipation term for the temperature equation (differentiable version)."
dissipation(u, setup) = dissipation!(scalarfield(setup), vectorfield(setup), u, setup)

function ChainRulesCore.rrule(::typeof(dissipation), u, setup)
    (; grid, backend, workgroupsize, Re, temperature) = setup
    (; dimension, Δ, N, Np, Ip) = grid
    (; α1, γ) = temperature
    D = dimension()
    e = Offset(D)
    d, d_pb = ChainRulesCore.rrule(diffusion, u, setup)
    φ = dissipation!(scalarfield(setup), d, u, setup)
    @kernel function ∂φ!(ubar, dbar, φbar, d, u, valdims)
        J = @index(Global, Cartesian)
        @unroll for β in getval(valdims)
            # Compute ubar
            a = zero(eltype(u))
            # 1
            I = J + e(β)
            I ∈ Ip && (a += Re * α1 / γ * d[I-e(β), β] / 2)
            # 2
            I = J
            I ∈ Ip && (a += Re * α1 / γ * d[I, β] / 2)
            ubar[J, β] += a

            # Compute dbar
            b = zero(eltype(u))
            # 1
            I = J + e(β)
            I ∈ Ip && (b += Re * α1 / γ * u[I-e(β), β] / 2)
            # 2
            I = J
            I ∈ Ip && (b += Re * α1 / γ * u[I, β] / 2)
            dbar[J, β] += b
        end
    end
    function dissipation_pullback(φbar)
        # Dφ/Du = ∂φ(u, d)/∂u + ∂φ(u, d)/∂d ⋅ ∂d(u)/∂u
        dbar = zero(u)
        ubar = zero(u)
        ∂φ!(backend, workgroupsize)(ubar, dbar, φbar, d, u, Val(1:D); ndrange = N)
        diffusion_adjoint!(ubar, dbar, setup)
        (NoTangent(), ubar, NoTangent())
    end
    φ, dissipation_pullback
end

"""
Compute dissipation term for the temperature equation (in-place version).
Add result to `diss`.
"""
function dissipation!(diss, diff, u, setup)
    (; grid, backend, workgroupsize, Re, temperature) = setup
    (; dimension, Δ, Np, Ip) = grid
    (; α1, γ) = temperature
    D = dimension()
    e = Offset(D)
    fill!(diff, 0)
    diffusion!(diff, u, setup)
    @kernel function interpolate!(diss, diff, u, I0, valdims)
        I = @index(Global, Cartesian)
        I += I0
        d = zero(eltype(diss))
        @unroll for β in getval(valdims)
            d += Re * α1 / γ * (u[I-e(β), β] * diff[I-e(β), β] + u[I, β] * diff[I, β]) / 2
        end
        diss[I] += d
    end
    I0 = first(Ip)
    I0 -= oneunit(I0)
    interpolate!(backend, workgroupsize)(diss, diff, u, I0, Val(1:D); ndrange = Np)
    diss
end

"""
Compute dissipation term
``2 \\nu \\langle S_{i j} S_{i j} \\rangle``
from strain-rate tensor (differentiable version).
"""
dissipation_from_strain(u, setup) = dissipation_from_strain!(scalarfield(setup), u, setup)

ChainRulesCore.rrule(::typeof(dissipation_from_strain), u, setup) =
    (dissipation_from_strain(u, setup), φ -> error("Not yet implemented"))

"Compute dissipation term from strain-rate tensor (in-place version)."
function dissipation_from_strain!(ϵ, u, setup)
    (; grid, backend, workgroupsize, Re) = setup
    (; Δ, Δu, Np, Ip) = grid
    visc = 1 / Re
    I0 = getoffset(Ip)
    kernel = dissipation_from_strain_kernel!(backend, workgroupsize)
    kernel(ϵ, u, visc, Δ, Δu, I0; ndrange = Np)
    ϵ
end

@kernel function dissipation_from_strain_kernel!(ϵ, u, visc, Δ, Δu, I0)
    I = @index(Global, Cartesian)
    I = I + I0
    S = strain(u, I, Δ, Δu)
    ϵ[I] = 2 * visc * sum(S .* S)
end

"Compute body force (differentiable version)."
function applybodyforce(u, t, setup)
    (; grid, bodyforce, issteadybodyforce) = setup
    (; dimension, xu) = grid
    D = dimension()
    if issteadybodyforce
        bodyforce
    else
        stack(map(α -> bodyforce.(α, xu[α]..., t), 1:D))
    end
end

# "Compute body force (differentiable version)."
# applybodyforce(u, t, setup) = applybodyforce!(zero.(u), u, t, setup)

# ChainRulesCore.rrule(::typeof(applybodyforce), u, t, setup) =
#     (applybodyforce(u, t, setup), φ -> error("Not yet implemented"))

"""
Compute body force (in-place version).
Add the result to `F`.
"""
function applybodyforce!(F, u, t, setup)
    (; grid, bodyforce, issteadybodyforce) = setup
    (; dimension, Iu, xu) = grid
    D = dimension()
    if issteadybodyforce
        F .+= bodyforce
    else
        for (α, Fα) in enumerate(eachslice(F; dims = D + 1))
            # xin = ntuple(
            #     β -> reshape(xu[α][β][Iu[α].indices[β]], ntuple(Returns(1), β - 1)..., :),
            #     D,
            # )
            # @. F[α][Iu[α]] += bodyforce(α, xin..., t)
            xin = ntuple(β -> reshape(xu[α][β], ntuple(Returns(1), β - 1)..., :), D)
            Fα .+= bodyforce.(α, xin..., t)
        end
    end
    F
end

"Compute gravity term (differentiable version)."
gravity(temp, setup) = gravity!(vectorfield(setup), temp, setup)

function ChainRulesCore.rrule(::typeof(gravity), temp, setup)
    (; grid, backend, workgroupsize, temperature) = setup
    (; dimension, Δ, N, Iu) = grid
    (; gdir, α2) = temperature
    backend = get_backend(temp)
    D = dimension()
    e = Offset(D)
    g = gravity(temp, setup)
    function gravity_pullback(φ)
        @kernel function g!(tempbar, φbar, valα)
            α = getval(valα)
            J = @index(Global, Cartesian)
            t = zero(eltype(tempbar))
            # 1
            I = J
            I ∈ Iu[α] && (t += α2 * Δ[α][I[α]+1] * φbar[I, α] / (Δ[α][I[α]] + Δ[α][I[α]+1]))
            # 2
            I = J - e(α)
            I ∈ Iu[α] && (t += α2 * Δ[α][I[α]] * φbar[I, α] / (Δ[α][I[α]] + Δ[α][I[α]+1]))
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
function gravity!(F, temp, setup)
    (; grid, backend, workgroupsize, temperature) = setup
    (; dimension, Δ, Nu, Iu) = grid
    (; gdir, α2) = temperature
    D = dimension()
    e = Offset(D)
    @kernel function g!(F, temp, ::Val{gdir}, I0) where {gdir}
        I = @index(Global, Cartesian)
        I = I + I0
        F[I, gdir] += α2 * avg(temp, Δ, I, gdir)
    end
    I0 = first(Iu[gdir])
    I0 -= oneunit(I0)
    g!(backend, workgroupsize)(F, temp, Val(gdir), I0; ndrange = Nu[gdir])
    F
end

"""
Right hand side of momentum equations, excluding pressure gradient
(differentiable version).
"""
function momentum(u, temp, t, setup)
    (; bodyforce) = setup
    d = diffusion(u, setup)
    c = convection(u, setup)
    F = @. d + c
    if !isnothing(bodyforce)
        f = applybodyforce(u, t, setup)
        F = @. F + f
    end
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
Right hand side of momentum equations, excluding pressure gradient
(in-place version).
"""
function momentum!(F, u, temp, t, setup)
    (; grid, bodyforce) = setup
    (; dimension) = grid
    D = dimension()
    fill!(F, 0)
    convectiondiffusion!(F, u, setup)
    isnothing(bodyforce) || applybodyforce!(F, u, t, setup)
    isnothing(temp) || gravity!(F, temp, setup)
    F
end

"Compute vorticity field (differentiable version)."
vorticity(u, setup) = vorticity!(
    setup.grid.dimension() == 2 ? scalarfield(setup) : vectorfield(setup),
    u,
    setup,
)

"Compute vorticity field (in-place version)."
vorticity!(ω, u, setup) = vorticity!(setup.grid.dimension, ω, u, setup)

# 2D version
function vorticity!(::Dimension{2}, ω, u, setup)
    (; grid, backend, workgroupsize) = setup
    (; dimension, Δu, N) = grid
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
    (; grid, backend, workgroupsize) = setup
    (; dimension, Δu, N) = grid
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
    α == β ? (u[I, α] - u[I-e(β), α]) / Δβ[I[β]] :
    (
        (u[I+e(β), α] - u[I, α]) / Δuβ[I[β]] +
        (u[I-e(α)+e(β), α] - u[I-e(α), α]) / Δuβ[I[β]] +
        (u[I, α] - u[I-e(β), α]) / Δuβ[I[β]-1] +
        (u[I-e(α), α] - u[I-e(α)-e(β), α]) / Δuβ[I[β]-1]
    ) / 4
@inline ∇(u, I::CartesianIndex{2}, Δ, Δu) =
    @SMatrix [∂x(u, I, α, β, Δ[β], Δu[β]) for α = 1:2, β = 1:2]
@inline ∇(u, I::CartesianIndex{3}, Δ, Δu) =
    @SMatrix [∂x(u, I, α, β, Δ[β], Δu[β]) for α = 1:3, β = 1:3]
@inline idtensor(u, ::CartesianIndex{2}) =
    @SMatrix [(α == β) * oneunit(eltype(u)) for α = 1:2, β = 1:2]
@inline idtensor(u, ::CartesianIndex{3}) =
    @SMatrix [(α == β) * oneunit(eltype(u)) for α = 1:3, β = 1:3]
@inline function strain(u, I, Δ, Δu)
    ∇u = ∇(u, I, Δ, Δu)
    (∇u + ∇u') / 2
end
@inline gridsize(Δ, I::CartesianIndex{D}) where {D} =
    sqrt(sum(ntuple(α -> Δ[α][I[α]]^2, D)))

"""
Compute Smagorinsky stress tensors `σ[I]` (in-place version).
The Smagorinsky constant `θ` should be a scalar between `0` and `1`.
"""
function smagtensor!(σ, u, θ, setup)
    # TODO: Combine with normal diffusion tensor
    (; grid, backend, workgroupsize) = setup
    (; Np, Ip, Δ, Δu) = grid
    @kernel function σ!(σ, u, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        S = strain(u, I, Δ, Δu)
        d = gridsize(Δ, I)
        eddyvisc = θ^2 * d^2 * sqrt(2 * sum(S .* S))
        σ[I] = 2 * eddyvisc * S
    end
    I0 = getoffset(Ip)
    σ!(backend, workgroupsize)(σ, u, I0; ndrange = Np)
    σ
end

"""
Compute divergence of a tensor with all components
in the pressure points (in-place version).
The stress tensors should be precomputed and stored in `σ`.
"""
function divoftensor!(s, σ, setup)
    (; grid, backend, workgroupsize) = setup
    (; dimension, Nu, Iu, Δ, Δu, A) = grid
    D = dimension()
    e = Offset(D)
    @kernel function s!(s, σ, ::Val{α}, ::Val{βrange}, I0) where {α,βrange}
        I = @index(Global, Cartesian)
        I = I + I0
        s[I, α] = zero(eltype(s[1]))
        # for β = 1:D
        @unroll for β in βrange
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
            s[I, α] += (σαβ2 - σαβ1) / Δuαβ[I[β]]
        end
    end
    for α = 1:D
        I0 = getoffset(Iu[α])
        s!(backend, workgroupsize)(s, σ, Val(α), Val(1:D), I0; ndrange = Nu[α])
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
    s = vectorfield(setup)
    function closure(u, θ)
        smagtensor!(σ, u, θ, setup)
        apply_bc_p!(σ, zero(T), setup)
        divoftensor!(s, σ, setup)
    end
end

"Compute symmetry tensor basis (differentiable version)."
function tensorbasis(u, setup)
    T = eltype(u)
    D = setup.grid.dimension()
    tensorbasis!(
        ntuple(α -> similar(u, SMatrix{D,D,T,D * D}, setup.grid.N), D == 2 ? 3 : 11),
        ntuple(α -> similar(u, setup.grid.N), D == 2 ? 2 : 5),
        u,
        setup,
    )
end

ChainRulesCore.rrule(::typeof(tensorbasis), u, setup) =
    (tensorbasis(u, setup), φ -> error("Not yet implemented"))

"""
Compute symmetry tensor basis `B[1]`-`B[11]` and invariants `V[1]`-`V[5]`,
as specified in [Silvis2017](@cite) in equations (9) and (11).
Note that `B[1]` corresponds to ``T_0`` in the paper, and `V` to ``I``.
"""
function tensorbasis!(B, V, u, setup)
    (; grid, backend, workgroupsize) = setup
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
    basis!(backend, workgroupsize)(B, V, u, I0; ndrange = Np)
    B, V
end

"Interpolate velocity to pressure points (differentiable version)."
interpolate_u_p(u, setup) = interpolate_u_p!(vectorfield(setup), u, setup)

"Interpolate velocity to pressure points (in-place version)."
function interpolate_u_p!(up, u, setup)
    (; grid, backend, workgroupsize) = setup
    (; dimension, Np, Ip) = grid
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
    setup.grid.dimension() == 2 ? scalarfield(setup) : vectorfield(setup),
    ω,
    setup,
)

"Interpolate vorticity to pressure points (in-place version)."
interpolate_ω_p!(ωp, ω, setup) = interpolate_ω_p!(setup.grid.dimension, ωp, ω, setup)

# 2D version
function interpolate_ω_p!(::Dimension{2}, ωp, ω, setup)
    (; grid, backend, workgroupsize) = setup
    (; dimension, Np, Ip) = grid
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
    (; grid, backend, workgroupsize) = setup
    (; dimension, Np, Ip) = grid
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

"""
Compute the ``D``-field [LiJiajia2019](@cite) given by

```math
D = \\frac{2 | \\nabla p |}{\\nabla^2 p}.
```

Differentiable version.
"""
Dfield(p, setup; kwargs...) =
    Dfield!(scalarfield(setup), vectorfield(setup), p, setup; kwargs...)

ChainRulesCore.rrule(::typeof(Dfield), p, setup; kwargs...) =
    (Dfield(p, setup; kwargs...), φ -> error("Not yet implemented"))

"Compute the ``D``-field (in-place version)."
function Dfield!(d, G, p, setup; ϵ = eps(eltype(p)))
    (; grid, backend, workgroupsize) = setup
    (; dimension, Np, Ip, Δ) = grid
    T = eltype(p)
    D = dimension()
    e = Offset(D)
    @kernel function D!(d, G, p, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        g = zero(eltype(p))
        for α = 1:D
            g += (G[I-e(α), α] + G[I, α])^2
        end
        lap = zero(eltype(p))
        # for α = 1:D
        #     lap += (G[I, α] - G[I-e(α), α]) / Δ[α][I[α]]
        # end
        if D == 2
            lap += (G[I, 1] - G[I-e(1), 1]) / Δ[1][I[1]]
            lap += (G[I, 2] - G[I-e(2), 2]) / Δ[2][I[2]]
        elseif D == 3
            lap += (G[I, 1] - G[I-e(1), 1]) / Δ[1][I[1]]
            lap += (G[I, 2] - G[I-e(2), 2]) / Δ[2][I[2]]
            lap += (G[I, 3] - G[I-e(3), 3]) / Δ[3][I[3]]
        end
        lap = lap > 0 ? max(lap, ϵ) : min(lap, -ϵ)
        # lap = abs(lap)
        d[I] = sqrt(g) / 2 / lap
    end
    pressuregradient!(G, p, setup)
    I0 = getoffset(Ip)
    D!(backend, workgroupsize)(d, G, p, I0; ndrange = Np)
    d
end

"""
Compute ``Q``-field [Jeong1995](@cite) given by

```math
Q = - \\frac{1}{2} \\sum_{α, β} \\frac{\\partial u^α}{\\partial x^β}
\\frac{\\partial u^β}{\\partial x^α}.
```

Differentiable version.
"""
Qfield(u, setup) = Qfield!(scalarfield(setup), u, setup)

ChainRulesCore.rrule(::typeof(Qfield), u, setup) =
    (Qfield(u, setup), φ -> error("Not yet implemented"))

"Compute the ``Q``-field (in-place version)."
function Qfield!(Q, u, setup)
    (; grid, backend, workgroupsize) = setup
    (; dimension, Np, Ip, Δ) = grid
    D = dimension()
    e = Offset(D)
    @kernel function Q!(Q, u, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        q = zero(eltype(Q))
        for α = 1:D, β = 1:D
            q -=
                (u[I, α] - u[I-e(β), α]) / Δ[β][I[β]] * (u[I, β] - u[I-e(α), β]) /
                Δ[α][I[α]] / 2
        end
        Q[I] = q
    end
    I0 = getoffset(Ip)
    Q!(backend, workgroupsize)(Q, u, I0; ndrange = Np)
    Q
end

"""
Compute the second eigenvalue of ``S^2 + R^2``,
as proposed by Jeong and Hussain [Jeong1995](@cite).

Differentiable version.
"""
eig2field(u, setup) = eig2field!(scalarfield(setup), u, setup)

"Compute the second eigenvalue of ``S^2 + R^2`` (in-place version)."
function eig2field!(λ, u, setup)
    (; grid, backend, workgroupsize) = setup
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
    λ!(backend, workgroupsize)(λ, u, I0; ndrange = Np)
    λ
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
    (; grid, backend, workgroupsize) = setup
    (; dimension, Np, Ip) = grid
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
    (; Ip) = setup.grid
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
function get_scale_numbers(u, setup)
    (; grid, Re) = setup
    (; dimension, Iu, Ip, Δ, Δu) = grid
    D = dimension()
    T = eltype(u)
    visc = 1 / Re
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
    ϵ = dissipation_from_strain(u, setup)
    ϵ = sum((Ω.*ϵ)[Ip]) / sum(Ω[Ip])
    η = (visc^3 / ϵ)^T(1 / 4)
    λ = sqrt(5 * visc / ϵ) * uavg
    Reλ = λ * uavg / sqrt(T(3)) / visc
    # TODO: L and τ
    L = nothing
    τ = nothing
    (; uavg, ϵ, η, λ, Reλ, L, τ)
end

# COV_EXCL_START
# Wrap a function to return `nothing`, because Enzyme can not handle vector return values.
function enzyme_wrap(
    f::Union{
        typeof(divergence!),
        typeof(pressuregradient!),
        typeof(convection!),
        typeof(diffusion!),
        typeof(applybodyforce!),
        typeof(gravity!),
        typeof(dissipation!),
        typeof(convection_diffusion_temp!),
        typeof(momentum!),
    },
)
    function wrapped_f(args...)
        f(args...)
        return nothing
    end
    return wrapped_f
end

function EnzymeRules.augmented_primal(
    config::RevConfigWidth{1},
    func::Union{
        Const{typeof(enzyme_wrap(divergence!))},
        Const{typeof(enzyme_wrap(pressuregradient!))},
        Const{typeof(enzyme_wrap(convection!))},
        Const{typeof(enzyme_wrap(diffusion!))},
        Const{typeof(enzyme_wrap(gravity!))},
    },
    ::Type{<:Const},
    y::Duplicated,
    u::Duplicated,
    setup::Const,
)
    primal = func.val(y.val, u.val, setup.val)
    if overwritten(config)[3]
        tape = copy(u.val)
    else
        tape = nothing
    end
    return AugmentedReturn(primal, nothing, tape)
end
function EnzymeRules.reverse(
    config::RevConfigWidth{1},
    func::Const{typeof(enzyme_wrap(divergence!))},
    dret,
    tape,
    y::Duplicated,
    u::Duplicated,
    setup::Const,
)
    adj = vectorfield(setup.val)
    divergence_adjoint!(adj, y.val, setup.val)
    u.dval .+= adj
    EnzymeCore.make_zero!(y.dval)
    return (nothing, nothing, nothing)
end
function EnzymeRules.reverse(
    config::RevConfigWidth{1},
    func::Const{typeof(enzyme_wrap(pressuregradient!))},
    dret,
    tape,
    y::Duplicated,
    p::Duplicated,
    setup::Const,
)
    adj = scalarfield(setup.val)
    pressuregradient_adjoint!(adj, y.val, setup.val)
    p.dval .+= adj
    EnzymeCore.make_zero!(y.dval)
    return (nothing, nothing, nothing)
end
function EnzymeRules.reverse(
    config::RevConfigWidth{1},
    func::Const{typeof(enzyme_wrap(convection!))},
    dret,
    tape,
    y::Duplicated,
    u::Duplicated,
    setup::Const,
)
    adj = zero(u.val)
    convection_adjoint!(adj, y.val, u.val, setup.val)
    u.dval .+= adj
    EnzymeCore.make_zero!(y.dval)
    return (nothing, nothing, nothing)
end
function EnzymeRules.reverse(
    config::RevConfigWidth{1},
    func::Const{typeof(enzyme_wrap(diffusion!))},
    dret,
    tape,
    y::Duplicated,
    u::Duplicated,
    setup::Const,
)
    adj = zero(u.val)
    diffusion_adjoint!(adj, y.val, setup.val)
    u.dval .+= adj
    EnzymeCore.make_zero!(y.dval)
    return (nothing, nothing, nothing)
end

function EnzymeRules.augmented_primal(
    config::RevConfigWidth{1},
    func::Union{Const{typeof(enzyme_wrap(applybodyforce!))}},
    ::Type{<:Const},
    y::Duplicated,
    u::Duplicated,
    t::Const,
    setup::Const,
)
    primal = func.val(y.val, u.val, t.val, setup.val)
    if overwritten(config)[3]
        tape = copy(u.val)
    else
        tape = nothing
    end
    return AugmentedReturn(primal, nothing, tape)
end
function EnzymeRules.reverse(
    config::RevConfigWidth{1},
    func::Const{typeof(enzyme_wrap(applybodyforce!))},
    dret,
    tape,
    y::Duplicated,
    u::Duplicated,
    t::Const,
    setup::Const,
)
    @warn "bodyforce Enzyme-AD tested only for issteadybodyforce=true"
    adj = setup.val.bodyforce
    u.dval .+= adj .* y.dval
    EnzymeCore.make_zero!(y.dval)
    return (nothing, nothing, nothing, nothing)
end

function EnzymeRules.reverse(
    config::RevConfigWidth{1},
    func::Const{typeof(enzyme_wrap(gravity!))},
    dret,
    tape,
    y::Duplicated,
    temp::Duplicated,
    setup::Const,
)
    (; grid, backend, workgroupsize, temperature) = setup.val
    (; dimension, Δ, N, Iu) = grid
    (; gdir, α2) = temperature
    backend = get_backend(temp.val)
    D = dimension()
    e = Offset(D)
    function gravity_pullback(φ)
        @kernel function g!(tempbar, φbar, valα)
            α = getval(valα)
            J = @index(Global, Cartesian)
            t = zero(eltype(tempbar))
            # 1
            I = J
            I ∈ Iu[α] && (t += α2 * Δ[α][I[α]+1] * φbar[I, α] / (Δ[α][I[α]] + Δ[α][I[α]+1]))
            # 2
            I = J - e(α)
            I ∈ Iu[α] && (t += α2 * Δ[α][I[α]] * φbar[I, α] / (Δ[α][I[α]] + Δ[α][I[α]+1]))
            tempbar[J] = t
        end
        tempbar = zero(temp.val)
        g!(backend, workgroupsize)(tempbar, φ, Val(gdir); ndrange = N)
        tempbar
    end
    adj = gravity_pullback(y.val)
    temp.dval .+= adj
    EnzymeCore.make_zero!(y.dval)
    return (nothing, nothing, nothing)
end

function EnzymeRules.augmented_primal(
    config::RevConfigWidth{1},
    func::Union{
        Const{typeof(enzyme_wrap(dissipation!))},
        Const{typeof(enzyme_wrap(convection_diffusion_temp!))},
    },
    ::Type{<:Const},
    y::Duplicated,
    x1::Duplicated,
    x2::Duplicated,
    setup::Const,
)
    primal = func.val(y.val, x1.val, x2.val, setup.val)
    if overwritten(config)[3]
        tape = copy(x2.val)
    else
        tape = nothing
    end
    return AugmentedReturn(primal, nothing, tape)
end
function EnzymeRules.reverse(
    config::RevConfigWidth{1},
    func::Const{typeof(enzyme_wrap(dissipation!))},
    dret,
    tape,
    y::Duplicated,
    d::Duplicated,
    u::Duplicated,
    setup::Const,
)
    (; grid, backend, workgroupsize, Re, temperature) = setup.val
    (; dimension, N, Ip) = grid
    (; α1, γ) = temperature
    D = dimension()
    e = Offset(D)
    @kernel function ∂φ!(ubar, dbar, φbar, d, u, valdims)
        J = @index(Global, Cartesian)
        @unroll for β in getval(valdims)
            # Compute ubar
            a = zero(eltype(u))
            # 1
            I = J + e(β)
            I ∈ Ip && (a += Re * α1 / γ * d[I-e(β), β] / 2)
            # 2
            I = J
            I ∈ Ip && (a += Re * α1 / γ * d[I, β] / 2)
            ubar[J, β] += a

            # Compute dbar
            b = zero(eltype(u))
            # 1
            I = J + e(β)
            I ∈ Ip && (b += Re * α1 / γ * u[I-e(β), β] / 2)
            # 2
            I = J
            I ∈ Ip && (b += Re * α1 / γ * u[I, β] / 2)
            dbar[J, β] += b
        end
    end
    function dissipation_pullback(φbar)
        # Dφ/Du = ∂φ(u, d)/∂u + ∂φ(u, d)/∂d ⋅ ∂d(u)/∂u
        dbar = zero(u.val)
        ubar = zero(u.val)
        ∂φ!(backend, workgroupsize)(ubar, dbar, φbar, d.val, u.val, Val(1:D); ndrange = N)
        diffusion_adjoint!(ubar, dbar, setup.val)
        ubar
    end
    adj = dissipation_pullback(y.val)
    u.dval .+= adj
    EnzymeCore.make_zero!(y.dval)
    return (nothing, nothing, nothing, nothing)
end

function EnzymeRules.reverse(
    config::RevConfigWidth{1},
    func::Const{typeof(enzyme_wrap(convection_diffusion_temp!))},
    dret,
    tape,
    y::Duplicated,
    temp::Duplicated,
    u::Duplicated,
    setup::Const,
)
    @error "convection_diffusion_temp Enzyme-AD not yet implemented"
end

function EnzymeRules.augmented_primal(
    config::RevConfigWidth{1},
    func::Union{Const{typeof(enzyme_wrap(momentum!))}},
    ::Type{<:Const},
    y::Duplicated,
    x1::Duplicated,
    x2::Duplicated,
    x3::Duplicated,
    t::Const,
    setup::Const,
)
    @error "momentum Enzyme-AD not yet implemented"
end
function EnzymeRules.reverse(
    config::RevConfigWidth{1},
    func::Const{typeof(enzyme_wrap(momentum!))},
    dret,
    tape,
    y::Duplicated,
    u::Duplicated,
    temp::Duplicated,
    t::Const,
    setup::Const,
)
    @error "momentum Enzyme-AD not yet implemented"
end
# COV_EXCL_STOP
