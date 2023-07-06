"""
    diffusion!(model, V, setup; bc_vectors, get_jacobian = false)

Evaluate diffusive terms `d` and optionally Jacobian `∇d = ∂d/∂V` using viscosity model `model`.

Non-mutating/allocating/out-of-place version.

See also [`diffusion!`](@ref).
"""
function diffusion end

diffusion(m, V, setup; kwargs...) = diffusion(setup.grid.dimension, m, V, setup; kwargs...)

function diffusion(m::LaminarModel, V, setup; bc_vectors, get_jacobian = false)
    (; Re) = m
    (; Diff) = setup.operators
    (; yDiff) = bc_vectors

    d = 1 ./ Re .* (Diff * V .+ yDiff)

    if get_jacobian
        ∇d = 1 / Re * Diff
    else
        ∇d = nothing
    end

    d, ∇d
end

# 2D version
function diffusion(
    ::Dimension{2},
    model::Union{QRModel,SmagorinskyModel,MixingLengthModel},
    V,
    setup;
    bc_vectors,
    get_jacobian = false,
)
    (; indu, indv) = setup.grid
    (; Dux, Duy, Dvx, Dvy) = setup.operators
    (; Su_ux, Su_uy, Su_vx, Sv_vx, Sv_vy, Sv_uy) = setup.operators
    (; Aν_ux, Aν_uy, Aν_vx, Aν_vy) = setup.operators
    (; yAν_ux, yAν_uy, yAν_vx, yAν_vy) = bc_vectors

    # Get components of strain tensor and its magnitude;
    # The magnitude S_abs is evaluated at pressure points
    S11, S12, S21, S22, S_abs, S_abs_u, S_abs_v =
        strain_tensor(V, setup; bc_vectors, get_jacobian)

    # Turbulent viscosity at all pressure points
    ν_t = turbulent_viscosity(model, setup, S_abs)

    # To compute the diffusion, we need ν_t at ux, uy, vx and vy locations
    # This means we have to reverse the process of `strain_tensor`: go
    # from pressure points back to the ux, uy, vx, vy locations
    ν_t_ux = Aν_ux * ν_t + yAν_ux
    ν_t_uy = Aν_uy * ν_t + yAν_uy
    ν_t_vx = Aν_vx * ν_t + yAν_vx
    ν_t_vy = Aν_vy * ν_t + yAν_vy

    # Now the total diffusive terms (laminar + turbulent) is as follows
    # Note that the factor 2 is because
    # Tau = 2*(ν+ν_t)*S(u), with S(u) = 1/2*(∇u + (∇u)^T)

    # Molecular viscosity
    ν = 1 / model.Re

    du = Dux * (2 .* (ν .+ ν_t_ux) .* S11[:]) .+ Duy * (2 .* (ν .+ ν_t_uy) .* S12[:])
    dv = Dvx * (2 .* (ν .+ ν_t_vx) .* S21[:]) .+ Dvy * (2 .* (ν .+ ν_t_vy) .* S22[:])

    if get_jacobian
        # Freeze ν_t, i.e. we skip the derivative of ν_t wrt V in the Jacobian
        Jacu1 =
            Dux * 2 * spdiagm(ν .+ ν_t_ux) * Su_ux +
            Duy * 2 * spdiagm(ν .+ ν_t_uy) * 1 / 2 * Su_uy
        Jacu2 = Duy * 2 * spdiagm(ν .+ ν_t_uy) * 1 / 2 * Sv_uy
        Jacv1 = Dvx * 2 * spdiagm(ν .+ ν_t_vx) * 1 / 2 * Su_vx
        Jacv2 =
            Dvx * 2 * spdiagm(ν .+ ν_t_vx) * 1 / 2 * Sv_vx +
            Dvy * 2 * spdiagm(ν .+ ν_t_vy) * Sv_vy
        Jacu = [Jacu1 Jacu2]
        Jacv = [Jacv1 Jacv2]

        K = turbulent_K(model, setup)

        tmpu1 =
            2 * Dux * spdiagm(S11) * Aν_ux * S_abs_u +
            2 * Duy * spdiagm(S12) * Aν_uy * S_abs_u
        tmpu2 = 2 * Duy * spdiagm(S12) * Aν_uy * S_abs_v
        tmpv1 = 2 * Dvx * spdiagm(S21) * Aν_vx * S_abs_u
        tmpv2 =
            2 * Dvx * spdiagm(S21) * Aν_vx * S_abs_v +
            2 * Dvy * spdiagm(S22) * Aν_vy * S_abs_v
        Jacu += K * [tmpu1 tmpu2]
        Jacv += K * [tmpv1 tmpv2]

        ∇d = [Jacu; Jacv]
    else
        ∇d = nothing
    end

    d = [du; dv]

    d, ∇d
end

# 3D version
function diffusion(
    ::Dimension{3},
    model::Union{QRModel,SmagorinskyModel,MixingLengthModel},
    V,
    setup;
    bc_vectors,
    get_jacobian = false,
)
    error("Not implemented")
end

"""
    diffusion!(model, d, ∇d, V, setup; bc_vectors, get_jacobian = false)

Evaluate diffusive terms `d` and optionally Jacobian `∇d = ∂d/∂V` using viscosity model `model`.
"""
function diffusion! end

diffusion!(m, d, ∇d, V, setup; kwargs...) =
    diffusion!(setup.grid.dimension, m, d, ∇d, V, setup; kwargs...)

function diffusion!(m::LaminarModel, d, ∇d, V, setup; bc_vectors, get_jacobian = false)
    (; Re) = m
    (; Diff) = setup.operators
    (; yDiff) = bc_vectors

    # d = 1 ./ Re .* (Diff * V .+ yDiff)
    mul!(d, Diff, V)
    @. d = 1 / Re * (d + yDiff)

    get_jacobian && (@. ∇d = 1 / Re * Diff)

    d, ∇d
end

# 2D version
function diffusion!(
    ::Dimension{2},
    model::Union{QRModel,SmagorinskyModel,MixingLengthModel},
    d,
    ∇d,
    V,
    setup;
    bc_vectors,
    get_jacobian = false,
)
    (; indu, indv) = setup.grid
    (; Dux, Duy, Dvx, Dvy) = setup.operators
    (; Su_ux, Su_uy, Su_vx, Sv_vx, Sv_vy, Sv_uy) = setup.operators
    (; Aν_ux, Aν_uy, Aν_vx, Aν_vy) = setup.operators
    (; yAν_ux, yAν_uy, yAν_vx, yAν_vy) = bc_vectors

    du = @view d[indu]
    dv = @view d[indv]

    # Get components of strain tensor and its magnitude;
    # The magnitude S_abs is evaluated at pressure points
    S11, S12, S21, S22, S_abs, S_abs_u, S_abs_v =
        strain_tensor(V, setup; bc_vectors, get_jacobian)

    # Turbulent viscosity at all pressure points
    ν_t = turbulent_viscosity(model, setup, S_abs)

    # To compute the diffusion, we need ν_t at ux, uy, vx and vy locations
    # This means we have to reverse the process of `strain_tensor`: go
    # from pressure points back to the ux, uy, vx, vy locations
    ν_t_ux = Aν_ux * ν_t + yAν_ux
    ν_t_uy = Aν_uy * ν_t + yAν_uy
    ν_t_vx = Aν_vx * ν_t + yAν_vx
    ν_t_vy = Aν_vy * ν_t + yAν_vy

    # Now the total diffusive terms (laminar + turbulent) is as follows
    # Note that the factor 2 is because
    # Tau = 2*(ν+ν_t)*S(u), with S(u) = 1/2*(∇u + (∇u)^T)

    # Molecular viscosity
    ν = 1 / model.Re

    du .= Dux * (2 .* (ν .+ ν_t_ux) .* S11[:]) .+ Duy * (2 .* (ν .+ ν_t_uy) .* S12[:])
    dv .= Dvx * (2 .* (ν .+ ν_t_vx) .* S21[:]) .+ Dvy * (2 .* (ν .+ ν_t_vy) .* S22[:])

    if get_jacobian
        # Freeze ν_t, i.e. we skip the derivative of ν_t wrt V in the Jacobian
        Jacu1 =
            Dux * 2 * spdiagm(ν .+ ν_t_ux) * Su_ux +
            Duy * 2 * spdiagm(ν .+ ν_t_uy) * 1 / 2 * Su_uy
        Jacu2 = Duy * 2 * spdiagm(ν .+ ν_t_uy) * 1 / 2 * Sv_uy
        Jacv1 = Dvx * 2 * spdiagm(ν .+ ν_t_vx) * 1 / 2 * Su_vx
        Jacv2 =
            Dvx * 2 * spdiagm(ν .+ ν_t_vx) * 1 / 2 * Sv_vx +
            Dvy * 2 * spdiagm(ν .+ ν_t_vy) * Sv_vy
        Jacu = [Jacu1 Jacu2]
        Jacv = [Jacv1 Jacv2]

        K = turbulent_K(model, setup)

        tmpu1 =
            2 * Dux * spdiagm(S11) * Aν_ux * S_abs_u +
            2 * Duy * spdiagm(S12) * Aν_uy * S_abs_u
        tmpu2 = 2 * Duy * spdiagm(S12) * Aν_uy * S_abs_v
        tmpv1 = 2 * Dvx * spdiagm(S21) * Aν_vx * S_abs_u
        tmpv2 =
            2 * Dvx * spdiagm(S21) * Aν_vx * S_abs_v +
            2 * Dvy * spdiagm(S22) * Aν_vy * S_abs_v
        Jacu += K * [tmpu1 tmpu2]
        Jacv += K * [tmpv1 tmpv2]

        ∇d .= [Jacu; Jacv]
    end

    d, ∇d
end

# 3D version
function diffusion!(
    ::Dimension{3},
    model::Union{QRModel,SmagorinskyModel,MixingLengthModel},
    d,
    ∇d,
    V,
    setup;
    bc_vectors,
    get_jacobian = false,
)
    error("Not implemented")
end
