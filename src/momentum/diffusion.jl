diffusion(m::LaminarModel, V, setup) = 1 / m.Re * (setup.operators.Diff * V)

"""
    diffusion!(model, d, ∇d, V, setup; getJacobian = false)

Evaluate diffusive terms `d` and optionally Jacobian `∇d = ∂d/∂V` using viscosity model `model`.
"""
function diffusion! end

function diffusion!(m::LaminarModel, d, V, setup) 
    mul!(d, setup.operators.Diff, V)
    @. d = 1 / m.Re * d
end

function diffusion_jacobian!(m::LaminarModel, ∇d, V, setup) 
    @. ∇d = 1 / m.Re * setup.operators.Diff
end

function diffusion!(
    model::Union{QRModel,SmagorinskyModel,MixingLengthModel},
    d,
    ∇d,
    V,
    setup;
    getJacobian = false,
)
    (; indu, indv, indw) = setup.grid
    (; Dux, Duy, Duz, Dvx, Dvy, Dvz, Dwx, Dwy, Dwz) = setup.operators
    (; Su_ux, Su_uy, Su_vx, Sv_vx, Sv_vy, Sv_uy) = setup.operators
    (; Aν_ux, Aν_uy, Aν_vx, Aν_vy) = setup.operators

    du = @view d[indu]
    dv = @view d[indv]
    # dw = @view d[indw]

    # Get components of strain tensor and its magnitude;
    # The magnitude S_abs is evaluated at pressure points
    S11, S12, S21, S22, S_abs, S_abs_u, S_abs_v = strain_tensor(V, setup; getJacobian)

    # Turbulent viscosity at all pressure points
    ν_t = turbulent_viscosity(model, setup, S_abs)

    # To compute the diffusion, we need ν_t at ux, uy, vx and vy locations
    # This means we have to reverse the process of strain_tensor.m: go
    # from pressure points back to the ux, uy, vx, vy locations
    ν_t_ux, ν_t_uy, ν_t_vx, ν_t_vy = interpolate_ν(ν_t, setup)

    # Now the total diffusive terms (laminar + turbulent) is as follows
    # Note that the factor 2 is because
    # Tau = 2*(ν+ν_t)*S(u), with S(u) = 1/2*(∇u + (∇u)^T)

    ν = 1 / model.Re # Molecular viscosity

    du .= Dux * (2 .* (ν .+ ν_t_ux) .* S11[:]) .+ Duy * (2 .* (ν .+ ν_t_uy) .* S12[:])
    dv .= Dvx * (2 .* (ν .+ ν_t_vx) .* S21[:]) .+ Dvy * (2 .* (ν .+ ν_t_vy) .* S22[:])

    if getJacobian
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
