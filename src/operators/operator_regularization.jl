## Regularization matrices
function operator_regularization!(setup)
    # TODO: Identify correct references
    @unpack indu, indv = setup.grid
    @unpack Ω, Dux, Duy, Dvx, Dvy, Su_ux, Su_uy, Sv_vx, Sv_vy, ySu_ux, ySu_uy, ySv_vx, ySv_vy = setup.discretization
    α = 1 / 16 * Δx^2

    Ωu⁻¹ = 1 ./ Ω[indu]
    Ωv⁻¹ = 1 ./ Ω[indv]

    # Diffusive matrices in finite-difference setting, without viscosity
    Diffu_f = spdiagm(Ωu⁻¹) * (Dux * Su_ux + Duy * Su_uy)
    Diffv_f = spdiagm(Ωv⁻¹) * (Dux * Sv_vx + Dvy * Sv_vy)

    yDiffu_f = Ωu⁻¹ .* (Dux * ySu_ux + Duy * ySu_uy)
    yDiffv_f = Ωv⁻¹ .* (Dvx * ySv_vx + Dvy * ySv_vy)

    setup
end
