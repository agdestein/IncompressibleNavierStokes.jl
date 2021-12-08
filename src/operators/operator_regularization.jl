## Regularization matrices
function operator_regularization!(setup)
    # TODO: Identify correct references
    @unpack indu, indv = setup.grid
    @unpack Ω, Dux, Duy, Duz, Dvx, Dvy, Dvz, Dwx, Dwy, Dwz = setup.discretization
    @unpack Su_ux, Su_uy, Su_uz = setup.discretization
    @unpack Sv_vx, Sv_vy, Sv_vz = setup.discretization
    @unpack Sw_wx, Sw_wy, Sw_wz = setup.discretization
    @unpack ySu_ux, ySu_uy, ySu_uz = setup.discretization
    @unpack ySv_vx, ySv_vy, ySv_vz = setup.discretization
    @unpack ySw_wx, ySw_wy, ySw_wz = setup.discretization

    Δ = max_size(grid)
    α = 1 / 16 * Δ^2

    Ωu⁻¹ = 1 ./ Ω[indu]
    Ωv⁻¹ = 1 ./ Ω[indv]
    Ωw⁻¹ = 1 ./ Ω[indw]

    # Diffusive matrices in finite-difference setting, without viscosity
    Diffu_f = spdiagm(Ωu⁻¹) * (Dux * Su_ux + Duy * Su_uy + Duz * Su_uz)
    Diffv_f = spdiagm(Ωv⁻¹) * (Dvx * Sv_vx + Dvy * Sv_vy + Dvz * Sv_vz)
    Diffw_f = spdiagm(Ωw⁻¹) * (Dwx * Sw_vx + Dwy * Sw_wy + Dwz * Sw_wz)

    yDiffu_f = Ωu⁻¹ .* (Dux * ySu_ux + Duy * ySu_uy + Duz * ySu_uz)
    yDiffv_f = Ωv⁻¹ .* (Dvx * ySv_vx + Dvy * ySv_vy + Dvz * ySv_vz)
    yDiffw_f = Ωw⁻¹ .* (Dwx * ySw_wx + Dwy * ySw_wy + Dwz * ySw_wz)

    setup
end
