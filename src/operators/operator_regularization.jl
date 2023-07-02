"""
    operator_regularization(dimension, grid, operators)

Build regularization matrices.
"""
function operator_regularization end

# 2D version
function operator_regularization(::Dimension{2}, grid, operators)
    (; Ω, indu, indv) = grid
    (; Dux, Duy, Dvx, Dvy) = operators
    (; Su_ux, Su_uy, Sv_vx, Sv_vy) = operators

    Δ = max_size(grid)
    α_reg = Δ^2 / 16

    Ωu⁻¹ = 1 ./ Ω[indu]
    Ωv⁻¹ = 1 ./ Ω[indv]

    # Diffusive matrices in finite-difference setting, without viscosity
    Diffu_f = Diagonal(Ωu⁻¹) * (Dux * Su_ux + Duy * Su_uy)
    Diffv_f = Diagonal(Ωv⁻¹) * (Dvx * Sv_vx + Dvy * Sv_vy)

    (; Diffu_f, Diffv_f, α_reg)
end

# 3D version
function operator_regularization(::Dimension{3}, grid, operators)
    (; Ω, indu, indv, indw) = grid
    (; Dux, Duy, Duz, Dvx, Dvy, Dvz, Dwx, Dwy, Dwz) = operators
    (; Su_ux, Su_uy, Su_uz) = operators
    (; Sv_vx, Sv_vy, Sv_vz) = operators
    (; Sw_wx, Sw_wy, Sw_wz) = operators

    Δ = max_size(grid)
    α_reg = Δ^2 / 16

    Ωu⁻¹ = 1 ./ Ω[indu]
    Ωv⁻¹ = 1 ./ Ω[indv]
    Ωw⁻¹ = 1 ./ Ω[indw]

    # Diffusive matrices in finite-difference setting, without viscosity
    Diffu_f = Diagonal(Ωu⁻¹) * (Dux * Su_ux + Duy * Su_uy + Duz * Su_uz)
    Diffv_f = Diagonal(Ωv⁻¹) * (Dvx * Sv_vx + Dvy * Sv_vy + Dvz * Sv_vz)
    Diffw_f = Diagonal(Ωw⁻¹) * (Dwx * Sw_wx + Dwy * Sw_wy + Dwz * Sw_wz)

    (; Diffu_f, Diffv_f, Diffw_f, α_reg)
end
