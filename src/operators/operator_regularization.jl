"""
    operator_regularization!(setup)

Build regularization matrices.
"""
function operator_regularization! end

# 2D version
function operator_regularization!(setup::Setup{T,2}) where {T}
    (; grid, operators) = setup
    (; Ω, indu, indv) = grid
    (; Dux, Duy, Dvx, Dvy) = operators
    (; Su_ux, Su_uy, Sv_vx, Sv_vy) = operators

    Δ = max_size(grid)
    α = 1 / 16 * Δ^2

    Ωu⁻¹ = 1 ./ Ω[indu]
    Ωv⁻¹ = 1 ./ Ω[indv]

    # Diffusive matrices in finite-difference setting, without viscosity
    Diffu_f = Diagonal(Ωu⁻¹) * (Dux * Su_ux + Duy * Su_uy)
    Diffv_f = Diagonal(Ωv⁻¹) * (Dvx * Sv_vx + Dvy * Sv_vy)

    @pack! operators = Diffu_f, Diffv_f, α

    setup
end

# 3D version
function operator_regularization!(setup::Setup{T,3}) where {T}
    (; grid, operators) = setup
    (; Ω, indu, indv, indw) = grid
    (; Dux, Duy, Duz, Dvx, Dvy, Dvz, Dwx, Dwy, Dwz) = operators
    (; Su_ux, Su_uy, Su_uz) = operators
    (; Sv_vx, Sv_vy, Sv_vz) = operators
    (; Sw_wx, Sw_wy, Sw_wz) = operators

    Δ = max_size(grid)
    α = 1 / 16 * Δ^2

    Ωu⁻¹ = 1 ./ Ω[indu]
    Ωv⁻¹ = 1 ./ Ω[indv]
    Ωw⁻¹ = 1 ./ Ω[indw]

    # Diffusive matrices in finite-difference setting, without viscosity
    Diffu_f = Diagonal(Ωu⁻¹) * (Dux * Su_ux + Duy * Su_uy + Duz * Su_uz)
    Diffv_f = Diagonal(Ωv⁻¹) * (Dvx * Sv_vx + Dvy * Sv_vy + Dvz * Sv_vz)
    Diffw_f = Diagonal(Ωw⁻¹) * (Dwx * Sw_wx + Dwy * Sw_wy + Dwz * Sw_wz)

    @pack! operators = Diffu_f, Diffv_f, Diffw_f, α

    setup
end
