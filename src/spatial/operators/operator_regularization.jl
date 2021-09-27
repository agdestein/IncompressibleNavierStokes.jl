## regularization matrices
function operator_regularization!(setup)
    α = 1 / 16 * deltax^2

    # diffusive matrices in finite-difference setting, without viscosity
    Diffu_f = spdiagm(Omu_inv) * (Dux * Su_ux + Duy * Su_uy)
    Diffv_f = spdiagm(Omv_inv) * (Dux * Sv_vx + Dvy * Sv_vy)

    yDiffu_f = Omu_inv .* (Dux * ySu_ux + Duy * ySu_uy)
    yDiffv_f = Omv_inv .* (Dvx * ySv_vx + Dvy * ySv_vy)

    setup
end
