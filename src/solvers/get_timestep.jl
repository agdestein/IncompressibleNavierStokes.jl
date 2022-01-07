"""
    get_timestep(setup)

Estimate time step based on eigenvalues of operators, using Gershgorin.
"""
function get_timestep(stepper)
    (; setup, method, V) = stepper
    (; NV, Ω⁻¹, indu, indv, indw) = setup.grid
    (; Diff) = setup.operators
    (; Cux, Cuy, Cuz, Cvx, Cvy, Cvz, Cwx, Cwy, Cwz) = setup.operators
    (; Au_ux, Au_uy, Au_uz, Av_vx, Av_vy, Av_vz, Aw_wx, Aw_wy, Aw_wz) = setup.operators
    (; Iu_ux, Iu_vx, Iu_wx, Iv_uy, Iv_vy, Iv_wy, Iw_uz, Iw_vz, Iw_wz) = setup.operators
    (; yIu_ux, yIu_vx, yIu_wx) = setup.operators
    (; yIv_uy, yIv_vy, yIv_wy) = setup.operators
    (; yIw_uz, yIw_vz, yIw_wz) = setup.operators
    (; CFL) = method

    uₕ = @view V[indu]
    vₕ = @view V[indv]
    wₕ = @view V[indw]

    # For explicit methods only
    if isexplicit(method)
        # Convective part
        Cu =
            Cux * spdiagm(Iu_ux * uₕ + yIu_ux) * Au_ux +
            Cuy * spdiagm(Iv_uy * vₕ + yIv_uy) * Au_uy +
            Cuz * spdiagm(Iw_uz * wₕ + yIw_uz) * Au_uz
        Cv =
            Cvx * spdiagm(Iu_vx * uₕ + yIu_vx) * Av_vx +
            Cvy * spdiagm(Iv_vy * vₕ + yIv_vy) * Av_vy +
            Cvz * spdiagm(Iw_vz * wₕ + yIw_vz) * Av_vz
        Cw =
            Cwx * spdiagm(Iu_wx * uₕ + yIu_wx) * Aw_wx +
            Cwy * spdiagm(Iv_wy * vₕ + yIv_wy) * Aw_wy +
            Cwz * spdiagm(Iw_wz * wₕ + yIw_wz) * Aw_wz
        C = blockdiag(Cu, Cv, Cw)
        test = spdiagm(Ω⁻¹) * C
        sum_conv = abs.(test) * ones(NV) - diag(abs.(test)) - diag(test)
        λ_conv = maximum(sum_conv)

        # Based on max. value of stability region (not a very good indication
        # For the methods that do not include the imaginary axis)
        Δt_conv = λ_conv_max / λ_conv

        ## Diffusive part
        test = Diagonal(Ω⁻¹) * Diff
        sum_diff = abs.(test) * ones(NV) - diag(abs.(test)) - diag(test)
        λ_diff = maximum(sum_diff)

        # Based on max. value of stability region
        Δt_diff = λ_diff_max(method, setup) / λ_diff

        Δt = CFL * min(Δt_conv, Δt_diff)
    end

    Δt
end
