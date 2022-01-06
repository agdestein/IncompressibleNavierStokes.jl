"""
    get_timestep(setup)

Estimate time step based on eigenvalues of operators, using Gershgorin.
"""
function get_timestep(setup, method)
    @unpack NV, Ω⁻¹= setup.grid
    (; Iu_ux, Iu_vx, Iv_uy, Iv_vy) = setup.operators
    (; yIu_ux, yIu_vx, yIv_uy, yIv_vy) = setup.operators
    (; Au_ux, Au_uy, Av_vx, Av_vy) = setup.operators
    (; CFL) = setup.time

    # For explicit methods only
    if isexplicit(method)
        ## Convective part
        Cu =
            Cux * spdiagm(Iu_ux * uₕ + yIu_ux) * Au_ux +
            Cuy * spdiagm(Iv_uy * vₕ + yIv_uy) * Au_uy
        Cv =
            Cvx * spdiagm(Iu_vx * uₕ + yIu_vx) * Av_vx +
            Cvy * spdiagm(Iv_vy * vₕ + yIv_vy) * Av_vy

        C = blockdiag(Cu, Cv)

        test = spdiagm(Ω⁻¹) * C
        sum_conv = abs.(test) * ones(NV) - diag(abs.(test)) - diag(test)
        λ_conv = max(maximum(sum_conv_u), maximum(sum_conv_v))

        ## Diffusive part
        test = Diagonal(Ω⁻¹) * Diff
        sum_diff = abs.(test) * ones(NV) - diag(abs.(test)) - diag(test)
        λ_diff = maximum(sum_diff)

        # Based on max. value of stability region
        Δt_diff = λ_diff_max(method, setup) / λ_diff

        # Based on max. value of stability region (not a very good indication
        # For the methods that do not include the imaginary axis)

        Δt_conv = λ_conv_max / λ_conv
        Δt = CFL * min(Δt_conv, Δt_diff)
    end

    Δt
end
