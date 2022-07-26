"""
    get_timestep(stepper, cfl_number)

Estimate time step based on eigenvalues of operators, using Gershgorin.
"""
function get_timestep end

# 2D version
function get_timestep(stepper::AbstractTimeStepper{T,2}, cfl_number) where {M,T}
    (; setup, method, V) = stepper
    (; grid, operators) = setup
    (; NV, indu, indv, Ω⁻¹) = grid
    (; Diff) = operators
    (; Cux, Cuy, Cvx, Cvy) = operators
    (; Au_ux, Au_uy, Av_vx, Av_vy) = operators
    (; Iu_ux, Iu_vx, Iv_uy, Iv_vy) = operators

    uₕ = @view V[indu]
    vₕ = @view V[indv]

    # For explicit methods only
    if isexplicit(method)
        # Convective part
        Cu = Cux * spdiagm(Iu_ux * uₕ) * Au_ux + Cuy * spdiagm(Iv_uy * vₕ) * Au_uy
        Cv = Cvx * spdiagm(Iu_vx * uₕ) * Av_vx + Cvy * spdiagm(Iv_vy * vₕ) * Av_vy
        C = blockdiag(Cu, Cv)
        test = spdiagm(Ω⁻¹) * C
        sum_conv = abs.(test) * ones(NV) - diag(abs.(test)) - diag(test)
        λ_conv = maximum(sum_conv)

        # Based on max. value of stability region (not a very good indication
        # For the methods that do not include the imaginary axis)
        Δt_conv = λ_conv_max(method) / λ_conv

        ## Diffusive part
        test = Diagonal(Ω⁻¹) * Diff
        sum_diff = abs.(test) * ones(NV) - diag(abs.(test)) - diag(test)
        λ_diff = maximum(sum_diff)

        # Based on max. value of stability region
        Δt_diff = λ_diff_max(method) / λ_diff

        Δt = cfl_number * min(Δt_conv, Δt_diff)
    end

    Δt
end

# 3D version
function get_timestep(stepper::AbstractTimeStepper{T,3}, cfl_number) where {M,T}
    (; setup, method, V) = stepper
    (; grid, operators) = setup
    (; NV, indu, indv, indw, Ω⁻¹) = grid
    (; Diff) = operators
    (; Cux, Cuy, Cuz, Cvx, Cvy, Cvz, Cwx, Cwy, Cwz) = operators
    (; Au_ux, Au_uy, Au_uz, Av_vx, Av_vy, Av_vz, Aw_wx, Aw_wy, Aw_wz) = operators
    (; Iu_ux, Iu_vx, Iu_wx, Iv_uy, Iv_vy, Iv_wy, Iw_uz, Iw_vz, Iw_wz) = operators

    uₕ = @view V[indu]
    vₕ = @view V[indv]
    wₕ = @view V[indw]

    # For explicit methods only
    if isexplicit(method)
        # Convective part
        Cu =
            Cux * spdiagm(Iu_ux * uₕ) * Au_ux +
            Cuy * spdiagm(Iv_uy * vₕ) * Au_uy +
            Cuz * spdiagm(Iw_uz * wₕ) * Au_uz
        Cv =
            Cvx * spdiagm(Iu_vx * uₕ) * Av_vx +
            Cvy * spdiagm(Iv_vy * vₕ) * Av_vy +
            Cvz * spdiagm(Iw_vz * wₕ) * Av_vz
        Cw =
            Cwx * spdiagm(Iu_wx * uₕ) * Aw_wx +
            Cwy * spdiagm(Iv_wy * vₕ) * Aw_wy +
            Cwz * spdiagm(Iw_wz * wₕ) * Aw_wz
        C = blockdiag(Cu, Cv, Cw)
        test = spdiagm(Ω⁻¹) * C
        sum_conv = abs.(test) * ones(NV) - diag(abs.(test)) - diag(test)
        λ_conv = maximum(sum_conv)

        # Based on max. value of stability region (not a very good indication
        # For the methods that do not include the imaginary axis)
        Δt_conv = λ_conv_max(method) / λ_conv

        ## Diffusive part
        test = Diagonal(Ω⁻¹) * Diff
        sum_diff = abs.(test) * ones(NV) - diag(abs.(test)) - diag(test)
        λ_diff = maximum(sum_diff)

        # Based on max. value of stability region
        Δt_diff = λ_diff_max(method) / λ_diff

        Δt = cfl_number * min(Δt_conv, Δt_diff)
    end

    Δt
end
