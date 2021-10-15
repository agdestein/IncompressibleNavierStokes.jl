"""
    get_timestep(setup)

Estimate time step based on eigenvalues of operators, using Gershgorin.
"""
function get_timestep(setup)
    @unpack Nu, Nv = setup.grid
    @unpack Iu_ux, Iu_vx, Iv_uy, Iv_vy = setup.discretization
    @unpack yIu_ux, yIu_vx, yIv_uy, yIv_vy = setup.discretization
    @unpack Au_ux, Au_uy, Av_vx, Av_vy = setup.discretization
    @unpack CFL, β = setup.time

    # For explicit methods only
    if method ∈ [1, 2, 5, 81, 82]
        ## Convective part
        Cu =
            Cux * spdiagm(Iu_ux * uₕ + yIu_ux) * Au_ux +
            Cuy * spdiagm(Iv_uy * vₕ + yIv_uy) * Au_uy
        Cv =
            Cvx * spdiagm(Iu_vx * uₕ + yIu_vx) * Av_vx +
            Cvy * spdiagm(Iv_vy * vₕ + yIv_vy) * Av_vy

        test = spdiagm(Ωu⁻¹) * Cu
        sum_conv_u = abs.(test) * ones(Nu) - Diagonal(abs.(test)) - Diagonal(test)
        test = spdiagm(Ωv⁻¹) * Cv
        sum_conv_v = abs.(test) * ones(Nv) - Diagonal(abs.(test)) - Diagonal(test)
        λ_conv = max(maximum(sum_conv_u), maximum(sum_conv_v))

        ## Diffusive part
        test = spdiagm(Ωu⁻¹) * Diffu
        sum_diff_u = abs.(test) * ones(Nu) - Diagonal(abs.(test)) - Diagonal(test)
        test = spdiagm(Ωv⁻¹) * Diffv
        sum_diff_v = abs.(test) * ones(Nv) - Diagonal(abs.(test)) - Diagonal(test)
        λ_diff = max(maximum(sum_diff_u), maximum(sum_diff_v))

        # Based on max. value of stability region
        if method == 5
            λ_diff_max = 4 * β / (2 * β + 1)
        elseif method == 1
            λ_diff_max = 2
        elseif method == 2
            λ_diff_max = 1
        elseif method ∈ [81, 82]
            λ_diff_max = 2.78
        end

        Δt_diff = λ_diff_max / λ_diff

        # Based on max. value of stability region (not a very good indication
        # For the methods that do not include the imaginary axis)
        if method ∈ [81, 82]
            λ_conv_max = 2 * sqrt(2)
        elseif method == 11
            λ_conv_max = sqrt(3)
        else
            λ_conv_max = 1
        end
        Δt_conv = λ_conv_max / λ_conv
        Δt = CFL * min(Δt_conv, Δt_diff)
    end

    Δt
end
