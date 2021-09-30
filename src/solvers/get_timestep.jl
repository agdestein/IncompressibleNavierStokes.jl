"""
estimate time step based on eigenvalues of operators,
using Gershgorin
"""
function get_timestep(setup)
    # for explicit methods only
    if method ∈ [1, 2, 5, 81, 82]
        ## Convective part
        Cu =
            Cux * spdiagm(Iu_ux * uₕ + yIu_ux) * Au_ux +
            Cuy * spdiagm(Iv_uy * vₕ + yIv_uy) * Au_uy
        Cv =
            Cvx * spdiagm(Iu_vx * uₕ + yIu_vx) * Av_vx +
            Cvy * spdiagm(Iv_vy * vₕ + yIv_vy) * Av_vy

        test = spdiagm(Ωu⁻¹) * Cu
        sum_conv_u = abs(test) * ones(Nu) - diag(abs(test)) - diag(test)
        test = spdiagm(Ωv⁻¹) * Cv
        sum_conv_v = abs(test) * ones(Nv) - diag(abs(test)) - diag(test)
        λ_conv = max([max(sum_conv_u) max(sum_conv_v)])

        ## diffusive part
        test = spdiagm(Ωu⁻¹) * Diffu
        sum_diff_u = abs(test) * ones(Nu) - diag(abs(test)) - diag(test)
        test = spdiagm(Ωv⁻¹) * Diffv
        sum_diff_v = abs(test) * ones(Nv) - diag(abs(test)) - diag(test)
        λ_diff = max(maximum(sum_diff_u), maximum(sum_diff_v))

        # based on max. value of stability region
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

        # based on max. value of stability region (not a very good indication
        # for the methods that do not include the imaginary axis)
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
