"""
estimate time step based on eigenvalues of operators,
using Gershgorin
"""
function set_timestep(setup)
    # for explicit methods only
    if method ∈ [1, 2, 5, 81, 82]
        ## Convective part
        Cu =
            Cux * spdiagm(Iu_ux * uh + yIu_ux) * Au_ux +
            Cuy * spdiagm(Iv_uy * vh + yIv_uy) * Au_uy
        Cv =
            Cvx * spdiagm(Iu_vx * uh + yIu_vx) * Av_vx +
            Cvy * spdiagm(Iv_vy * vh + yIv_vy) * Av_vy

        test = spdiagm(Omu_inv) * Cu
        sum_conv_u = abs(test) * ones(Nu) - diag(abs(test)) - diag(test)
        test = spdiagm(Omv_inv) * Cv
        sum_conv_v = abs(test) * ones(Nv) - diag(abs(test)) - diag(test)
        λ_conv = max([max(sum_conv_u) max(sum_conv_v)])

        ## diffusive part
        test = spdiagm(Omu_inv) * Diffu
        sum_diff_u = abs(test) * ones(Nu) - diag(abs(test)) - diag(test)
        test = spdiagm(Omv_inv) * Diffv
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

        dt_diff = λ_diff_max / λ_diff

        # based on max. value of stability region (not a very good indication
        # for the methods that do not include the imaginary axis)
        if method ∈ [81, 82]
            λ_conv_max = 2 * sqrt(2)
        elseif method == 11
            λ_conv_max = sqrt(3)
        else
            λ_conv_max = 1
        end
        dt_conv = λ_conv_max / λ_conv
        dt = CFL * min(dt_conv, dt_diff)

    end

    dt
end
