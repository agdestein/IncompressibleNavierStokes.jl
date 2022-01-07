# Based on max. value of stability region
function λ_diff_max end

λ_diff_max(m::OneLegMethod, setup) = 4 * m.β / (2 * m.β + 1)
λ_diff_max(::AdamsBashforthCrankNicolsonMethod, setup) = 1

# Based on max. value of stability region (not a very good indication
# for the methods that do not include the imaginary axis)
function λ_conv_max end

λ_conv_max(::OneLegMethod, setup) = 1
λ_conv_max(::AdamsBashforthCrankNicolsonMethod, setup) = 1
