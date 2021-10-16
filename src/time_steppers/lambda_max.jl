# Based on max. value of stability region
function λ_diff_max end

λ_diff_max(ts::OneLegStepper, setup) = 4 * ts.β / (2 * ts.β + 1)
λ_diff_max(::AdamsBashforthCrankNicolsonStepper, setup) = 1

# Based on max. value of stability region (not a very good indication
# For the methods that do not include the imaginary axis)
function λ_conv_max end

λ_conv_max(::OneLegStepper, setup) = 1
λ_conv_max(::AdamsBashforthCrankNicolsonStepper, setup) = 1
