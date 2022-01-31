"""
    λ_diff_max(method)

Get maximum value of stability region for the diffusion operator.
"""
function λ_diff_max end

λ_diff_max(::AbstractODEMethod) = 1
λ_diff_max(m::OneLegMethod) = 4 * m.β / (2 * m.β + 1)


"""
    λ_conv_max(method)


Get maximum value of stability region for the convection operator (not a very good
indication for the methods that do not include the imaginary axis)
"""
function λ_conv_max end

λ_conv_max(::AbstractODEMethod) = 1
