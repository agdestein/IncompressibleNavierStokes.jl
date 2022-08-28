"""
    lambda_diff_max(method)

Get maximum value of stability region for the diffusion operator.
"""
function lambda_diff_max end

lambda_diff_max(::AbstractODEMethod) = 1
lambda_diff_max(m::OneLegMethod) = 4 * m.β / (2 * m.β + 1)

"""
    lambda_conv_max(method)


Get maximum value of stability region for the convection operator (not a very good
indication for the methods that do not include the imaginary axis)
"""
function lambda_conv_max end

lambda_conv_max(::AbstractODEMethod) = 1
