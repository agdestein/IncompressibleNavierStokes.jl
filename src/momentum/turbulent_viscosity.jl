"""
    turbulent_viscosity(model, S_abs)

Compute turbulent viscosity based on `S_abs`.
"""
function turbulent_viscosity end
# FIXME: get parameters `δx`, `α` etc

function turbulent_viscosity(model::SmagorinskyModel, S_abs)
    @unpack C_S = model
    filter_length = δx
    C_S^2 * filter_length^2 * S_abs
end

function turbulent_viscosity(model::QRModel, S_abs)
    C_d = δx^2 / 8
    C_d * 1//2 * S_abs * (1 - α / C_d)^2
end

function turbulent_viscosity(model::MixingLengthModel, S_abs)
    @unpack lm = model
    lm^2 * S_abs
end
