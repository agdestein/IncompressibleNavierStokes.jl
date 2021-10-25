"""
    turbulent_viscosity(model, S_abs)

Compute turbulent viscosity based on `S_abs`.
"""
function turbulent_viscosity end
# FIXME: get parameters `δx`, `α` etc

function turbulent_viscosity(model::SmagorinskyModel, setup, S_abs)
    @unpack C_s = model
    @unpack Δx = setup.grid
    filter_length = Δx
    C_s^2 * filter_length^2 * S_abs
end

function turbulent_viscosity(model::QRModel, setup, S_abs)
    @unpack α = setup.discretization
    @unpack Δx = setup.grid
    C_d = Δx^2 / 8
    C_d * 1//2 * S_abs * (1 - α / C_d)^2
end

function turbulent_viscosity(model::MixingLengthModel, setup, S_abs)
    @unpack lm = model
    lm^2 * S_abs
end
