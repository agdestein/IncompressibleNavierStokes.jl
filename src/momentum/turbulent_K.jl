"""
    turbulent_K(model, setup)

Compute the constant part of the turbulent viscosity.
"""
function turbulent_K end

turbulent_K(model::SmagorinskyModel, setup) = model.C_s^2 * Δmax_size(setup.grid)^2

function turbulent_K(model::QRModel, setup)
    # FIXME: Is this the correct `α`?
    @unpack α = setup.discretization
    Δ = max_size(setup.grid) 
    C_d = Δ^2 / 8
    C_d * 1//2 * (1 - α / C_d)^2
end

turbulent_K(model::MixingLengthModel, setup) = model.lm^2
