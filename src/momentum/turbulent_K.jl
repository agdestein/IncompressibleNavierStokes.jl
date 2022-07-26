"""
    turbulent_K(model, setup)

Compute the constant part of the turbulent viscosity.
"""
function turbulent_K end

turbulent_K(m::SmagorinskyModel, setup) = m.C_s^2 * max_size(setup.grid)^2

function turbulent_K(::QRModel, setup)
    (; α) = setup.operators
    Δ = max_size(setup.grid) 
    C_d = Δ^2 / 8
    C_d * 1//2 * (1 - α / C_d)^2
end

turbulent_K(m::MixingLengthModel, setup) = m.lm^2
