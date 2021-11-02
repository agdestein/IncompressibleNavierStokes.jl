"""
    turbulent_viscosity(model, setup, S_abs)

Compute turbulent viscosity based on `S_abs`.
"""
function turbulent_viscosity end

turbulent_viscosity(model, setup, S_abs) = turbulent_K(model, setup) * S_abs

