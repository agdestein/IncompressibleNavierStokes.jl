"""
    operator_viscosity(model, dimension, grid, boundary_conditions)

Classical turbulence modelling via the diffusive term
"""
function operator_viscosity end

operator_viscosity(::LaminarModel, dimension, grid, boundary_conditions) = (;)

function operator_viscosity(
    ::Union{QRModel,SmagorinskyModel,MixingLengthModel},
    dimension,
    grid,
    boundary_conditions,
)
    operator_turbulent_diffusion(dimension, grid, boundary_conditions)
end
