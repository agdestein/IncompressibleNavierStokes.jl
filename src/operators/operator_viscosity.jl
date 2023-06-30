"""
    operator_viscosity(model, grid, boundary_conditions)

Classical turbulence modelling via the diffusive term
"""
function operator_viscosity end

operator_viscosity(::LaminarModel, grid, boundary_conditions) = (;)

function operator_viscosity(
    ::Union{QRModel,SmagorinskyModel,MixingLengthModel},
    grid,
    boundary_conditions,
)
    operator_turbulent_diffusion(grid, boundary_conditions)
end
