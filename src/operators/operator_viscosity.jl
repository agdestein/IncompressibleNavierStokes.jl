"""
    operator_viscosity(model, grid, boundary_conditions)

Classical turbulence modelling via the diffusive term
"""
function operator_viscosity end

operator_viscosity(::LaminarModel, grid, boundary_conditions) = (;)

function operator_viscosity(::KEpsilonModel, grid, boundary_conditions)
    # Averaging viscosity to cell faces of u- and v- volumes
    ke_vis = ke_viscosity(grid, boundary_conditions)

    # K-e operators
    ke_con = ke_convection(grid, boundary_conditions)
    ke_dif = ke_diffusion(grid, boundary_conditions)
    ke_pro = ke_production(grid, boundary_conditions)

    (; ke_vis..., ke_con..., ke_dif..., ke_pro...)
end

function operator_viscosity(
    ::Union{QRModel,SmagorinskyModel,MixingLengthModel},
    grid,
    boundary_conditions,
)
    operator_turbulent_diffusion(grid, boundary_conditions)
end
