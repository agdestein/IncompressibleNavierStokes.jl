"""
    operator_viscosity(model, grid, bc)

Classical turbulence modelling via the diffusive term
"""
function operator_viscosity end

operator_viscosity(::LaminarModel, grid, bc) = (;)

function operator_viscosity(::KEpsilonModel, grid, bc)
    # Averaging viscosity to cell faces of u- and v- volumes
    ke_vis = ke_viscosity(grid, bc)

    # K-e operators
    ke_con = ke_convection(grid, bc)
    ke_dif = ke_diffusion(grid, bc)
    ke_pro = ke_production(grid, bc)

    (; ke_vis..., ke_con..., ke_dif..., ke_pro...)
end

function operator_viscosity(::Union{QRModel,SmagorinskyModel,MixingLengthModel}, grid, bc)
    operator_turbulent_diffusion(grid, bc)
end
