"""
    operator_viscosity!(model, setup)

Classical turbulence modelling via the diffusive term
"""
function operator_viscosity! end

operator_viscosity!(::LaminarModel, setup) = nothing

function operator_viscosity!(::KEpsilonModel, setup)
    # Averaging viscosity to cell faces of u- and v- volumes
    ke_viscosity!(setup)

    # K-e operators
    ke_convection!(setup)
    ke_diffusion!(setup)
    ke_production!(setup)
end

function operator_viscosity!(::Union{QRModel,SmagorinskyModel,MixingLengthModel}, setup)
    operator_turbulent_diffusion!(setup)
end
