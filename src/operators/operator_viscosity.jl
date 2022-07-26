"""
    operator_viscosity!(model, setup)

Classical turbulence modelling via the diffusive term
"""
function operator_viscosity! end

operator_viscosity!(::LaminarModel, setup) = nothing
operator_viscosity!(::Union{QRModel,SmagorinskyModel,MixingLengthModel}, setup) =
    operator_turbulent_diffusion!(setup)
