"""
    build_operators!(setup)

Build discrete operators.
"""
function build_operators!(setup)
    (; viscosity_model) = setup

    # Averaging operators
    operator_averaging!(setup)

    # Interpolation operators
    operator_interpolation!(setup)

    # Divergence (u, v)-> p and gradient p->(u, v) operator
    operator_divergence!(setup)

    # Convection operators on u- and v- centered volumes
    operator_convection_diffusion!(setup)

    # Turbulence

    # Regularization modelling - this changes the convective term
    operator_regularization!(setup)

    # Classical turbulence modelling via the diffusive term
    operator_viscosity!(viscosity_model, setup)

    # Post-processing
    operator_postprocessing!(setup)

    setup
end
