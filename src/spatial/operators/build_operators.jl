"""
    build_operators!(setup)

Build discrete operators.
"""
function build_operators!(setup)
    @unpack regularization, visc = setup.case

    # Mesh
    operator_mesh!(setup)

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
    if regularization != "no"
        operator_regularization!(setup)
    end

    # Classical turbulence modelling via the diffusive term
    if visc == "keps"
        # Averaging viscosity to cell faces of u- and v- volumes
        ke_viscosity!(setup)

        # k-e operators
        ke_convection!(setup)
        ke_diffusion!(setup)
        ke_production!(setup)
    elseif visc ∈ ["qr", "LES", "ML"]
        operator_turbulent_diffusion!(setup)
    elseif visc == "laminar"
        # do nothing, these are constructed in operator_convection_diffusion
    else
        error("Wrong value for visc parameter")
    end

    # post-processing
    operator_postprocessing!(setup)

    setup
end
