"""
    Operators(grid, boundary_conditions, viscosity_model)

Build operators.
"""
function Operators(grid, boundary_conditions, viscosity_model)
    (; dimension) = grid

    # Averaging operators
    op_ave = operator_averaging(dimension, grid, boundary_conditions)

    # Interpolation operators
    op_int = operator_interpolation(dimension, grid, boundary_conditions)

    # Divergence (u, v) -> p and gradient p -> (u, v) operator
    op_div = operator_divergence(dimension, grid, boundary_conditions)

    # Convection operators on u- and v- centered volumes
    op_con = operator_convection_diffusion(dimension, grid, boundary_conditions)

    # Regularization modelling - this changes the convective term
    op_reg = operator_regularization(dimension, grid, op_con)

    # Classical turbulence modelling via the diffusive term
    op_vis = operator_viscosity(viscosity_model, dimension, grid, boundary_conditions)

    # Post-processing
    op_pos = operator_postprocessing(dimension, grid, boundary_conditions)

    (;
        op_ave...,
        op_int...,
        op_div...,
        op_con...,
        op_reg...,
        op_vis...,
        op_pos...,
    )
end
