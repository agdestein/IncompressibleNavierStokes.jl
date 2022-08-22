"""
    build_operators(grid, bc, viscosity_model)

Build discrete operators.
"""
function build_operators(grid::Grid{T}, bc, viscosity_model) where {T}
    # Averaging operators
    op_ave = operator_averaging(grid, bc)

    # Interpolation operators
    op_int = operator_interpolation(grid, bc)

    # Divergence (u, v) -> p and gradient p -> (u, v) operator
    op_div = operator_divergence(grid, bc)

    # Convection operators on u- and v- centered volumes
    op_con = operator_convection_diffusion(grid, bc, viscosity_model)

    # Regularization modelling - this changes the convective term
    op_reg = operator_regularization(grid, op_con)

    # Classical turbulence modelling via the diffusive term
    op_vis = operator_viscosity(viscosity_model, grid, bc)

    # Post-processing
    op_pos = operator_postprocessing(grid, bc)

    Operators{T}(;
        op_ave...,
        op_int...,
        op_div...,
        op_con...,
        op_reg...,
        op_vis...,
        op_pos...,
    )
end
