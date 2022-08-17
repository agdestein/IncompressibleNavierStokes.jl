"""
    build_operators(grid)

Build discrete operators.
"""
function build_operators(grid::Grid{T}) where {T}
    # Averaging operators
    op_ave = operator_averaging(grid)

    # Interpolation operators
    op_int = operator_interpolation(grid)

    # Divergence (u, v)-> p and gradient p->(u, v) operator
    op_div = operator_divergence(grid)

    # Convection operators on u- and v- centered volumes
    op_con = operator_convection_diffusion(grid)

    # Classical turbulence modelling via the diffusive term
    op_tur = operator_turbulent_diffusion(grid)

    # Post-processing
    op_pos = operator_postprocessing(grid)

    Operators{T}(;
        op_ave...,
        op_int...,
        op_div...,
        op_con...,
        op_tur...,
        op_pos...,
    )
end
