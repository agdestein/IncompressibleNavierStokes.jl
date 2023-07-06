"""
    operator_filter(grid, boundary_conditions)

Construct filtering operator.
"""
operator_filter(grid, boundary_conditions) = operator_filter(grid.dimension, grid, boundary_conditions)

# 2D version
function operator_filter(::Dimension{2}, grid, boundary_conditions, Nx_coarse, Ny_coarse)
    (; hx, hy) = grid
end

# 3D version
function operator_filter(::Dimension{3}, grid, boundary_conditions)
end
