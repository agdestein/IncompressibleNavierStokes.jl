"""
    get_dimension(grid)

Get dimension of grid.
"""
get_dimension(::Grid{T,N}) where {T,N} = N
