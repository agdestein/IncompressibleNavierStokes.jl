"""
    filter_convection(u, diff_matrix, bc, α)

Construct filter for convective terms.

Non-mutating/allocating/out-of-place version.

See also [`filter_convection!`](@ref).
"""
function filter_convection(u, diff_matrix, bc, α)
    u + α * (diff_matrix * u + bc)
end

"""
    filter_convection!(ū, u, diff_matrix, bc, α)

Construct filter for convective terms.

Mutating/non-allocating/in-place version.

See also [`filter_convection`](@ref).
"""
function filter_convection!(ū, u, diff_matrix, bc, α)
    # ū = u + α * (diff_matrix * u + bc)
    mul!(ū, diff_matrix, u)
    @. ū = u + α * (ū + bc)
    ū
end
