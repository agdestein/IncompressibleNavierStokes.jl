"""
    filter_convection(u, diff_matrix, bc, α)

Construct filter for convective terms
"""
function filter_convection(u, diff_matrix, bc, α)
    u_filtered = u + α*(diff_matrix*u + bc);
end
