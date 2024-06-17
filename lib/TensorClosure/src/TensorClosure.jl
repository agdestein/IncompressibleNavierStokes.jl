module TensorClosure

using IncompressibleNavierStokes

"""
    rot2(u)

Rotate scalar field `u` by 90 degrees through the transformation
``(x, y) \\mapsto (-y, x)``.
"""
function rot2 end

function rot2(u)
    nx, ny = size(u)
    @assert ndims(u) == 2
    @assert nx == ny
    i = (1:nx)'
    j = nx:-1:1
    I = CartesianIndex.(i, j)
    u[I]
end

function rot2(u::Tuple, setup)
    # u = IncompressibleNavierStokes.apply_bc_u(u, eltype(u[1])(0), setup)
    ru = rot2(u[1])
    rv = rot2(u[2])
    # Velocities are to the right in a volume, but should now be to the left
    rv = circshift(rv, (-1, 0))
    r = (-rv, ru)
    IncompressibleNavierStokes.apply_bc_u(r, eltype(u[1])(0), setup)
end

export rot2

end
