"""
    get_velocity(V, t, setup)

Get velocity values at pressure points. Interpolate velocities to pressure positions using
`BMx` and `BMy` (and `BMz`), constructed in operator_divergence.jl.
"""
function get_velocity end

get_velocity(setup, V, t) = get_velocity(setup.grid.dimension, setup, V, t)

# 2D version
function get_velocity(::Dimension{2}, setup, V, t)
    (; grid, operators) = setup
    (; Npx, Npy, indu, indv) = grid
    (; Au_ux, Av_vy, Bup, Bvp) = operators

    # Evaluate boundary conditions at current time
    bc_vectors = get_bc_vectors(setup, t)
    (; yAu_ux, yAv_vy) = bc_vectors

    uh = @view V[indu]
    vh = @view V[indv]

    up = reshape(Bup * (Au_ux * uh + yAu_ux), Npx, Npy)
    vp = reshape(Bvp * (Av_vy * vh + yAv_vy), Npx, Npy)

    up, vp
end

# 3D version
function get_velocity(::Dimension{3}, setup, V, t)
    (; grid, operators) = setup
    (; Au_ux, Av_vy, Aw_wz, Bup, Bvp, Bwp) = operators
    (; Npx, Npy, Npz, indu, indv, indw) = grid

    # Evaluate boundary conditions at current time
    bc_vectors = get_bc_vectors(setup, t)
    (; yAu_ux, yAv_vy, yAw_wz) = bc_vectors

    uh = @view V[indu]
    vh = @view V[indv]
    wh = @view V[indw]

    up = reshape(Bup * (Au_ux * uh + yAu_ux), Npx, Npy, Npz)
    vp = reshape(Bvp * (Av_vy * vh + yAv_vy), Npx, Npy, Npz)
    wp = reshape(Bwp * (Aw_wz * wh + yAw_wz), Npx, Npy, Npz)

    up, vp, wp
end
