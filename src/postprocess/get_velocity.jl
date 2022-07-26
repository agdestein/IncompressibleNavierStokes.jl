"""
    get_velocity(V, t, setup)

Get velocity values at pressure points. Interpolate velocities to pressure
positions using `BMx` and `BMy` (and `BMz`), constructed in
operator_divergence.jl.
"""
function get_velocity end

# 2D version
function get_velocity(V, t, setup::Setup{T,2}) where {T}
    (; Au_ux, Av_vy, Bup, Bvp) = setup.operators
    (; Npx, Npy, indu, indv) = setup.grid

    uh = @view V[indu]
    vh = @view V[indv]

    up = reshape(Bup * (Au_ux * uh), Npx, Npy)
    vp = reshape(Bvp * (Av_vy * vh), Npx, Npy)

    up, vp
end

# 3D version
function get_velocity(V, t, setup::Setup{T,3}) where {T}
    (; Au_ux, Av_vy, Aw_wz, Bup, Bvp, Bwp) = setup.operators
    (; Npx, Npy, Npz, indu, indv, indw) = setup.grid

    uh = @view V[indu]
    vh = @view V[indv]
    wh = @view V[indw]

    up = reshape(Bup * (Au_ux * uh), Npx, Npy, Npz)
    vp = reshape(Bvp * (Av_vy * vh), Npx, Npy, Npz)
    wp = reshape(Bwp * (Aw_wz * wh), Npx, Npy, Npz)

    up, vp, wp
end
