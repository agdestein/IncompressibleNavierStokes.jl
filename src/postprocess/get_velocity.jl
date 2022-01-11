"""
    get_velocity(V, t, setup)

Get velocity values at pressure points. Interpolate velocities to pressure positions using
`BMx` and `BMy` (and `BMz`), constructed in operator_divergence.jl.
"""
function get_velocity end

function get_velocity(V, t, setup::Setup{T,2}) where {T}
    # Evaluate boundary conditions at current time
    set_bc_vectors!(setup, t)

    (; Au_ux, yAu_ux, Av_vy, yAv_vy, Bup, Bvp) = setup.operators
    (; Npx, Npy, indu, indv) = setup.grid

    uh = @view V[indu]
    vh = @view V[indv]

    up = reshape(Bup * (Au_ux * uh + yAu_ux), Npx, Npy)
    vp = reshape(Bvp * (Av_vy * vh + yAv_vy), Npx, Npy)

    up, vp
end

function get_velocity(V, t, setup::Setup{T,3}) where {T}
    # Evaluate boundary conditions at current time
    set_bc_vectors!(setup, t)

    (; Au_ux, yAu_ux, Av_vy, yAv_vy, Aw_wz, yAw_wz, Bup, Bvp, Bwp) = setup.operators
    (; Npx, Npy, Npz, indu, indv, indw) = setup.grid

    uh = @view V[indu]
    vh = @view V[indv]
    wh = @view V[indw]

    up = reshape(Bup * (Au_ux * uh + yAu_ux), Npx, Npy, Npz)
    vp = reshape(Bvp * (Av_vy * vh + yAv_vy), Npx, Npy, Npz)
    wp = reshape(Bwp * (Aw_wz * wh + yAw_wz), Npx, Npy, Npz)

    up, vp, wp
end
