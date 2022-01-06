"""
    get_velocity(V, t, setup)

Get velocity values at pressure points. Interpolate velocities to pressure positions using BMx and BMy (constructed in operator_divergence.jl).
"""
function get_velocity(V, t, setup)
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

    qp = .√(up .^ 2 .+ vp .^ 2 .+ wp .^ 2)

    ## get wake profiles
    # u = reshape(uh, Nux_in, Nuy_in);
    # v = reshape(vh, Nvx_in, Nvy_in);
    # uwake1 = interp2(xin', yp, u', x_c+0.5, yp);
    # uwake2 = interp2(xin', yp, u', x_c+5, yp);
    # vwake1 = interp2(xp', yin, v', x_c+0.5, yin);
    # vwake2 = interp2(xp', yin, v', x_c+5, yin);

    up, vp, wp, qp
end
