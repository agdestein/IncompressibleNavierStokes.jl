"""
    get_velocity(V, t, options)

Get velocity values at pressure points. Interpolate velocities to pressure positions using BMx and BMy (constructed in operator_divergence.jl).
"""
function get_velocity(V, t, setup)
    # Evaluate boundary conditions at current time
    set_bc_vectors!(setup, t)

    @unpack Au_ux, yAu_ux, Av_vy, yAv_vy, Bup, Bvp = setup.discretization
    @unpack Npx, Npy, indu, indv = setup.grid

    uh = @view V[indu]
    vh = @view V[indv]

    up = reshape(Bup * (Au_ux * uh + yAu_ux), Npx, Npy)
    vp = reshape(Bvp * (Av_vy * vh + yAv_vy), Npx, Npy)

    qp = .√(up .^ 2 .+ vp .^ 2)

    ## get wake profiles
    # u = reshape(uh, Nux_in, Nuy_in);
    # v = reshape(vh, Nvx_in, Nvy_in);
    # uwake1 = interp2(xin', yp, u', x_c+0.5, yp);
    # uwake2 = interp2(xin', yp, u', x_c+5, yp);
    # vwake1 = interp2(xp', yin, v', x_c+0.5, yin);
    # vwake2 = interp2(xp', yin, v', x_c+5, yin);

    up, vp, qp
end
