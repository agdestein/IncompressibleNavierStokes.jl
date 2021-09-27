"""
    S11, S12, S21, S22, S_abs, Jacu, Jacv = strain_tensor(V, t, setup, getJacobian)

Evaluate rate of strain tensor S(u) and its magnitude
"""
function strain_tensor(V, t, setup, getJacobian)
    indu = setup.grid.indu
    indv = setup.grid.indv
    uh = V[indu]
    vh = V[indv]

    Nx = setup.grid.Nx
    Ny = setup.grid.Ny

    Nu = setup.grid.Nu
    Nv = setup.grid.Nv
    Np = setup.grid.Np

    @unpack Nux_in, Nuy_in, Nvx_in, Nvy_in, = setup.grid
    @unpack Su_ux, Su_uy, Su_vx, Sv_vx, Sv_vy, Sv_uy = setup.discretization
    @unpack ySu_ux, ySu_uy, ySu_vx, ySv_vx, ySv_vy, ySv_uy = setup.discretization

    # these four components are approximated by
    S11 = 1 / 2 * 2 * (Su_ux * uh + ySu_ux)
    S12 = 1 / 2 * (Su_uy * uh + ySu_uy + Sv_uy * vh + ySv_uy)
    S21 = 1 / 2 * (Su_vx * uh + ySu_vx + Sv_vx * vh + ySv_vx)
    S22 = 1 / 2 * 2 * (Sv_vy * vh + ySv_vy)

    # Note: S11 and S22 at xp, yp locations (pressure locations)
    # S12, S21 at vorticity locations (corners of pressure cells, (x, y))

    # option 1: get each S11, S12, S21, S22 at 4 locations (ux locations, at uy locations, at vx
    # locations and at vy locations); this gives 16 S fields. determine S_abs at each of these
    # locations, giving 4 S_abs fields, that can be used in computing
    # Dux*(S_abs_ux .* (Su_ux*uh+ySu_ux)) etc.

    # option 2: interpolate S11, S12, S21, S22 to pressure locations
    # determine S_abs at pressure location
    # then interpolate to ux, uy, vx, vy locations

    # we will use option 2;
    # within option 2, we can decide to interpolate S12, S21 etc (option 2a), or we can use
    # directly the operators that map from velocity field to the S locations,
    # as used for example in ke_production (option 2b).

    get_S_abs = false

    if get_S_abs
        # option 2b
        BC = setup.BC
        if BC.u.left == "per" && BC.u.right == "per"
            # "cut-off" the double points in case of periodic BC
            # for periodic boundary conditions S11(Npx+1, :) = S11(1, :)
            # so S11 has size (Npx+1)*Npy; the last row are "ghost" points equal to the
            # first points. we have S11 at positions ([xp[1]-0.5*(hx[1]+hx[end]); xp], yp)
            S11_p = reshape(S11, Nux_in + 1, Nuy_in)
            S11_p = S11_p(2:Nux_in+1, :) # b

            # S12 is defined on the corners: size Nux_in*(Nuy_in+1), positions (xin, y)
            # get S12 and S21 at all corner points
            S12_temp = zeros(Nx + 1, Ny + 1)
            S12_temp[1:Nx, :] = reshape(S12, Nx, Ny + 1)
            S12_temp[Nx+1, :] = S12_temp[1, :]
        elseif BC.u.left == "dir" && BC.u.right == "pres"
            S11_p = reshape(S11, Nux_in + 1, Nuy_in)
            S11_p = S11_p[1:Nux_in, :] # cut off last point

            # S12 is defined on the corners: size Nux_in*(Nuy_in+1), positions (xin, y)
            # get S12 and S21 at all corner points
            S12_temp = zeros(Nx + 1, Ny + 1)
            S12_temp[2:Nx+1, :] = reshape(S12, Nx, Ny + 1)
            S12_temp[1, :] = S12_temp[2, :] # copy from x[2] to x[1]; one could do this more accurately in principle by using the BC
        else
            error("BC not implemented in strain_tensor.m")
        end

        if BC.v.low == "per" && BC.v.up == "per"
            # similarly, S22(:, Npy+1) = S22(:, 1). positions (xp, [yp;yp[1]])
            S22_p = reshape(S22, Nvx_in, Nvy_in + 1)
            S22_p = S22_p(:, 2:Nvy_in+1) # why not 1:Nvy_in?

            # similarly S21 is size (Nux_in+1)*Nuy_in, positions (x, yin)
            S21_temp = zeros(Nx + 1, Ny + 1)
            S21_temp[:, 1:Ny] = reshape(S21, Nx + 1, Ny)
            S21_temp[:, Ny+1] = S21_temp[:, 1]
        elseif strcmp(BC.v.low, "pres") && strcmp(BC.v.up, "pres")
            S22_p = reshape(S22, Nvx_in, Nvy_in + 1)
            S22_p = S22_p(:, 2:Nvy_in)

            # this is nicely defined on all corners
            S21_temp = reshape(S21, Nx + 1, Ny + 1)
        else
            error("BC not implemented in strain_tensor.m")
        end

        # now interpolate S12 and S21 to pressure points
        # S11 and S22 have already been trimmed down to this grid
        @unpack x, y, xp, yp = setup.grid

        S12_p = interp2(y', x, S12_temp, yp', xp) #interp2(x, y', S12_temp, xp, yp');
        S21_p = interp2(y', x, S21_temp, yp', xp) #interp2(x, y', S21_temp, xp, yp');

        # S21 and S12 should be equal!

        # contour(xp, yp, S11");
        # contour(xp, yp, S22");
        # contour(xp, yp, S12");
        # contour(xp, yp, S21");


        ## invariants

        # Verstappen:
        # q = 1/2 * trace(S^2)
        # r = -1/3 * trace(S^3) ( = -det(S) only in 3D); in 2D we should get r = 0

        q = (1 / 2) * (S11_p[:] .^ 2 + S12_p[:] .^ 2 + S21_p[:] .^ 2 + S22_p[:] .^ 2)

        # absolute value of strain tensor
        # with S as defined above, i.e. 0.5*(grad u + grad u^T)
        # S_abs = sqrt(2*tr(S^2)) = sqrt(4*q)
        S_abs = sqrt(4 * q)

        # should be zero:
        # r = (S11[:].^2+S12[:].*S21[:]).*S11[:] + (S11[:].*S12[:]+S12[:].*S22[:]).*S21[:] +         #     (S11[:].*S21[:]+S21[:].*S22[:]).*S12[:] + (S12[:].*S21[:]+S22[:].^2).*S22[:]; #-(S11[:].*S22[:] - S12[:].*S21[:]);

        # figure
        # contour(xp, yp, reshape(q, Npx, Npy)", 25);
        # figure
        # contour(xp, yp, reshape(r, Npx, Npy)", 25);
    else
        # option 2a
        @unpack Cux_k, Cuy_k, Cvx_k, Cvy_k, Auy_k, Avx_k = setup.discretization
        @unpack yCux_k, yCuy_k, yCvx_k, yCvy_k, yAuy_k, yAvx_k = setup.discretization

        S11_p = 1 / 2 * 2 * (Cux_k * uh + yCux_k)
        S12_p =
            1 / 2 * (
                Cuy_k * (Auy_k * uh + yAuy_k) +
                yCuy_k +
                Cvx_k * (Avx_k * vh + yAvx_k) +
                yCvx_k
            )
        S21_p = S12_p
        S22_p = 1 / 2 * 2 * (Cvy_k * vh + yCvy_k)

        S_abs = sqrt(2 * S11_p .^ 2 + 2 * S22_p .^ 2 + 2 * S12_p .^ 2 + 2 * S21_p .^ 2)

        # Jacobian of S_abs wrt u and v
        if getJacobian
            eps = 1e-14
            Sabs_inv = spdiagm(1 ./ (2 * S_abs + eps))
            Jacu =
                Sabs_inv * (4 * spdiagm(S11_p) * Cux_k + 4 * spdiagm(S12_p) * Cuy_k * Auy_k)
            Jacv =
                Sabs_inv * (4 * spdiagm(S12_p) * Cvx_k * Avx_k + 4 * spdiagm(S22_p) * Cvy_k)
        else
            Jacu = sparse(Np, Nu)
            Jacv = sparse(Np, Nv)
        end
    end
    S11, S12, S21, S22, S_abs, Jacu, Jacv
end
