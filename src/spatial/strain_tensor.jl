"""
    S11, S12, S21, S22, S_abs, Jacu, Jacv = strain_tensor(V, t, setup, getJacobian)

Evaluate rate of strain tensor S(u) and its magnitude
"""
function strain_tensor(V, t, setup, getJacobian)
    @unpack Nx, Ny, Nu, Nv, Np, indu, indv = setup.grid
    @unpack Nux_in, Nuy_in, Nvx_in, Nvy_in, = setup.grid
    @unpack x, y, xp, yp = setup.grid
    @unpack Su_ux, Su_uy, Su_vx, Sv_vx, Sv_vy, Sv_uy = setup.discretization
    @unpack ySu_ux, ySu_uy, ySu_vx, ySv_vx, ySv_vy, ySv_uy = setup.discretization
    @unpack Cux_k, Cuy_k, Cvx_k, Cvy_k, Auy_k, Avx_k = setup.discretization
    @unpack yCux_k, yCuy_k, yCvx_k, yCvy_k, yAuy_k, yAvx_k = setup.discretization

    uₕ = @view V[indu]
    vₕ = @view V[indv]

    # these four components are approximated by
    S11 = 1 / 2 * 2 * (Su_ux * uₕ + ySu_ux)
    S12 = 1 / 2 * (Su_uy * uₕ + ySu_uy + Sv_uy * vₕ + ySv_uy)
    S21 = 1 / 2 * (Su_vx * uₕ + ySu_vx + Sv_vx * vₕ + ySv_vx)
    S22 = 1 / 2 * 2 * (Sv_vy * vₕ + ySv_vy)

    # Note: S11 and S22 at xp, yp locations (pressure locations)
    # S12, S21 at vorticity locations (corners of pressure cells, (x, y))

    # option 1: get each S11, S12, S21, S22 at 4 locations (ux locations, at uy locations, at vx
    # locations and at vy locations); this gives 16 S fields. determine S_abs at each of these
    # locations, giving 4 S_abs fields, that can be used in computing
    # Dux*(S_abs_ux .* (Su_ux*uₕ+ySu_ux)) etc.

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
        bc = setup.bc
        if bc.u.left == "per" && bc.u.right == "per"
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
        elseif bc.u.left == "dir" && bc.u.right == "pres"
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

        if bc.v.low == "per" && bc.v.up == "per"
            # similarly, S22(:, Npy+1) = S22(:, 1). positions (xp, [yp;yp[1]])
            S22_p = reshape(S22, Nvx_in, Nvy_in + 1)
            S22_p = S22_p(:, 2:Nvy_in+1) # why not 1:Nvy_in?

            # similarly S21 is size (Nux_in+1)*Nuy_in, positions (x, yin)
            S21_temp = zeros(Nx + 1, Ny + 1)
            S21_temp[:, 1:Ny] = reshape(S21, Nx + 1, Ny)
            S21_temp[:, Ny+1] = S21_temp[:, 1]
        elseif strcmp(bc.v.low, "pres") && strcmp(bc.v.up, "pres")
            S22_p = reshape(S22, Nvx_in, Nvy_in + 1)
            S22_p = S22_p(:, 2:Nvy_in)

            # this is nicely defined on all corners
            S21_temp = reshape(S21, Nx + 1, Ny + 1)
        else
            error("BC not implemented in strain_tensor.m")
        end

        # now interpolate S12 and S21 to pressure points
        # S11 and S22 have already been trimmed down to this grid

        # S21 and S12 should be equal!
        S12_p = interp2(y', x, S12_temp, yp', xp)
        S21_p = interp2(y', x, S21_temp, yp', xp)

        ## invariants
        q = 1 / 2 * (S11_p[:] .^ 2 + S12_p[:] .^ 2 + S21_p[:] .^ 2 + S22_p[:] .^ 2)

        # absolute value of strain tensor
        # with S as defined above, i.e. 0.5*(grad u + grad u^T)
        # S_abs = sqrt(2*tr(S^2)) = sqrt(4*q)
        S_abs = sqrt(4q)
    else
        # option 2a
        S11_p = 1 / 2 * 2 * (Cux_k * uₕ + yCux_k)
        S12_p =
            1 / 2 * (
                Cuy_k * (Auy_k * uₕ + yAuy_k) +
                yCuy_k +
                Cvx_k * (Avx_k * vₕ + yAvx_k) +
                yCvx_k
            )
        S21_p = S12_p
        S22_p = 1 / 2 * 2 * (Cvy_k * vₕ + yCvy_k)

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
