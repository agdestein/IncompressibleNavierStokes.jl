"""
    strain_tensor(V, setup; getJacobian = false, get_S_abs = false)

Evaluate rate of strain tensor `S(V)` and its magnitude.
"""
function strain_tensor end

# 2D version
function strain_tensor(V, setup::Setup{T,2}; getJacobian = false, get_S_abs = false) where {T}
    (; Nx, Ny, Nu, Nv, Np, indu, indv) = setup.grid
    (; Nux_in, Nuy_in, Nvx_in, Nvy_in) = setup.grid
    (; x, y, xp, yp) = setup.grid
    (; Su_ux, Su_uy, Su_vx, Sv_vx, Sv_vy, Sv_uy) = setup.operators
    (; Cux_k, Cuy_k, Cvx_k, Cvy_k, Auy_k, Avx_k) = setup.operators

    uₕ = @view V[indu]
    vₕ = @view V[indv]

    # These four components are approximated by
    S11 = Su_ux * uₕ
    S12 = 1 / 2 * (Su_uy * uₕ + Sv_uy * vₕ)
    S21 = 1 / 2 * (Su_vx * uₕ + Sv_vx * vₕ)
    S22 = Sv_vy * vₕ

    # Note: S11 and S22 at xp, yp locations (pressure locations)
    # S12, S21 at vorticity locations (corners of pressure cells, (x, y))

    # Option 1: get each S11, S12, S21, S22 at 4 locations (ux locations, at uy locations, at vx
    # Locations and at vy locations); this gives 16 S fields. determine S_abs at each of these
    # Locations, giving 4 S_abs fields, that can be used in computing
    # Dux*(S_abs_ux .* (Su_ux*uₕ)) etc.

    # Option 2: interpolate S11, S12, S21, S22 to pressure locations
    # Determine S_abs at pressure location
    # Then interpolate to ux, uy, vx, vy locations

    # We will use option 2;
    # Within option 2, we can decide to interpolate S12, S21 etc (option 2a), or we can use
    # Directly the operators that map from velocity field to the S locations,
    # As used for example in ke_production (option 2b).

    if get_S_abs
        # Option 2b
            # "cut-off" the double points in case of periodic BC
            # For periodic boundary conditions S11(Npx+1, :) = S11(1, :)
            # So S11 has size (Npx+1)*Npy; the last row are "ghost" points equal to the
            # First points. we have S11 at positions ([xp[1] - 1/2*(hx[1]+hx[end]); xp], yp)
            S11_p = reshape(S11, Nux_in + 1, Nuy_in)
            S11_p = S11_p(2:(Nux_in + 1), :) # B

            # S12 is defined on the corners: size Nux_in*(Nuy_in+1), positions (xin, y)
            # Get S12 and S21 at all corner points
            S12_temp = zeros(Nx + 1, Ny + 1)
            S12_temp[1:Nx, :] = reshape(S12, Nx, Ny + 1)
            S12_temp[Nx + 1, :] = S12_temp[1, :]

            # Similarly, S22(:, Npy+1) = S22(:, 1). positions (xp, [yp;yp[1]])
            S22_p = reshape(S22, Nvx_in, Nvy_in + 1)
            S22_p = S22_p(:, 2:(Nvy_in + 1)) # Why not 1:Nvy_in?

            # Similarly S21 is size (Nux_in+1)*Nuy_in, positions (x, yin)
            S21_temp = zeros(Nx + 1, Ny + 1)
            S21_temp[:, 1:Ny] = reshape(S21, Nx + 1, Ny)
            S21_temp[:, Ny + 1] = S21_temp[:, 1]

        # Now interpolate S12 and S21 to pressure points
        # S11 and S22 have already been trimmed down to this grid

        # S21 and S12 should be equal!
        S12_p = interp2(y', x, S12_temp, yp', xp)
        S21_p = interp2(y', x, S21_temp, yp', xp)

        ## Invariants
        q = @. 1 / 2 * (S11_p[:]^2 + S12_p[:]^2 + S21_p[:]^2 + S22_p[:]^2)

        # Absolute value of strain tensor
        # With S as defined above, i.e. 1/2*(grad u + grad u^T)
        # S_abs = sqrt(2*tr(S^2)) = sqrt(4*q)
        S_abs = sqrt(4q)
    else
        # Option 2a
        S11_p = Cux_k * uₕ
        S12_p =
            1 / 2 * (
                Cuy_k * (Auy_k * uₕ) +
                Cvx_k * (Avx_k * vₕ)
            )
        S21_p = S12_p
        S22_p = Cvy_k * vₕ

        S_abs = @. sqrt(2 * S11_p^2 + 2 * S22_p^2 + 2 * S12_p^2 + 2 * S21_p^2)

        # Jacobian of S_abs wrt u and v
        if getJacobian
            eps = 1e-14
            Sabs_inv = spdiagm(1 ./ (2 .* S_abs .+ eps))
            Jacu =
                Sabs_inv * (4 * spdiagm(S11_p) * Cux_k + 4 * spdiagm(S12_p) * Cuy_k * Auy_k)
            Jacv =
                Sabs_inv * (4 * spdiagm(S12_p) * Cvx_k * Avx_k + 4 * spdiagm(S22_p) * Cvy_k)
        else
            Jacu = spzeros(Np, Nu)
            Jacv = spzeros(Np, Nv)
        end
    end

    S11, S12, S21, S22, S_abs, Jacu, Jacv
end

# 3D version
function strain_tensor(V, setup::Setup{T,3}; getJacobian = false, get_S_abs = false) where {T}
    error("Not implemented (3D)")
end
