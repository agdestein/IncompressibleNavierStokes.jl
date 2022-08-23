"""
    ke_viscosity(grid, boundary_conditions)

Average (turbulent) viscosity to cell faces.
"""
function ke_viscosity end

# 2D version
function ke_viscosity(grid::Grid{T,2}, boundary_conditions) where {T}
    (; x, y, xp, yp, hx, hy) = grid
    (; Nx, Ny, Npx, Npy, Nux_in, Nuy_in, Nvx_in, Nvy_in, Bvux, Buvy, Bkux, Bkvy) = grid
    (; k_bc, e_bc) = boundary_conditions

    kLe = fill(k_bc.x[1], Ny + 1)
    kRi = fill(k_bc.x[2], Ny + 1)
    kLo = fill(k_bc.y[1], Nx + 1)
    kUp = fill(k_bc.y[2], Nx + 1)

    eLe = fill(e_bc.x[1], Ny + 1)
    eRi = fill(e_bc.x[2], Ny + 1)
    eLo = fill(e_bc.y[1], Nx + 1)
    eUp = fill(e_bc.y[2], Nx + 1)

    kLe = LinearInterpolation(y, kLe)(yp)
    kRi = LinearInterpolation(y, kRi)(yp)
    kLo = LinearInterpolation(x, kLo)(xp)
    kUp = LinearInterpolation(x, kUp)(xp)

    eLe = LinearInterpolation(y, eLe)(yp)
    eRi = LinearInterpolation(y, eRi)(yp)
    eLo = LinearInterpolation(x, eLo)(xp)
    eUp = LinearInterpolation(x, eUp)(xp)

    # Averaging weight:
    weight = 1 / 2

    ## Nu to ux positions

    A1D = sparse(I, Npx + 2, Npx + 2)

    ## K
    # Boundary conditions for k; mapping from Npx (k) points to Npx+2 points
    B1D, Btemp, ybcl, ybcr = bc_general_stag(
        Npx + 2,
        Npx,
        2,
        boundary_conditions.k.x[1],
        boundary_conditions.k.x[2],
        hx[1],
        hx[end],
    )
    # Then map back to Nux_in+1 points (ux-faces) with Bkux

    # Extend to 2D
    Ak_ux = kron(sparse(I, Nuy_in, Nuy_in), Bkux * A1D * B1D)
    ybc = kron(kLe, ybcl) + kron(kRi, ybcr)
    yAk_ux = kron(sparse(I, Nuy_in, Nuy_in), Bkux * A1D * Btemp) * ybc

    ## Epsilon
    # In a similar way but with different boundary conditions
    B1D, Btemp, ybcl, ybcr = bc_general_stag(
        Npx + 2,
        Npx,
        2,
        boundary_conditions.e.x[1],
        boundary_conditions.e.x[2],
        hx[1],
        hx[end],
    )
    # Extend to 2D
    Ae_ux = kron(sparse(I, Nuy_in, Nuy_in), Bkux * A1D * B1D)
    ybc = kron(eLe, ybcl) + kron(eRi, ybcr)
    yAe_ux = kron(sparse(I, Nuy_in, Nuy_in), Bkux * A1D * Btemp) * ybc


    # So nu at ux is given by:
    # Cmu * (Ak_ux * k + yAk_ux).^2 / (Ae_ux * e + yAe_ux)


    ## Nu to uy positions
    diag1 = weight * ones(Npx + 1)
    A1D = spdiagm(Npx + 1, Npx + 2, 0 => diag1, 1 => diag1)

    ## K
    # Boundary conditions for k; mapping from Npx (k) points to Npx+2 points
    B1D, Btemp, ybcl, ybcr = bc_general_stag(
        Npx + 2,
        Npx,
        2,
        boundary_conditions.k.x[1],
        boundary_conditions.k.x[2],
        hx[1],
        hx[end],
    )

    # Then map to Nux_in points (like Iv_uy) with Bvux

    # Extend to 2D
    A2D = kron(sparse(I, Npy + 2, Npy + 2), Bvux * A1D * B1D)
    kLe_i = [kLe[1]; kLe; kLe[end]]
    kRi_i = [kRi[1]; kRi; kRi[end]]
    ybc = kron(kLe_i, ybcl) + kron(kRi_i, ybcr)
    yAk_uy_lr = kron(sparse(I, Npy + 2, Npy + 2), Bvux * A1D * Btemp) * ybc

    # Apply BC in y-direction
    B2D, Btemp, ybcl, ybcu = bc_general_stag(
        Npy + 2,
        Npy,
        2,
        boundary_conditions.k.y[1],
        boundary_conditions.k.y[2],
        hy[1],
        hy[end],
    )
    ybc = kron(Btemp * ybcl, kLo) + kron(Btemp * ybcu, kUp)
    yAk_uy_lu = A2D * ybc

    A2Dx = A2D * kron(B2D, sparse(I, Npx, Npx))

    # Calculate average in y-direction; no boundary conditions
    diag1 = weight * ones(Npy + 1)
    A1Dy = spdiagm(Npy + 1, Npy + 2, 0 => diag1, 1 => diag1)
    A2Dy = kron(A1Dy, sparse(I, Nux_in, Nux_in))

    Ak_uy = A2Dy * A2Dx
    yAk_uy = A2Dy * (yAk_uy_lu + yAk_uy_lr)


    ## Epsilon
    # In a similar way but with different boundary conditions
    B1D, Btemp, ybcl, ybcr = bc_general_stag(
        Npx + 2,
        Npx,
        2,
        boundary_conditions.e.x[1],
        boundary_conditions.e.x[2],
        hx[1],
        hx[end],
    )

    # Then map to Nux_in points (like Iv_uy) with Bvux

    # Extend to 2D
    A2D = kron(sparse(I, Npy + 2, Npy + 2), Bvux * A1D * B1D)
    eLe_i = [eLe[1]; eLe; eLe[end]]
    eRi_i = [eRi[1]; eRi; eRi[end]]
    ybc = kron(eLe_i, ybcl) + kron(eRi_i, ybcr)
    yAe_uy_lr = kron(sparse(I, Npy + 2, Npy + 2), Bvux * A1D * Btemp) * ybc

    # Apply BC in y-direction
    B2D, Btemp, ybcl, ybcu = bc_general_stag(
        Npy + 2,
        Npy,
        2,
        boundary_conditions.e.y[1],
        boundary_conditions.e.y[2],
        hy[1],
        hy[end],
    )
    ybc = kron(Btemp * ybcl, eLo) + kron(Btemp * ybcu, eUp)
    yAe_uy_lu = A2D * ybc

    A2Dx = A2D * kron(B2D, sparse(I, Npx, Npx))

    # Calculate average in y-direction; no boundary conditions
    diag1 = weight * ones(Npy + 1)
    A1Dy = spdiagm(Npy + 1, Npy + 2, 0 => diag1, 1 => diag1)
    A2Dy = kron(A1Dy, sparse(I, Nux_in, Nux_in))

    Ae_uy = A2Dy * A2Dx
    yAe_uy = A2Dy * (yAe_uy_lu + yAe_uy_lr)

    # So nu at uy is given by:
    # Cmu * (Ak_uy * k + yAk_uy).^2 / (Ae_uy * e + yAe_uy)


    ## Nu to vx positions
    diag1 = weight * ones(Npy + 1)
    A1D = spdiagm(Npy + 1, Npy + 2, 0 => diag1, 1 => diag1)

    ## K
    # Boundary conditions for k; mapping from Npy (k) points to Npy+2 points
    B1D, Btemp, ybcl, ybcu = bc_general_stag(
        Npy + 2,
        Npy,
        2,
        boundary_conditions.k.y[1],
        boundary_conditions.k.y[2],
        hy[1],
        hy[end],
    )

    # Map to Nvy_in points (like Iu_vx) with Buvy

    # Extend to 2D
    A2D = kron(Buvy * A1D * B1D, sparse(I, Npx + 2, Npx + 2))
    kLo_i = [kLo[1]; kLo; kLo[end]]
    kUp_i = [kUp[1]; kUp; kUp[end]]
    ybc = kron(ybcl, kLo_i) + kron(ybcu, kUp_i)
    yAk_vx_lu = kron(Buvy * A1D * Btemp, sparse(I, Npx + 2, Npx + 2)) * ybc

    # Apply boundary conditions also in x-direction:
    B2D, Btemp, ybcl, ybcr = bc_general_stag(
        Npx + 2,
        Npx,
        2,
        boundary_conditions.k.x[1],
        boundary_conditions.k.x[2],
        hx[1],
        hx[end],
    )
    ybc = kron(kLe, Btemp * ybcl) + kron(kRi, Btemp * ybcr)
    yAk_vx_lr = A2D * ybc

    A2Dy = A2D * kron(sparse(I, Npy, Npy), B2D)

    # Calculate average in x-direction; no boundary conditions
    diag1 = weight * ones(Npx + 1)
    A1Dx = spdiagm(Npx + 1, Npx + 2, 0 => diag1, 1 => diag1)
    A2Dx = kron(sparse(I, Nvy_in, Nvy_in), A1Dx)

    Ak_vx = A2Dx * A2Dy
    yAk_vx = A2Dx * (yAk_vx_lr + yAk_vx_lu)


    ## Epsilon
    # In a similar way but with different boundary conditions
    B1D, Btemp, ybcl, ybcu = bc_general_stag(
        Npy + 2,
        Npy,
        2,
        boundary_conditions.e.y[1],
        boundary_conditions.e.y[2],
        hy[1],
        hy[end],
    )

    # Extend to 2D
    A2D = kron(Buvy * A1D * B1D, sparse(I, Npx + 2, Npx + 2))
    eLo_i = [eLo[1]; eLo; eLo[end]]
    eUp_i = [eUp[1]; eUp; eUp[end]]
    ybc = kron(ybcl, eLo_i) + kron(ybcu, eUp_i)
    yAe_vx_lu = kron(Buvy * A1D * Btemp, sparse(I, Npx + 2, Npx + 2)) * ybc

    # Apply boundary conditions also in x-direction:
    B2D, Btemp, ybcl, ybcr = bc_general_stag(
        Npx + 2,
        Npx,
        2,
        boundary_conditions.e.x[1],
        boundary_conditions.e.x[2],
        hx[1],
        hx[end],
    )
    ybc = kron(eLe, Btemp * ybcl) + kron(eRi, Btemp * ybcr)
    yAe_vx_lr = A2D * ybc

    A2Dy = A2D * kron(sparse(I, Npy, Npy), B2D)

    # Calculate average in x-direction; no boundary conditions
    diag1 = weight * ones(Npx + 1)
    A1Dx = spdiagm(Npx + 1, Npx + 2, 0 => diag1, 1 => diag1)
    A2Dx = kron(sparse(I, Nvy_in, Nvy_in), A1Dx)

    Ae_vx = A2Dx * A2Dy
    yAe_vx = A2Dx * (yAe_vx_lr + yAe_vx_lu)


    # So nu at vx is given by:
    # Cmu * (Ak_vx * k + yAk_vx).^2 / (Ae_vx * e + yAe_vx)


    ## Nu to vy positions
    A1D = sparse(I, Npy + 2, Npy + 2)

    ## K
    # Boundary conditions; mapping from Npy (k) points to Npy+2 (vy faces) points
    B1D, Btemp, ybcl, ybcu = bc_general_stag(
        Npy + 2,
        Npy,
        2,
        boundary_conditions.k.y[1],
        boundary_conditions.k.y[2],
        hy[1],
        hy[end],
    )

    # Then map back to Nvy_in+1 points (vy-faces) with Bkvy

    # Extend to 2D
    Ak_vy = kron(Bkvy * A1D * B1D, sparse(I, Nvx_in, Nvx_in))
    ybc = kron(ybcl, kLo) + kron(ybcu, kUp)
    yAk_vy = kron(Bkvy * A1D * Btemp, sparse(I, Nvx_in, Nvx_in)) * ybc


    ## Epsilon
    # In a similar way but with different boundary conditions
    B1D, Btemp, ybcl, ybcu = bc_general_stag(
        Npy + 2,
        Npy,
        2,
        boundary_conditions.e.y[1],
        boundary_conditions.e.y[2],
        hy[1],
        hy[end],
    )
    # Extend to 2D
    Ae_vy = kron(Bkvy * A1D * B1D, sparse(I, Nvx_in, Nvx_in))
    ybc = kron(ybcl, eLo) + kron(ybcu, eUp)
    yAe_vy = kron(Bkvy * A1D * Btemp, sparse(I, Nvx_in, Nvx_in)) * ybc


    # So nu at vy is given by:
    # Cmu * (Ak_vy * k + yAk_vy).^2 / (Ae_vy * e + yAe_vy)

    # TODO: Return correct operators
    (;
        Ak_ux,
        Ak_uy,
        Ak_vx,
        Ak_vy,
        yAk_ux,
        yAk_uy,
        yAk_vx,
        yAk_vy,
        Ae_ux,
        Ae_uy,
        Ae_vx,
        Ae_vy,
        yAe_ux,
        yAe_uy,
        yAe_vx,
        yAe_vy,
    )
end

# 3D version
function ke_viscosity(grid::Grid{T,3}, boundary_conditions) where {T}
    error("Not implemented")
end
