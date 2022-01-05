"""
    operator_turbulent_diffusion!(setup)

Average (turbulent) viscosity to cell faces: from `ν` at `xp`, `yp` to `ν` at `ux`, `uy`,
`vx`, `vy` locations.

See also `ke_viscosity.jl`.
"""
function operator_turbulent_diffusion!(setup)
    (; bc) = setup
    (; Npx, Npy) = setup.grid
    (; Nux_in, Nuy_in, Nvx_in, Nvy_in) = setup.grid
    (; hx, hy, gxd, gyd) = setup.grid
    (; Buvy, Bvux, Bkux, Bkvy) = setup.grid

    # FIXME: 3D implementation

    # Averaging weight:
    weight = 1 / 2

    ## Nu to ux positions

    A1D = sparse(I, Npx + 2, Npx + 2)
    A1D = Bkux * A1D

    # Boundary conditions for ν; mapping from Npx (k) points to Npx+2 points
    Aν_ux_bc = bc_general_stag(Npx + 2, Npx, 2, bc.ν.x[1], bc.ν.x[2], hx[1], hx[end])
    # Then map back to Nux_in+1 points (ux-faces) with Bkux

    # Extend to 2D
    Aν_ux = kron(sparse(I, Nuy_in, Nuy_in), A1D * Aν_ux_bc.B1D)
    Aν_ux_bc = (;Aν_ux_bc..., Bbc = kron(sparse(I, Nuy_in, Nuy_in), A1D * Aν_ux_bc.Btemp))


    ## Nu to uy positions

    # Average in x-direction
    diag1 = weight * ones(Npx + 1)
    A1D = spdiagm(Npx + 1, Npx + 2, 0 => diag1, 1 => diag1)
    # Then map to Nux_in points (like Iv_uy) with Bvux
    A1D = Bvux * A1D

    # Calculate average in y-direction; no boundary conditions
    diag1 = weight * ones(Npy + 1)
    A1Dy = spdiagm(Npy + 1, Npy + 2, 0 => diag1, 1 => diag1)
    A2Dy = kron(A1Dy, sparse(I, Nux_in, Nux_in))

    # Boundary conditions for ν in x-direction;
    # Mapping from Npx (ν) points to Npx+2 points
    Aν_uy_bc_lr = bc_general_stag(Npx + 2, Npx, 2, bc.ν.x[1], bc.ν.x[2], hx[1], hx[end])

    # Extend BC to 2D
    A2D = kron(sparse(I, Npy + 2, Npy + 2), A1D * Aν_uy_bc_lr.B1D)

    # Apply bc in y-direction
    Aν_uy_bc_lu = bc_general_stag(Npy + 2, Npy, 2, bc.ν.y[1], bc.ν.y[2], hy[1], hy[end])

    A2Dx = A2D * kron(Aν_uy_bc_lu.B1D, sparse(I, Npx, Npx))

    Aν_uy = A2Dy * A2Dx

    Aν_uy_bc_lr =(;Aν_uy_bc_lr..., B2D = A2Dy * kron(sparse(I, Npy + 2, Npy + 2), A1D * Aν_uy_bc_lr.Btemp))
    Aν_uy_bc_lu = (; Aν_uy_bc_lu..., B2D = A2Dy * A2D * kron(Aν_uy_bc_lu.Btemp, sparse(I, Npx, Npx)))

    ## Nu to vx positions
    diag1 = weight * ones(Npy + 1)
    A1D = spdiagm(Npy + 1, Npy + 2, 0 => diag1, 1 => diag1)

    # Map to Nvy_in points (like Iu_vx) with Buvy
    A1D = Buvy * A1D

    # Calculate average in x-direction; no boundary conditions
    diag1 = weight * ones(Npx + 1)
    A1Dx = spdiagm(Npx + 1, Npx + 2, 0 => diag1, 1 => diag1)
    A2Dx = kron(sparse(I, Nvy_in, Nvy_in), A1Dx)


    # Boundary conditions for ν in y-direction;
    # Mapping from Npy (ν) points to Npy+2 points
    Aν_vx_bc_lu = bc_general_stag(Npy + 2, Npy, 2, bc.ν.y[1], bc.ν.y[2], hy[1], hy[end])

    # Extend BC to 2D
    A2D = kron(A1D * Aν_vx_bc_lu.B1D, sparse(I, Npx + 2, Npx + 2))


    # Apply boundary conditions also in x-direction:
    Aν_vx_bc_lr = bc_general_stag(Npx + 2, Npx, 2, bc.ν.x[1], bc.ν.x[2], hx[1], hx[end])

    A2Dy = A2D * kron(sparse(I, Npy, Npy), Aν_vx_bc_lr.B1D)

    Aν_vx = A2Dx * A2Dy

    Aν_vx_bc_lu = (;Aν_vx_bc_lu..., B2D = A2Dx * kron(A1D * Aν_vx_bc_lu.Btemp, sparse(I, Npx + 2, Npx + 2)))
    Aν_vx_bc_lr = (;Aν_vx_bc_lr..., B2D = A2Dx * A2D * kron(sparse(I, Npy, Npy), Aν_vx_bc_lr.Btemp))


    ## Nu to vy positions
    A1D = sparse(I, Npy + 2, Npy + 2)
    # Then map back to Nvy_in+1 points (vy-faces) with Bkvy
    A1D = Bkvy * A1D

    # Boundary conditions for ν; mapping from Npy (ν) points to Npy+2 (vy faces) points
    Aν_vy_bc = bc_general_stag(Npy + 2, Npy, 2, bc.ν.y[1], bc.ν.y[2], hy[1], hy[end])

    # Extend to 2D
    Aν_vy = kron(A1D * Aν_vy_bc.B1D, sparse(I, Nvx_in, Nvx_in))
    Aν_vy_bc = (; Aν_vy_bc..., Bbc = kron(A1D * Aν_vy_bc.Btemp, sparse(I, Nvx_in, Nvx_in)))


    # So ν at vy is given by:
    # (Aν_vy * k + yAν_vy)

    ## Store in struct
    @pack! setup.discretization = Aν_ux, Aν_ux_bc
    @pack! setup.discretization = Aν_uy, Aν_uy_bc_lr, Aν_uy_bc_lu
    @pack! setup.discretization = Aν_vx, Aν_vx_bc_lr, Aν_vx_bc_lu
    @pack! setup.discretization = Aν_vy, Aν_vy_bc

    ## Get derivatives u_x, u_y, v_x and v_y at cell centers
    # Differencing velocity to ν-points

    ## Du/dx

    # Differencing matrix
    diag1 = 1 ./ hx
    C1D = spdiagm(Npx, Npx + 1, 0 => -diag1, 1 => diag1)

    # Boundary conditions
    Cux_k_bc =
        bc_general(Npx + 1, Nux_in, Npx + 1 - Nux_in, bc.u.x[1], bc.u.x[2], hx[1], hx[end])

    Cux_k = kron(sparse(I, Npy, Npy), C1D * Cux_k_bc.B1D)
    Cux_k_bc = (; Cux_k_bc..., Bbc = kron(sparse(I, Npy, Npy), C1D * Cux_k_bc.Btemp))

    # Cux_k*uₕ+yCux_k;


    ## Du/dy

    # Average to k-positions (in x-dir)
    weight = 1 / 2
    diag1 = weight * ones(Npx)
    A1D = spdiagm(Npx, Npx + 1, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Auy_k_bc =
        bc_general(Npx + 1, Nux_in, Npx + 1 - Nux_in, bc.u.x[1], bc.u.x[2], hx[1], hx[end])

    Auy_k = kron(sparse(I, Npy, Npy), A1D * Auy_k_bc.B1D)
    Auy_k_bc = (; Auy_k_bc..., Bbc = kron(sparse(I, Npy, Npy), A1D * Auy_k_bc.Btemp))

    # Take differences
    gydnew = gyd[1:end-1] + gyd[2:end] # Differencing over 2*Δy
    diag2 = 1 ./ gydnew
    C1D = spdiagm(Npy, Npy + 2, 0 => -diag2, 2 => diag2)

    Cuy_k_bc = bc_general_stag(Npy + 2, Npy, 2, bc.u.y[1], bc.u.y[2], hy[1], hy[end])

    Cuy_k = kron(C1D * Cuy_k_bc.B1D, sparse(I, Npx, Npx))
    Cuy_k_bc = (; Cuy_k_bc..., Bbc = kron(C1D * Cuy_k_bc.Btemp, sparse(I, Npx, Npx)))

    ## Dv/dx

    # Average to k-positions (in y-dir)
    weight = 1 / 2
    diag1 = weight * ones(Npy)
    A1D = spdiagm(Npy, Npy + 1, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Avx_k_bc =
        bc_general(Npy + 1, Nvy_in, Npy + 1 - Nvy_in, bc.v.y[1], bc.v.y[2], hy[1], hy[end])
    Avx_k = kron(A1D * Avx_k_bc.B1D, sparse(I, Npx, Npx))
    Avx_k_bc = (; Avx_k_bc..., Bbc = kron(A1D * Avx_k_bc.Btemp, sparse(I, Npx, Npx)))

    # Take differences
    gxdnew = gxd[1:end-1] + gxd[2:end] # Differencing over 2*Δx
    diag2 = 1 ./ gxdnew
    C1D = spdiagm(Npx, Npx + 2, 0 => -diag2, 2 => diag2)

    Cvx_k_bc =
        bc_general_stag(Npx + 2, Npx, Npx + 2 - Npx, bc.v.x[1], bc.v.x[2], hx[1], hx[end])

    Cvx_k = kron(sparse(I, Npy, Npy), C1D * Cvx_k_bc.B1D)
    Cvx_k_bc = (; Cvx_k_bc..., Bbc = kron(sparse(I, Npy, Npy), C1D * Cvx_k_bc.Btemp))

    ## Dv/dy

    # Differencing matrix
    diag1 = 1 ./ hy
    C1D = spdiagm(Npy, Npy + 1, 0 => -diag1, 1 => diag1)

    # Boundary conditions
    Cvy_k_bc =
        bc_general(Npy + 1, Nvy_in, Npy + 1 - Nvy_in, bc.v.y[1], bc.v.y[2], hy[1], hy[end])

    Cvy_k = kron(C1D * Cvy_k_bc.B1D, sparse(I, Npx, Npx))
    Cvy_k_bc = (;Cvy_k_bc..., Bbc = kron(C1D * Cvy_k_bc.Btemp, sparse(I, Npx, Npx)))


    ## Store in struct
    @pack! setup.discretization =
        Cux_k, Cux_k_bc, Cuy_k, Cuy_k_bc, Cvx_k, Cvx_k_bc, Cvy_k, Cvy_k_bc
    @pack! setup.discretization = Auy_k, Auy_k_bc, Avx_k, Avx_k_bc

    setup
end
