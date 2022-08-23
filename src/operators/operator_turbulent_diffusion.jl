"""
    operator_turbulent_diffusion(grid, boundary_conditions)

Average (turbulent) viscosity to cell faces: from `ν` at `xp`, `yp` to `ν` at `ux`, `uy`,
`vx`, `vy` locations.

See also `ke_viscosity.jl`.
"""
function operator_turbulent_diffusion end

# 2D version
function operator_turbulent_diffusion(grid::Grid{T,2}, boundary_conditions) where {T}
    (; Npx, Npy) = grid
    (; Nux_in, Nuy_in, Nvx_in, Nvy_in) = grid
    (; hx, hy, gxd, gyd) = grid
    (; Buvy, Bvux, Bkux, Bkvy) = grid


    # Averaging weight:
    weight = 1 / 2

    ## Nu to ux positions

    A1D = I(Npx + 2)
    A1D = Bkux * A1D

    # Boundary conditions for ν; mapping from Npx (k) points to Npx+2 points
    Aν_ux_bc = bc_general_stag(
        Npx + 2,
        Npx,
        2,
        boundary_conditions.ν.x[1],
        boundary_conditions.ν.x[2],
        hx[1],
        hx[end],
    )
    # Then map back to Nux_in+1 points (ux-faces) with Bkux

    # Extend to 2D
    Aν_ux = I(Nuy_in) ⊗ (A1D * Aν_ux_bc.B1D)
    Aν_ux_bc = (; Aν_ux_bc..., Bbc = I(Nuy_in) ⊗ (A1D * Aν_ux_bc.Btemp))


    ## Nu to uy positions

    # Average in x-direction
    diag1 = weight * ones(Npx + 1)
    A1D = spdiagm(Npx + 1, Npx + 2, 0 => diag1, 1 => diag1)
    # Then map to Nux_in points (like Iv_uy) with Bvux
    A1D = Bvux * A1D

    # Calculate average in y-direction; no boundary conditions
    diag1 = weight * ones(Npy + 1)
    A1Dy = spdiagm(Npy + 1, Npy + 2, 0 => diag1, 1 => diag1)
    A2Dy = A1Dy ⊗ I(Nux_in)

    # Boundary conditions for ν in x-direction;
    # Mapping from Npx (ν) points to Npx+2 points
    Aν_uy_bc_lr = bc_general_stag(
        Npx + 2,
        Npx,
        2,
        boundary_conditions.ν.x[1],
        boundary_conditions.ν.x[2],
        hx[1],
        hx[end],
    )

    # Extend BC to 2D
    A2D = I(Npy + 2) ⊗ (A1D * Aν_uy_bc_lr.B1D)

    # Apply BC in y-direction
    Aν_uy_bc_lu = bc_general_stag(
        Npy + 2,
        Npy,
        2,
        boundary_conditions.ν.y[1],
        boundary_conditions.ν.y[2],
        hy[1],
        hy[end],
    )

    A2Dx = A2D * (Aν_uy_bc_lu.B1D ⊗ I(Npx))

    Aν_uy = A2Dy * A2Dx

    Aν_uy_bc_lr = (; Aν_uy_bc_lr..., B2D = A2Dy * (I(Npy + 2) ⊗ (A1D * Aν_uy_bc_lr.Btemp)))
    Aν_uy_bc_lu = (; Aν_uy_bc_lu..., B2D = A2Dy * A2D * (Aν_uy_bc_lu.Btemp ⊗ I(Npx)))

    ## Nu to vx positions
    diag1 = weight * ones(Npy + 1)
    A1D = spdiagm(Npy + 1, Npy + 2, 0 => diag1, 1 => diag1)

    # Map to Nvy_in points (like Iu_vx) with Buvy
    A1D = Buvy * A1D

    # Calculate average in x-direction; no boundary conditions
    diag1 = weight * ones(Npx + 1)
    A1Dx = spdiagm(Npx + 1, Npx + 2, 0 => diag1, 1 => diag1)
    A2Dx = kron(I(Nvy_in), A1Dx)


    # Boundary conditions for ν in y-direction;
    # Mapping from Npy (ν) points to Npy+2 points
    Aν_vx_bc_lu = bc_general_stag(
        Npy + 2,
        Npy,
        2,
        boundary_conditions.ν.y[1],
        boundary_conditions.ν.y[2],
        hy[1],
        hy[end],
    )

    # Extend BC to 2D
    A2D = (A1D * Aν_vx_bc_lu.B1D) ⊗ I(Npx + 2)


    # Apply boundary conditions also in x-direction:
    Aν_vx_bc_lr = bc_general_stag(
        Npx + 2,
        Npx,
        2,
        boundary_conditions.ν.x[1],
        boundary_conditions.ν.x[2],
        hx[1],
        hx[end],
    )

    A2Dy = A2D * (I(Npy) ⊗ Aν_vx_bc_lr.B1D)

    Aν_vx = A2Dx * A2Dy

    Aν_vx_bc_lu = (; Aν_vx_bc_lu..., B2D = A2Dx * ((A1D * Aν_vx_bc_lu.Btemp) ⊗ I(Npx + 2)))
    Aν_vx_bc_lr = (; Aν_vx_bc_lr..., B2D = A2Dx * A2D * (I(Npy) ⊗ Aν_vx_bc_lr.Btemp))


    ## Nu to vy positions
    A1D = I(Npy + 2)
    # Then map back to Nvy_in+1 points (vy-faces) with Bkvy
    A1D = Bkvy * A1D

    # Boundary conditions for ν; mapping from Npy (ν) points to Npy+2 (vy faces) points
    Aν_vy_bc = bc_general_stag(
        Npy + 2,
        Npy,
        2,
        boundary_conditions.ν.y[1],
        boundary_conditions.ν.y[2],
        hy[1],
        hy[end],
    )

    # Extend to 2D
    Aν_vy = (A1D * Aν_vy_bc.B1D) ⊗ I(Nvx_in)
    Aν_vy_bc = (; Aν_vy_bc..., Bbc = ((A1D * Aν_vy_bc.Btemp) ⊗ I(Nvx_in)))


    # So ν at vy is given by:
    # (Aν_vy * k + yAν_vy)

    ## Get derivatives u_x, u_y, v_x and v_y at cell centers
    # Differencing velocity to ν-points

    ## Du/dx

    # Differencing matrix
    diag1 = 1 ./ hx
    C1D = spdiagm(Npx, Npx + 1, 0 => -diag1, 1 => diag1)

    # Boundary conditions
    Cux_k_bc = bc_general(
        Npx + 1,
        Nux_in,
        Npx + 1 - Nux_in,
        boundary_conditions.u.x[1],
        boundary_conditions.u.x[2],
        hx[1],
        hx[end],
    )

    Cux_k = I(Npy) ⊗ (C1D * Cux_k_bc.B1D)
    Cux_k_bc = (; Cux_k_bc..., Bbc = I(Npy) ⊗ (C1D * Cux_k_bc.Btemp))

    # Cux_k*uₕ+yCux_k;


    ## Du/dy

    # Average to k-positions (in x-dir)
    weight = 1 / 2
    diag1 = weight * ones(Npx)
    A1D = spdiagm(Npx, Npx + 1, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Auy_k_bc = bc_general(
        Npx + 1,
        Nux_in,
        Npx + 1 - Nux_in,
        boundary_conditions.u.x[1],
        boundary_conditions.u.x[2],
        hx[1],
        hx[end],
    )

    Auy_k = I(Npy) ⊗ (A1D * Auy_k_bc.B1D)
    Auy_k_bc = (; Auy_k_bc..., Bbc = I(Npy) ⊗ (A1D * Auy_k_bc.Btemp))

    # Take differences
    gydnew = gyd[1:end-1] + gyd[2:end] # Differencing over 2*Δy
    diag2 = 1 ./ gydnew
    C1D = spdiagm(Npy, Npy + 2, 0 => -diag2, 2 => diag2)

    Cuy_k_bc = bc_general_stag(
        Npy + 2,
        Npy,
        2,
        boundary_conditions.u.y[1],
        boundary_conditions.u.y[2],
        hy[1],
        hy[end],
    )

    Cuy_k = (C1D * Cuy_k_bc.B1D) ⊗ I(Npx)
    Cuy_k_bc = (; Cuy_k_bc..., Bbc = (C1D * Cuy_k_bc.Btemp) ⊗ I(Npx))

    ## Dv/dx

    # Average to k-positions (in y-dir)
    weight = 1 / 2
    diag1 = weight * ones(Npy)
    A1D = spdiagm(Npy, Npy + 1, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Avx_k_bc = bc_general(
        Npy + 1,
        Nvy_in,
        Npy + 1 - Nvy_in,
        boundary_conditions.v.y[1],
        boundary_conditions.v.y[2],
        hy[1],
        hy[end],
    )
    Avx_k = (A1D * Avx_k_bc.B1D) ⊗ I(Npx)
    Avx_k_bc = (; Avx_k_bc..., Bbc = (A1D * Avx_k_bc.Btemp) ⊗ I(Npx))

    # Take differences
    gxdnew = gxd[1:end-1] + gxd[2:end] # Differencing over 2*Δx
    diag2 = 1 ./ gxdnew
    C1D = spdiagm(Npx, Npx + 2, 0 => -diag2, 2 => diag2)

    Cvx_k_bc = bc_general_stag(
        Npx + 2,
        Npx,
        Npx + 2 - Npx,
        boundary_conditions.v.x[1],
        boundary_conditions.v.x[2],
        hx[1],
        hx[end],
    )

    Cvx_k = I(Npy) ⊗ (C1D * Cvx_k_bc.B1D)
    Cvx_k_bc = (; Cvx_k_bc..., Bbc = I(Npy) ⊗ (C1D * Cvx_k_bc.Btemp))

    ## Dv/dy

    # Differencing matrix
    diag1 = 1 ./ hy
    C1D = spdiagm(Npy, Npy + 1, 0 => -diag1, 1 => diag1)

    # Boundary conditions
    Cvy_k_bc = bc_general(
        Npy + 1,
        Nvy_in,
        Npy + 1 - Nvy_in,
        boundary_conditions.v.y[1],
        boundary_conditions.v.y[2],
        hy[1],
        hy[end],
    )

    Cvy_k = (C1D * Cvy_k_bc.B1D) ⊗ I(Npx)
    Cvy_k_bc = (; Cvy_k_bc..., Bbc = (C1D * Cvy_k_bc.Btemp) ⊗ I(Npx))


    ## Group operators
    (;
        Aν_ux,
        Aν_ux_bc,
        Aν_uy,
        Aν_uy_bc_lr,
        Aν_uy_bc_lu,
        Aν_vx,
        Aν_vx_bc_lr,
        Aν_vx_bc_lu,
        Aν_vy,
        Aν_vy_bc,
        Cux_k,
        Cux_k_bc,
        Cuy_k,
        Cuy_k_bc,
        Cvx_k,
        Cvx_k_bc,
        Cvy_k,
        Cvy_k_bc,
        Auy_k,
        Auy_k_bc,
        Avx_k,
        Avx_k_bc,
    )
end

# TODO: 3D implementation
function operator_turbulent_diffusion(grid::Grid{T,3}, boundary_conditions) where {T}
    error("Not implemented (3D)")
end
