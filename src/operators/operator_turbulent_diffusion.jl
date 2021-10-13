function operator_turbulent_diffusion!(setup)

    ## Average (turbulent) viscosity to cell faces: from nu at xp, yp to nu at
    # Ux, uy, vx, vy locations

    # See also ke_viscosity.m

    # Averaging weight:
    weight = 1 / 2

    bc = setup.bc

    # Number of interior points and boundary points
    @unpack Npx, Npy = setup.grid
    @unpack Nux_in, Nuy_in, Nvx_in, Nvy_in = setup.grid
    @unpack hx, hy, gx, gy, gxd, gyd = setup.grid
    @unpack Buvy, Bvux, Bkux, Bkvy = setup.grid


    # Set BC for nu
    # In the periodic case, the value of nu is not needed
    # In all other cases, homogeneous (zero) Neumann conditions are used

    if bc.u.left == :periodic && bc.u.right == :periodic
        bcnu.left = :periodic
        bcnu.right = :periodic
    else
        bcnu.left = :symmetric
        bcnu.right = :symmetric
    end

    if bc.v.low == :periodic && bc.v.up == :periodic
        bcnu.low = :periodic
        bcnu.up = :periodic
    else
        bcnu.low = :symmetric
        bcnu.up = :symmetric
    end


    ## Nu to ux positions

    A1D = sparse(I, Npx + 2, Npx + 2)
    A1D = Bkux * A1D

    # Boundary conditions for nu; mapping from Npx (k) points to Npx+2 points
    Anu_ux_bc = bc_general_stag(Npx + 2, Npx, 2, bcnu.left, bcnu.right, hx[1], hx[end])
    # Then map back to Nux_in+1 points (ux-faces) with Bkux

    # Extend to 2D
    Anu_ux = kron(sparse(I, Nuy_in, Nuy_in), A1D * Anu_ux_bc.B1D)
    # Ybc = kron(nuLe, ybcl) + kron(nuRi, ybcr);
    # YAnu_ux = kron(sparse(I, Nuy_in, Nuy_in), Bkux*A1D*Anu_ux_bc.Btemp)*ybc;
    Anu_ux_bc.Bbc = kron(sparse(I, Nuy_in, Nuy_in), A1D * Anu_ux_bc.Btemp)


    # So nu at ux is given by:
    # (Anu_ux * nu + yAnu_ux)


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

    # Boundary conditions for nu in x-direction;
    # Mapping from Npx (nu) points to Npx+2 points
    Anu_uy_bc_lr = bc_general_stag(Npx + 2, Npx, 2, bcnu.left, bcnu.right, hx[1], hx[end])
    # Extend BC to 2D
    A2D = kron(sparse(I, Npy + 2, Npy + 2), A1D * Anu_uy_bc_lr.B1D)
    # NuLe_i = [nuLe[1];nuLe;nuLe[end]];
    # NuRi_i = [nuRi[1];nuRi;nuRi[end]];
    # Ybc = kron(nuLe_i, Anu_uy_bc_lr.ybc1)+ kron(nuRi_i, Anu_uy_bc_lr.ybc2);
    # YAnu_uy_lr = kron(sparse(I, Npy+2, Npy+2), A1D*Anu_uy_bc_lr.Btemp)*ybc;

    # Apply bc in y-direction
    Anu_uy_bc_lu = bc_general_stag(Npy + 2, Npy, 2, bcnu.low, bcnu.up, hy[1], hy[end])

    # Ybc = kron(Anu_uy_bc_lu.Btemp*Anu_uy_bc_lu.ybc1, nuLo) + kron(Anu_uy_bc_lu.Btemp*Anu_uy_bc_lu.ybc2, nuUp);
    # YAnu_uy_lu = A2D*ybc;

    A2Dx = A2D * kron(Anu_uy_bc_lu.B1D, sparse(I, Npx, Npx))

    Anu_uy = A2Dy * A2Dx
    # YAnu_uy = A2Dy*(yAnu_uy_lu + yAnu_uy_lr);

    # NEW:
    Anu_uy_bc_lr.B2D = A2Dy * kron(sparse(I, Npy + 2, Npy + 2), A1D * Anu_uy_bc_lr.Btemp)
    # Ybc = kron(nuLe_i, Anu_uy_bc_lr.ybc1)+ kron(nuRi_i, Anu_uy_bc_lr.ybc2);
    # YAnu_uy_lr = Anu_uy_bc_lr.B2D*ybc;

    Anu_uy_bc_lu.B2D = A2Dy * A2D * kron(Anu_uy_bc_lu.Btemp, sparse(I, Npx, Npx))
    # Ybc = kron(Anu_uy_bc_lu.ybc1, nuLo) + kron(Anu_uy_bc_lu.ybc2, nuUp);
    # YAnu_uy_lu = Anu_uy_bc_lu.B2D*ybc;

    # So nu at uy is given by:
    # (Anu_uy * nu + yAnu_uy)


    ## Nu to vx positions
    diag1 = weight * ones(Npy + 1)
    A1D = spdiagm(Npy + 1, Npy + 2, 0 => diag1, 1 => diag1)
    # Map to Nvy_in points (like Iu_vx) with Buvy
    A1D = Buvy * A1D

    # Calculate average in x-direction; no boundary conditions
    diag1 = weight * ones(Npx + 1)
    A1Dx = spdiagm(Npx + 1, Npx + 2, 0 => diag1, 1 => diag1)
    A2Dx = kron(sparse(I, Nvy_in, Nvy_in), A1Dx)


    # Boundary conditions for nu in y-direction;
    # Mapping from Npy (nu) points to Npy+2 points
    Anu_vx_bc_lu = bc_general_stag(Npy + 2, Npy, 2, bcnu.low, bcnu.up, hy[1], hy[end])
    # Extend BC to 2D
    A2D = kron(A1D * Anu_vx_bc_lu.B1D, sparse(I, Npx + 2, Npx + 2))


    # Apply boundary conditions also in x-direction:
    Anu_vx_bc_lr = bc_general_stag(Npx + 2, Npx, 2, bcnu.left, bcnu.right, hx[1], hx[end])

    A2Dy = A2D * kron(sparse(I, Npy, Npy), Anu_vx_bc_lr.B1D)

    Anu_vx = A2Dx * A2Dy

    # OLD:
    # NuLo_i = [nuLo[1];nuLo;nuLo[end]];
    # NuUp_i = [nuUp[1];nuUp;nuUp[end]];
    # Ybc = kron(Anu_vx_bc_lu.ybc1, nuLo_i) + kron(Anu_vx_bc_lu.ybc2, nuUp_i);
    # YAnu_vx_lu = kron(A1D*Anu_vx_bc_lu.Btemp, sparse(I, Npx+2, Npx+2))*ybc;
    # YAnu_vx_lu1 = A2Dx*yAnu_vx_lu;
    #
    # Ybc = kron(nuLe, Anu_vx_bc_lr.Btemp*Anu_vx_bc_lr.ybc1) + kron(nuRi, Anu_vx_bc_lr.Btemp*Anu_vx_bc_lr.ybc2);
    # YAnu_vx_lr1 = A2Dx*A2D*ybc;


    # NEW:
    Anu_vx_bc_lu.B2D = A2Dx * kron(A1D * Anu_vx_bc_lu.Btemp, sparse(I, Npx + 2, Npx + 2))
    Anu_vx_bc_lr.B2D = A2Dx * A2D * kron(sparse(I, Npy, Npy), Anu_vx_bc_lr.Btemp)

    # # in y-direction
    # Ybc = kron(Anu_vx_bc_lu.ybc1, nuLo_i) + kron(Anu_vx_bc_lu.ybc2, nuUp_i);
    # YAnu_vx_lu2 = Anu_vx_lu.B2D*ybc;
    # # in x-direction
    # Ybc = kron(nuLe, Anu_vx_bc_lr.ybc1) + kron(nuRi, Anu_vx_bc_lr.ybc2);
    # YAnu_vx_lr2 = Anu_vx_lr.B2D*ybc;

    # So nu at uy is given by:
    # (Anu_vx * nu + yAnu_vx)


    ## Nu to vy positions
    A1D = sparse(I, Npy + 2, Npy + 2)
    # Then map back to Nvy_in+1 points (vy-faces) with Bkvy
    A1D = Bkvy * A1D

    # Boundary conditions for nu; mapping from Npy (nu) points to Npy+2 (vy faces) points
    Anu_vy_bc = bc_general_stag(Npy + 2, Npy, 2, bcnu.low, bcnu.up, hy[1], hy[end])

    # Extend to 2D
    Anu_vy = kron(A1D * Anu_vy_bc.B1D, sparse(I, Nvx_in, Nvx_in))
    Anu_vy_bc.Bbc = kron(A1D * Anu_vy_bc.Btemp, sparse(I, Nvx_in, Nvx_in))


    # So nu at vy is given by:
    # (Anu_vy * k + yAnu_vy)

    ## Store in struct
    setup.discretization.Anu_ux = Anu_ux
    setup.discretization.Anu_ux_bc = Anu_ux_bc

    setup.discretization.Anu_uy = Anu_uy
    setup.discretization.Anu_uy_bc_lr = Anu_uy_bc_lr
    setup.discretization.Anu_uy_bc_lu = Anu_uy_bc_lu

    setup.discretization.Anu_vx = Anu_vx
    setup.discretization.Anu_vx_bc_lr = Anu_vx_bc_lr
    setup.discretization.Anu_vx_bc_lu = Anu_vx_bc_lu

    setup.discretization.Anu_vy = Anu_vy
    setup.discretization.Anu_vy_bc = Anu_vy_bc


    ## Get derivatives u_x, u_y, v_x and v_y at cell centers
    # Differencing velocity to nu-points

    ## Du/dx

    # Differencing matrix
    diag1 = 1 ./ hx
    C1D = spdiagm(Npx, Npx + 1, 0 => -diag1, 1 => diag1)

    # Boundary conditions
    Cux_k_bc =
        bc_general(Npx + 1, Nux_in, Npx + 1 - Nux_in, bc.u.left, bc.u.right, hx[1], hx[end])

    Cux_k = kron(sparse(I, Npy, Npy), C1D * Cux_k_bc.B1D)
    Cux_k_bc.Bbc = kron(sparse(I, Npy, Npy), C1D * Cux_k_bc.Btemp)

    # Cux_k*uₕ+yCux_k;


    ## Du/dy

    # Average to k-positions (in x-dir)
    weight = 1 / 2
    diag1 = weight * ones(Npx)
    A1D = spdiagm(Npx, Npx + 1, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Auy_k_bc =
        bc_general(Npx + 1, Nux_in, Npx + 1 - Nux_in, bc.u.left, bc.u.right, hx[1], hx[end])
    # ULe_i = interp1(y, uLe, yp);
    # URi_i = interp1(y, uRi, yp);
    # Ybc = kron(uLe_i, ybcl) + kron(uRi_i, ybcr);

    Auy_k = kron(sparse(I, Npy, Npy), A1D * Auy_k_bc.B1D)
    Auy_k_bc.Bbc = kron(sparse(I, Npy, Npy), A1D * Auy_k_bc.Btemp)
    # YAuy_k = kron(sparse(I, Ny, Ny), A1D*Auy_k_bc.Btemp)*ybc;

    # Take differences
    gydnew = gyd[1:end-1] + gyd[2:end] # Differencing over 2*deltay
    diag2 = 1 ./ gydnew
    C1D = spdiagm(Npy, Npy + 2, 0 => -diag2, 2 => diag2)

    Cuy_k_bc = bc_general_stag(Npy + 2, Npy, 2, bc.u.low, bc.u.up, hy[1], hy[end])

    Cuy_k = kron(C1D * Cuy_k_bc.B1D, sparse(I, Npx, Npx))
    Cuy_k_bc.Bbc = kron(C1D * Cuy_k_bc.Btemp, sparse(I, Npx, Npx))
    # ULo_i = interp1(x, uLo, xp);
    # UUp_i = interp1(x, uUp, xp);
    # Ybc = kron(ybcl, uLo_i) + kron(ybcu, uUp_i);
    # YCuy_k = kron(C1D*Cuy_k_bc.Btemp, sparse(I, Npx, Npx))*ybc;

    # Cuy_k*(Auy_k*uₕ+yAuy_k) + yCuy_k

    ## Dv/dx

    # Average to k-positions (in y-dir)
    weight = 1 / 2
    diag1 = weight * ones(Npy)
    A1D = spdiagm(Npy, Npy + 1, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Avx_k_bc =
        bc_general(Npy + 1, Nvy_in, Npy + 1 - Nvy_in, bc.v.low, bc.v.up, hy[1], hy[end])
    # VLo_i = interp1(x, vLo, xp);
    # VUp_i = interp1(x, vUp, xp);
    # Ybc = kron(ybcl, vLo_i) + kron(ybcu, vUp_i);
    Avx_k = kron(A1D * Avx_k_bc.B1D, sparse(I, Npx, Npx))
    Avx_k_bc.Bbc = kron(A1D * Avx_k_bc.Btemp, sparse(I, Npx, Npx))
    # YAvx_k = kron(A1D*Btemp, sparse(I, Nx, Nx))*ybc;

    # Take differences
    gxdnew = gxd[1:end-1] + gxd[2:end] # Differencing over 2*deltax
    diag2 = 1 ./ gxdnew
    C1D = spdiagm(Npx, Npx + 2, 0 => -diag2, 2 => diag2)

    Cvx_k_bc =
        bc_general_stag(Npx + 2, Npx, Npx + 2 - Npx, bc.v.left, bc.v.right, hx[1], hx[end])

    Cvx_k = kron(sparse(I, Npy, Npy), C1D * Cvx_k_bc.B1D)
    # VLe_i = interp1(y, vLe, yp);
    # VRi_i = interp1(y, vRi, yp);
    Cvx_k_bc.Bbc = kron(sparse(I, Npy, Npy), C1D * Cvx_k_bc.Btemp)

    # Cvx_k*(Avx_k*vₕ+yAvx_k) + yCvx_k;


    ## Dv/dy

    # Differencing matrix
    diag1 = 1 ./ hy
    C1D = spdiagm(Npy, Npy + 1, 0 => -diag1, 1 => diag1)

    # Boundary conditions
    Cvy_k_bc =
        bc_general(Npy + 1, Nvy_in, Npy + 1 - Nvy_in, bc.v.low, bc.v.up, hy[1], hy[end])

    Cvy_k = kron(C1D * Cvy_k_bc.B1D, sparse(I, Npx, Npx))
    # VLo_i = interp1(x, vLo, xp);
    # VUp_i = interp1(x, vUp, xp);
    # Ybc = kron(ybcl, vLo_i) + kron(ybcu, vUp_i);
    Cvy_k_bc.Bbc = kron(C1D * Cvy_k_bc.Btemp, sparse(I, Npx, Npx))

    # Cvy_k*vₕ+yCvy_k;

    ## Store in struct
    setup.discretization.Cux_k = Cux_k
    setup.discretization.Cux_k_bc = Cux_k_bc
    setup.discretization.Cuy_k = Cuy_k
    setup.discretization.Cuy_k_bc = Cuy_k_bc
    setup.discretization.Cvx_k = Cvx_k
    setup.discretization.Cvx_k_bc = Cvx_k_bc
    setup.discretization.Cvy_k = Cvy_k
    setup.discretization.Cvy_k_bc = Cvy_k_bc

    setup.discretization.Auy_k = Auy_k
    setup.discretization.Auy_k_bc = Auy_k_bc
    setup.discretization.Avx_k = Avx_k
    setup.discretization.Avx_k_bc = Avx_k_bc

    setup
end
