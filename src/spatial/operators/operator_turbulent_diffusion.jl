function operator_turbulent_diffusion!(setup)

    ## average (turbulent) viscosity to cell faces: from nu at xp, yp to nu at
    # ux, uy, vx, vy locations

    # see also ke_viscosity.m

    # averaging weight:
    weight = 1 / 2

    BC = setup.BC

    # number of interior points and boundary points
    @unpack Npx, Npy = setup.grid
    @unpack Nux_in, Nuy_in, Nvx_in, Nvy_in = setup.grid
    @unpack hx, hy, gx, gy, gxd, gyd = setup.grid
    @unpack Buvy, Bvux, Bkux, Bkvy = setup.grid


    # set BC for nu
    # in the periodic case, the value of nu is not needed
    # in all other cases, homogeneous (zero) Neumann conditions are used

    if BC.u.left == "per" && BC.u.right == "per"
        BC.nu.left = "per"
        BC.nu.right = "per"
    else
        BC.nu.left = "sym"
        BC.nu.right = "sym"
    end

    if BC.v.low == "per" && BC.v.up == "per"
        BC.nu.low = "per"
        BC.nu.up = "per"
    else
        BC.nu.low = "sym"
        BC.nu.up = "sym"
    end


    ## nu to ux positions

    A1D = sparse(I, Npx + 2, Npx + 2)
    A1D = Bkux * A1D

    # boundary conditions for nu; mapping from Npx (k) points to Npx+2 points
    Anu_ux_BC = BC_general_stag(Npx + 2, Npx, 2, BC.nu.left, BC.nu.right, hx[1], hx[end])
    # then map back to Nux_in+1 points (ux-faces) with Bkux

    # extend to 2D
    Anu_ux = kron(sparse(I, Nuy_in, Nuy_in), A1D * Anu_ux_BC.B1D)
    # ybc = kron(nuLe, ybcl) + kron(nuRi, ybcr);
    # yAnu_ux = kron(sparse(I, Nuy_in, Nuy_in), Bkux*A1D*Anu_ux_BC.Btemp)*ybc;
    Anu_ux_BC.Bbc = kron(sparse(I, Nuy_in, Nuy_in), A1D * Anu_ux_BC.Btemp)


    # so nu at ux is given by:
    # (Anu_ux * nu + yAnu_ux)


    ## nu to uy positions

    # average in x-direction
    diag1 = weight * ones(Npx + 1)
    A1D = spdiagm(Npx + 1, Npx + 2, 0 => diag1, 1 => diag1)
    # then map to Nux_in points (like Iv_uy) with Bvux
    A1D = Bvux * A1D

    # calculate average in y-direction; no boundary conditions
    diag1 = weight * ones(Npy + 1)
    A1Dy = spdiagm(Npy + 1, Npy + 2, 0 => diag1, 1 => diag1)
    A2Dy = kron(A1Dy, sparse(I, Nux_in, Nux_in))

    # boundary conditions for nu in x-direction;
    # mapping from Npx (nu) points to Npx+2 points
    Anu_uy_BC_lr = BC_general_stag(Npx + 2, Npx, 2, BC.nu.left, BC.nu.right, hx[1], hx[end])
    # extend BC to 2D
    A2D = kron(sparse(I, Npy + 2, Npy + 2), A1D * Anu_uy_BC_lr.B1D)
    # nuLe_i = [nuLe[1];nuLe;nuLe[end]];
    # nuRi_i = [nuRi[1];nuRi;nuRi[end]];
    # ybc = kron(nuLe_i, Anu_uy_BC_lr.ybc1)+ kron(nuRi_i, Anu_uy_BC_lr.ybc2);
    # yAnu_uy_lr = kron(sparse(I, Npy+2, Npy+2), A1D*Anu_uy_BC_lr.Btemp)*ybc;

    # apply bc in y-direction
    Anu_uy_BC_lu = BC_general_stag(Npy + 2, Npy, 2, BC.nu.low, BC.nu.up, hy[1], hy[end])

    # ybc = kron(Anu_uy_BC_lu.Btemp*Anu_uy_BC_lu.ybc1, nuLo) + kron(Anu_uy_BC_lu.Btemp*Anu_uy_BC_lu.ybc2, nuUp);
    # yAnu_uy_lu = A2D*ybc;

    A2Dx = A2D * kron(Anu_uy_BC_lu.B1D, sparse(I, Npx, Npx))

    Anu_uy = A2Dy * A2Dx
    # yAnu_uy = A2Dy*(yAnu_uy_lu + yAnu_uy_lr);

    # NEW:
    Anu_uy_BC_lr.B2D = A2Dy * kron(sparse(I, Npy + 2, Npy + 2), A1D * Anu_uy_BC_lr.Btemp)
    # ybc = kron(nuLe_i, Anu_uy_BC_lr.ybc1)+ kron(nuRi_i, Anu_uy_BC_lr.ybc2);
    # yAnu_uy_lr = Anu_uy_BC_lr.B2D*ybc;

    Anu_uy_BC_lu.B2D = A2Dy * A2D * kron(Anu_uy_BC_lu.Btemp, sparse(I, Npx, Npx))
    # ybc = kron(Anu_uy_BC_lu.ybc1, nuLo) + kron(Anu_uy_BC_lu.ybc2, nuUp);
    # yAnu_uy_lu = Anu_uy_BC_lu.B2D*ybc;

    # so nu at uy is given by:
    # (Anu_uy * nu + yAnu_uy)


    ## nu to vx positions
    diag1 = weight * ones(Npy + 1)
    A1D = spdiagm(Npy + 1, Npy + 2, 0 => diag1, 1 => diag1)
    # map to Nvy_in points (like Iu_vx) with Buvy
    A1D = Buvy * A1D

    # calculate average in x-direction; no boundary conditions
    diag1 = weight * ones(Npx + 1)
    A1Dx = spdiagm(Npx + 1, Npx + 2, 0 => diag1, 1 => diag1)
    A2Dx = kron(sparse(I, Nvy_in, Nvy_in), A1Dx)


    # boundary conditions for nu in y-direction;
    # mapping from Npy (nu) points to Npy+2 points
    Anu_vx_BC_lu = BC_general_stag(Npy + 2, Npy, 2, BC.nu.low, BC.nu.up, hy[1], hy[end])
    # extend BC to 2D
    A2D = kron(A1D * Anu_vx_BC_lu.B1D, sparse(I, Npx + 2, Npx + 2))


    # apply boundary conditions also in x-direction:
    Anu_vx_BC_lr = BC_general_stag(Npx + 2, Npx, 2, BC.nu.left, BC.nu.right, hx[1], hx[end])

    A2Dy = A2D * kron(sparse(I, Npy, Npy), Anu_vx_BC_lr.B1D)

    Anu_vx = A2Dx * A2Dy

    # OLD:
    # nuLo_i = [nuLo[1];nuLo;nuLo[end]];
    # nuUp_i = [nuUp[1];nuUp;nuUp[end]];
    # ybc = kron(Anu_vx_BC_lu.ybc1, nuLo_i) + kron(Anu_vx_BC_lu.ybc2, nuUp_i);
    # yAnu_vx_lu = kron(A1D*Anu_vx_BC_lu.Btemp, sparse(I, Npx+2, Npx+2))*ybc;
    # yAnu_vx_lu1 = A2Dx*yAnu_vx_lu;
    #
    # ybc = kron(nuLe, Anu_vx_BC_lr.Btemp*Anu_vx_BC_lr.ybc1) + kron(nuRi, Anu_vx_BC_lr.Btemp*Anu_vx_BC_lr.ybc2);
    # yAnu_vx_lr1 = A2Dx*A2D*ybc;


    # NEW:
    Anu_vx_BC_lu.B2D = A2Dx * kron(A1D * Anu_vx_BC_lu.Btemp, sparse(I, Npx + 2, Npx + 2))
    Anu_vx_BC_lr.B2D = A2Dx * A2D * kron(sparse(I, Npy, Npy), Anu_vx_BC_lr.Btemp)

    # # in y-direction
    # ybc = kron(Anu_vx_BC_lu.ybc1, nuLo_i) + kron(Anu_vx_BC_lu.ybc2, nuUp_i);
    # yAnu_vx_lu2 = Anu_vx_lu.B2D*ybc;
    # # in x-direction
    # ybc = kron(nuLe, Anu_vx_BC_lr.ybc1) + kron(nuRi, Anu_vx_BC_lr.ybc2);
    # yAnu_vx_lr2 = Anu_vx_lr.B2D*ybc;

    # so nu at uy is given by:
    # (Anu_vx * nu + yAnu_vx)


    ## nu to vy positions
    A1D = sparse(I, Npy + 2, Npy + 2)
    # then map back to Nvy_in+1 points (vy-faces) with Bkvy
    A1D = Bkvy * A1D

    # boundary conditions for nu; mapping from Npy (nu) points to Npy+2 (vy faces) points
    Anu_vy_BC = BC_general_stag(Npy + 2, Npy, 2, BC.nu.low, BC.nu.up, hy[1], hy[end])

    # extend to 2D
    Anu_vy = kron(A1D * Anu_vy_BC.B1D, sparse(I, Nvx_in, Nvx_in))
    Anu_vy_BC.Bbc = kron(A1D * Anu_vy_BC.Btemp, sparse(I, Nvx_in, Nvx_in))


    # so nu at vy is given by:
    # (Anu_vy * k + yAnu_vy)

    ## store in struct
    setup.discretization.Anu_ux = Anu_ux
    setup.discretization.Anu_ux_BC = Anu_ux_BC

    setup.discretization.Anu_uy = Anu_uy
    setup.discretization.Anu_uy_BC_lr = Anu_uy_BC_lr
    setup.discretization.Anu_uy_BC_lu = Anu_uy_BC_lu

    setup.discretization.Anu_vx = Anu_vx
    setup.discretization.Anu_vx_BC_lr = Anu_vx_BC_lr
    setup.discretization.Anu_vx_BC_lu = Anu_vx_BC_lu

    setup.discretization.Anu_vy = Anu_vy
    setup.discretization.Anu_vy_BC = Anu_vy_BC


    ## Get derivatives u_x, u_y, v_x and v_y at cell centers
    # differencing velocity to nu-points

    ## du/dx

    #differencing matrix
    diag1 = 1 ./ hx
    C1D = spdiagm(Npx, Npx + 1, 0 => -diag1, 1 => diag1)

    # boundary conditions
    Cux_k_BC =
        BC_general(Npx + 1, Nux_in, Npx + 1 - Nux_in, BC.u.left, BC.u.right, hx[1], hx[end])

    Cux_k = kron(sparse(I, Npy, Npy), C1D * Cux_k_BC.B1D)
    Cux_k_BC.Bbc = kron(sparse(I, Npy, Npy), C1D * Cux_k_BC.Btemp)

    # Cux_k*uh+yCux_k;


    ## du/dy

    # average to k-positions (in x-dir)
    weight = 1 / 2
    diag1 = weight * ones(Npx)
    A1D = spdiagm(Npx, Npx + 1, 0 => diag1, 1 => diag1)

    # boundary conditions
    Auy_k_BC =
        BC_general(Npx + 1, Nux_in, Npx + 1 - Nux_in, BC.u.left, BC.u.right, hx[1], hx[end])
    # uLe_i = interp1(y, uLe, yp);
    # uRi_i = interp1(y, uRi, yp);
    # ybc = kron(uLe_i, ybcl) + kron(uRi_i, ybcr);

    Auy_k = kron(sparse(I, Npy, Npy), A1D * Auy_k_BC.B1D)
    Auy_k_BC.Bbc = kron(sparse(I, Npy, Npy), A1D * Auy_k_BC.Btemp)
    # yAuy_k = kron(sparse(I, Ny, Ny), A1D*Auy_k_BC.Btemp)*ybc;

    # take differences
    gydnew = gyd[1:end-1] + gyd[2:end] # differencing over 2*deltay
    diag2 = 1 ./ gydnew
    C1D = spdiagm(Npy, Npy + 2, 0 => -diag2, 2 => diag2)

    Cuy_k_BC = BC_general_stag(Npy + 2, Npy, 2, BC.u.low, BC.u.up, hy[1], hy[end])

    Cuy_k = kron(C1D * Cuy_k_BC.B1D, sparse(I, Npx, Npx))
    Cuy_k_BC.Bbc = kron(C1D * Cuy_k_BC.Btemp, sparse(I, Npx, Npx))
    # uLo_i = interp1(x, uLo, xp);
    # uUp_i = interp1(x, uUp, xp);
    # ybc = kron(ybcl, uLo_i) + kron(ybcu, uUp_i);
    # yCuy_k = kron(C1D*Cuy_k_BC.Btemp, sparse(I, Npx, Npx))*ybc;

    # Cuy_k*(Auy_k*uh+yAuy_k) + yCuy_k

    ## dv/dx

    # average to k-positions (in y-dir)
    weight = 1 / 2
    diag1 = weight * ones(Npy)
    A1D = spdiagm(Npy, Npy + 1, 0 => diag1, 1 => diag1)

    # boundary conditions
    Avx_k_BC =
        BC_general(Npy + 1, Nvy_in, Npy + 1 - Nvy_in, BC.v.low, BC.v.up, hy[1], hy[end])
    # vLo_i = interp1(x, vLo, xp);
    # vUp_i = interp1(x, vUp, xp);
    # ybc = kron(ybcl, vLo_i) + kron(ybcu, vUp_i);
    Avx_k = kron(A1D * Avx_k_BC.B1D, sparse(I, Npx, Npx))
    Avx_k_BC.Bbc = kron(A1D * Avx_k_BC.Btemp, sparse(I, Npx, Npx))
    # yAvx_k = kron(A1D*Btemp, sparse(I, Nx, Nx))*ybc;

    # take differences
    gxdnew = gxd[1:end-1] + gxd[2:end] # differencing over 2*deltax
    diag2 = 1 ./ gxdnew
    C1D = spdiagm(Npx, Npx + 2, 0 => -diag2, 2 => diag2)

    Cvx_k_BC =
        BC_general_stag(Npx + 2, Npx, Npx + 2 - Npx, BC.v.left, BC.v.right, hx[1], hx[end])

    Cvx_k = kron(sparse(I, Npy, Npy), C1D * Cvx_k_BC.B1D)
    # vLe_i = interp1(y, vLe, yp);
    # vRi_i = interp1(y, vRi, yp);
    Cvx_k_BC.Bbc = kron(sparse(I, Npy, Npy), C1D * Cvx_k_BC.Btemp)

    # Cvx_k*(Avx_k*vh+yAvx_k) + yCvx_k;


    ## dv/dy

    # differencing matrix
    diag1 = 1 ./ hy
    C1D = spdiagm(Npy, Npy + 1, 0 => -diag1, 1 => diag1)

    # boundary conditions
    Cvy_k_BC =
        BC_general(Npy + 1, Nvy_in, Npy + 1 - Nvy_in, BC.v.low, BC.v.up, hy[1], hy[end])

    Cvy_k = kron(C1D * Cvy_k_BC.B1D, sparse(I, Npx, Npx))
    # vLo_i = interp1(x, vLo, xp);
    # vUp_i = interp1(x, vUp, xp);
    # ybc = kron(ybcl, vLo_i) + kron(ybcu, vUp_i);
    Cvy_k_BC.Bbc = kron(C1D * Cvy_k_BC.Btemp, sparse(I, Npx, Npx))

    # Cvy_k*vh+yCvy_k;

    ## store in struct
    setup.discretization.Cux_k = Cux_k
    setup.discretization.Cux_k_BC = Cux_k_BC
    setup.discretization.Cuy_k = Cuy_k
    setup.discretization.Cuy_k_BC = Cuy_k_BC
    setup.discretization.Cvx_k = Cvx_k
    setup.discretization.Cvx_k_BC = Cvx_k_BC
    setup.discretization.Cvy_k = Cvy_k
    setup.discretization.Cvy_k_BC = Cvy_k_BC

    setup.discretization.Auy_k = Auy_k
    setup.discretization.Auy_k_BC = Auy_k_BC
    setup.discretization.Avx_k = Avx_k
    setup.discretization.Avx_k_BC = Avx_k_BC

    setup
end
