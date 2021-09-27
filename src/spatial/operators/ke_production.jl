# differencing velocity to k-points
function ke_production!(setup)
    ## du/dx

    #differencing matrix
    diag1 = 1 ./ hx
    C1D = spdiagm(Npx, Npx + 1, 0 => -diag1, 1 => diag1)

    # boundary conditions
    B1D, Btemp, ybcl, ybcr =
        BC_general(Npx + 1, Nux_in, Npx + 1 - Nux_in, BC.u.left, BC.u.right, hx[1], hx[end])

    Cux_k = kron(sparse(I, Npy, Npy), C1D * B1D)
    uLe_i = interp1(y, uLe, yp)
    uRi_i = interp1(y, uRi, yp)
    ybc = kron(uLe_i, ybcl) + kron(uRi_i, ybcr)
    yCux_k = kron(sparse(I, Npy, Npy), C1D * Btemp) * ybc

    # Cux_k*uh+yCux_k;


    ## du/dy

    # average to k-positions (in x-dir)
    weight = 1 / 2
    diag1 = weight * ones(Npx)
    A1D = spdiagm(Npx, Npx + 1, 0 => diag1, 1 => diag1)

    # boundary conditions
    B1D, Btemp, ybcl, ybcr =
        BC_general(Npx + 1, Nux_in, Npx + 1 - Nux_in, BC.u.left, BC.u.right, hx[1], hx[end])
    uLe_i = interp1(y, uLe, yp)
    uRi_i = interp1(y, uRi, yp)
    ybc = kron(uLe_i, ybcl) + kron(uRi_i, ybcr)

    Auy_k = kron(sparse(I, Ny, Ny), A1D * B1D)
    yAuy_k = kron(sparse(I, Ny, Ny), A1D * Btemp) * ybc

    # take differences
    gydnew = gyd[1:end-1] + gyd[2:end] # differencing over 2*deltay
    diag2 = 1 ./ gydnew
    C1D = spdiagm(Npy, Npy + 2, 0 => -diag2, 1 => diag2)

    B1D, Btemp, ybcl, ybcu =
        BC_general_stag(Npy + 2, Npy, 2, BC.u.low, BC.u.up, hy[1], hy[end])

    Cuy_k = kron(C1D * B1D, sparse(I, Npx, Npx))
    uLo_i = interp1(x, uLo, xp)
    uUp_i = interp1(x, uUp, xp)
    ybc = kron(ybcl, uLo_i) + kron(ybcu, uUp_i)
    yCuy_k = kron(C1D * Btemp, sparse(I, Npx, Npx)) * ybc

    # Cuy_k*(Auy_k*uh+yAuy_k) + yCuy_k;


    ## dv/dx

    # average to k-positions (in y-dir)
    weight = 1 / 2
    diag1 = weight * ones(Npy, 1)
    A1D = spdiagm(Npy, Npy + 1, 0 => diag1, 1 => diag1)

    # boundary conditions
    B1D, Btemp, ybcl, ybcu =
        BC_general(Npy + 1, Nvy_in, Npy + 1 - Nvy_in, BC.v.low, BC.v.up, hy[1], hy[end])
    vLo_i = interp1(x, vLo, xp)
    vUp_i = interp1(x, vUp, xp)
    ybc = kron(ybcl, vLo_i) + kron(ybcu, vUp_i)
    Avx_k = kron(A1D * B1D, sparse(I, Nx, Nx))
    yAvx_k = kron(A1D * Btemp, sparse(I, Nx, Nx)) * ybc

    # take differences
    gxdnew = gxd[1:end-1] + gxd[2:end] # differencing over 2*deltax
    diag2 = 1 ./ gxdnew
    C1D = spdiagm(Npx, Npx + 2, 0 => -diag2, 1 => diag2)

    B1D, Btemp, ybcl, ybcr =
        BC_general_stag(Npx + 2, Npx, Npx + 2 - Npx, BC.v.left, BC.v.right, hx[1], hx[end])

    Cvx_k = kron(sparse(I, Npy, Npy), C1D * B1D)
    vLe_i = interp1(y, vLe, yp)
    vRi_i = interp1(y, vRi, yp)
    ybc = kron(vLe_i, ybcl) + kron(vRi_i, ybcr)
    yCvx_k = kron(sparse(I, Npy, Npy), C1D * Btemp) * ybc

    # Cvx_k*(Avx_k*vh+yAvx_k) + yCvx_k;


    ## dv/dy

    # differencing matrix
    diag1 = 1 ./ hy
    C1D = spdiagm(Npy, Npy + 1, 0 => -diag1, 1 => diag1)

    # boundary conditions
    B1D, Btemp, ybcl, ybcu =
        BC_general(Npy + 1, Nvy_in, Npy + 1 - Nvy_in, BC.v.low, BC.v.up, hy[1], hy[end])

    Cvy_k = kron(C1D * B1D, sparse(I, Npx, Npx))
    vLo_i = interp1(x, vLo, xp)
    vUp_i = interp1(x, vUp, xp)
    ybc = kron(ybcl, vLo_i) + kron(ybcu, vUp_i)
    yCvy_k = kron(C1D * Btemp, sparse(I, Npx, Npx)) * ybc

    # Cvy_k*vh+yCvy_k;
    setup
end
