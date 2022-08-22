"""
    ke_production(grid, bc)

Differencing velocity to k-points.
"""
function ke_convection end

# 2D version
function ke_production(grid::Grid{T,2}, bc) where {T}
    ## Du/dx

    # Differencing matrix
    diag1 = 1 ./ hx
    C1D = spdiagm(Npx, Npx + 1, 0 => -diag1, 1 => diag1)

    # Boundary conditions
    B1D, Btemp, ybcl, ybcr =
        bc_general(Npx + 1, Nux_in, Npx + 1 - Nux_in, bc.u.x[1], bc.u.x[2], hx[1], hx[end])

    Cux_k = kron(sparse(I, Npy, Npy), C1D * B1D)
    uLe_i = interp1(y, uLe, yp)
    uRi_i = interp1(y, uRi, yp)
    ybc = kron(uLe_i, ybcl) + kron(uRi_i, ybcr)
    yCux_k = kron(sparse(I, Npy, Npy), C1D * Btemp) * ybc

    # Cux_k*uₕ+yCux_k;

    ## Du/dy

    # Average to k-positions (in x-dir)
    weight = 1 / 2
    diag1 = weight * ones(Npx)
    A1D = spdiagm(Npx, Npx + 1, 0 => diag1, 1 => diag1)

    # Boundary conditions
    B1D, Btemp, ybcl, ybcr =
        bc_general(Npx + 1, Nux_in, Npx + 1 - Nux_in, bc.u.x[1], bc.u.x[2], hx[1], hx[end])
    uLe_i = interp1(y, uLe, yp)
    uRi_i = interp1(y, uRi, yp)
    ybc = kron(uLe_i, ybcl) + kron(uRi_i, ybcr)

    Auy_k = kron(sparse(I, Ny, Ny), A1D * B1D)
    yAuy_k = kron(sparse(I, Ny, Ny), A1D * Btemp) * ybc

    # Take differences
    gydnew = gyd[1:end-1] + gyd[2:end] # Differencing over 2*Δy
    diag2 = 1 ./ gydnew
    C1D = spdiagm(Npy, Npy + 2, 0 => -diag2, 1 => diag2)

    B1D, Btemp, ybcl, ybcu =
        bc_general_stag(Npy + 2, Npy, 2, bc.u.y[1], bc.u.y[2], hy[1], hy[end])

    Cuy_k = kron(C1D * B1D, sparse(I, Npx, Npx))
    uLo_i = interp1(x, uLo, xp)
    uUp_i = interp1(x, uUp, xp)
    ybc = kron(ybcl, uLo_i) + kron(ybcu, uUp_i)
    yCuy_k = kron(C1D * Btemp, sparse(I, Npx, Npx)) * ybc

    # Cuy_k*(Auy_k*uₕ+yAuy_k) + yCuy_k;


    ## Dv/dx

    # Average to k-positions (in y-dir)
    weight = 1 / 2
    diag1 = weight * ones(Npy, 1)
    A1D = spdiagm(Npy, Npy + 1, 0 => diag1, 1 => diag1)

    # Boundary conditions
    B1D, Btemp, ybcl, ybcu =
        bc_general(Npy + 1, Nvy_in, Npy + 1 - Nvy_in, bc.v.y[1], bc.v.y[2], hy[1], hy[end])
    vLo_i = interp1(x, vLo, xp)
    vUp_i = interp1(x, vUp, xp)
    ybc = kron(ybcl, vLo_i) + kron(ybcu, vUp_i)
    Avx_k = kron(A1D * B1D, sparse(I, Nx, Nx))
    yAvx_k = kron(A1D * Btemp, sparse(I, Nx, Nx)) * ybc

    # Take differences
    gxdnew = gxd[1:end-1] + gxd[2:end] # Differencing over 2*Δx
    diag2 = 1 ./ gxdnew
    C1D = spdiagm(Npx, Npx + 2, 0 => -diag2, 1 => diag2)

    B1D, Btemp, ybcl, ybcr =
        bc_general_stag(Npx + 2, Npx, Npx + 2 - Npx, bc.v.x[1], bc.v.x[2], hx[1], hx[end])

    Cvx_k = kron(sparse(I, Npy, Npy), C1D * B1D)
    vLe_i = interp1(y, vLe, yp)
    vRi_i = interp1(y, vRi, yp)
    ybc = kron(vLe_i, ybcl) + kron(vRi_i, ybcr)
    yCvx_k = kron(sparse(I, Npy, Npy), C1D * Btemp) * ybc

    # Cvx_k*(Avx_k*vₕ+yAvx_k) + yCvx_k;


    ## Dv/dy

    # Differencing matrix
    diag1 = 1 ./ hy
    C1D = spdiagm(Npy, Npy + 1, 0 => -diag1, 1 => diag1)

    # Boundary conditions
    B1D, Btemp, ybcl, ybcu =
        bc_general(Npy + 1, Nvy_in, Npy + 1 - Nvy_in, bc.v.y[1], bc.v.y[2], hy[1], hy[end])

    Cvy_k = kron(C1D * B1D, sparse(I, Npx, Npx))
    vLo_i = interp1(x, vLo, xp)
    vUp_i = interp1(x, vUp, xp)
    ybc = kron(ybcl, vLo_i) + kron(ybcu, vUp_i)
    yCvy_k = kron(C1D * Btemp, sparse(I, Npx, Npx)) * ybc

    # Cvy_k * vₕ + yCvy_k

    # TODO: Return correct operators
    (;
    )
end

# 3D version
function ke_production(grid::Grid{T,3}, bc) where {T}
    error("Not implemented")
end
