function ke_diffusion!(setup)
    # nu_T = C_mu*(Akx*k)^2/(Aex*e)
    # Dkx*(Anu*nu+C_mu*(Akx*k)^2/(Aex*e)).*Skx*k

    ##############
    # x-direction

    ## differencing from faces to centers
    diag1 = ones(Npx) # Nkx = Npx
    D1D = spdiagm(Npx, Npx + 1, 0 => -diag1, 1 => diag1)
    # No BC
    Dkx = kron(mat_hy, D1D)


    ## averaging from centers to faces
    diag2 = 0.5 * ones(Npx + 1)
    A1D = spdiagm(Npx + 1, Npx + 2, 0 => diag2, 1 => diag2)

    # BCs for k
    # Ak_kx is already constructed in ke_convection
    [B1Dk, Btempk, ybcl, ybcr] =
        BC_general_stag(Npx + 2, Npx, 2, BC.k.left, BC.k.right, hx[1], hx[end])
    ybck = kron(kLe, ybcl) + kron(kRi, ybcr)
    yAk_kx = kron(sparse(I, Npy, Npy), A1D * Btempk) * ybck
    Ak_kx = kron(sparse(I, Npy, Npy), A1D * B1Dk)

    # BCs for e
    [B1De, Btempe, ybcl, ybcr] =
        BC_general_stag(Npx + 2, Npx, 2, BC.e.left, BC.e.right, hx[1], hx[end])
    ybce = kron(eLe, ybcl) + kron(eRi, ybcr)
    yAe_ex = kron(sparse(I, Npy, Npy), A1D * Btempe) * ybce
    Ae_ex = kron(sparse(I, Npy, Npy), A1D * B1De)


    ## differencing from centers to faces
    diag3 = 1 ./ gxd
    S1D = spdiagm(Npx + 1, Npx + 2, 0 => -diag3, 1 => diag3)

    # re-use BC generated for averaging k
    Skx = kron(sparse(I, Npy, Npy), S1D * B1Dk)
    ySkx = kron(sparse(I, Npy, Npy), S1D * Btempk) * ybck

    # re-use BC generated for averaging e
    Sex = kron(sparse(I, Npy, Npy), S1D * B1De)
    ySex = kron(sparse(I, Npy, Npy), S1D * Btempe) * ybce


    ##############
    # y-direction


    ## differencing from faces to centers
    diag1 = ones(Npy) # Nky = Npy
    D1D = spdiagm(Npy, Npy + 1, 0 => -diag1, 1 => diag1)
    # No BC
    Dky = kron(D1D, mat_hx)


    ## averaging
    diag2 = 0.5 * ones(Npy + 1)
    A1D = spdiagm(Npy + 1, Npy + 2, 0 => diag2, 1 => diag2)

    # BCs for k:
    # k is already constructed in ke_convection
    B1Dk, Btempk, ybcl, ybcu =
        BC_general_stag(Npy + 2, Npy, 2, BC.k.low, BC.k.up, hy[1], hy[end])
    ybck = kron(ybcl, kLo) + kron(ybcu, kUp)
    yAk_ky = kron(A1D * Btempk, sparse(I, Npx, Npx)) * ybck
    Ak_ky = kron(A1D * B1Dk, sparse(I, Npx, Npx))

    # BCs for e:
    B1De, Btempe, ybcl, ybcu =
        BC_general_stag(Npy + 2, Npy, 2, BC.e.low, BC.e.up, hy[1], hy[end])
    ybce = kron(ybcl, eLo) + kron(ybcu, eUp)
    yAe_ey = kron(A1D * Btempe, sparse(I, Npx, Npx)) * ybce
    Ae_ey = kron(A1D * B1De, sparse(I, Npx, Npx))


    ## differencing from centers to faces
    diag3 = 1 ./ gyd
    S1D = spdiagm(Npy + 1, Npy + 2, 0 => -diag3, 1 => diag3)

    # re-use BC generated for averaging k
    Sky = kron(S1D * B1Dk, sparse(I, Npx, Npx))
    ySky = kron(S1D * Btempk, sparse(I, Npx, Npx)) * ybck

    # re-use BC generated for averaging e
    Sey = kron(S1D * B1De, sparse(I, Npx, Npx))
    ySey = kron(S1D * Btempe, sparse(I, Npx, Npx)) * ybce

    setup
end
