# average (turbulent) viscosity to cell faces
function ke_viscosity!(setup)
    # averaging weight:
    weight = 1 / 2


    ## nu to ux positions

    A1D = sparse(I, Npx + 2, Npx + 2)

    ## k
    # boundary conditions for k; mapping from Npx (k) points to Npx+2 points
    B1D, Btemp, ybcl, ybcr =
        bc_general_stag(Npx + 2, Npx, 2, bck.left, bck.right, hx[1], hx[end])
    # then map back to Nux_in+1 points (ux-faces) with Bkux

    # extend to 2D
    Ak_ux = kron(sparse(I, Nuy_in, Nuy_in), Bkux * A1D * B1D)
    ybc = kron(kLe, ybcl) + kron(kRi, ybcr)
    yAk_ux = kron(sparse(I, Nuy_in, Nuy_in), Bkux * A1D * Btemp) * ybc

    ## epsilon
    #in a similar way but with different boundary conditions
    B1D, Btemp, ybcl, ybcr =
        bc_general_stag(Npx + 2, Npx, 2, bce.left, bce.right, hx[1], hx[end])
    # extend to 2D
    Ae_ux = kron(sparse(I, Nuy_in, Nuy_in), Bkux * A1D * B1D)
    ybc = kron(eLe, ybcl) + kron(eRi, ybcr)
    yAe_ux = kron(sparse(I, Nuy_in, Nuy_in), Bkux * A1D * Btemp) * ybc


    # so nu at ux is given by:
    # Cmu * (Ak_ux * k + yAk_ux).^2 / (Ae_ux * e + yAe_ux)


    ## nu to uy positions
    diag1 = weight * ones(Npx + 1)
    A1D = spdiagm(Npx + 1, Npx + 2, 0 => diag1, 1 => diag1)

    ## k
    # boundary conditions for k; mapping from Npx (k) points to Npx+2 points
    [B1D, Btemp, ybcl, ybcr] =
        bc_general_stag(Npx + 2, Npx, 2, bck.left, bck.right, hx[1], hx[end])

    # then map to Nux_in points (like Iv_uy) with Bvux

    # extend to 2D
    A2D = kron(sparse(I, Npy + 2, Npy + 2), Bvux * A1D * B1D)
    kLe_i = [kLe[1]; kLe; kLe[end]]
    kRi_i = [kRi[1]; kRi; kRi[end]]
    ybc = kron(kLe_i, ybcl) + kron(kRi_i, ybcr)
    yAk_uy_lr = kron(sparse(I, Npy + 2, Npy + 2), Bvux * A1D * Btemp) * ybc

    # apply bc in y-direction
    B2D, Btemp, ybcl, ybcu =
        bc_general_stag(Npy + 2, Npy, 2, bck.low, bck.up, hy[1], hy[end])
    ybc = kron(Btemp * ybcl, kLo) + kron(Btemp * ybcu, kUp)
    yAk_uy_lu = A2D * ybc

    A2Dx = A2D * kron(B2D, sparse(I, Npx, Npx))

    # calculate average in y-direction; no boundary conditions
    diag1 = weight * ones(Npy + 1)
    A1Dy = spdiagm(Npy + 1, Npy + 2, 0 => diag1, 1 => diag1)
    A2Dy = kron(A1Dy, sparse(I, Nux_in, Nux_in))

    Ak_uy = A2Dy * A2Dx
    yAk_uy = A2Dy * (yAk_uy_lu + yAk_uy_lr)


    ## epsilon
    # in a similar way but with different boundary conditions
    B1D, Btemp, ybcl, ybcr =
        bc_general_stag(Npx + 2, Npx, 2, bce.left, bce.right, hx[1], hx[end])

    # then map to Nux_in points (like Iv_uy) with Bvux

    # extend to 2D
    A2D = kron(sparse(I, Npy + 2, Npy + 2), Bvux * A1D * B1D)
    eLe_i = [eLe[1]; eLe; eLe[end]]
    eRi_i = [eRi[1]; eRi; eRi[end]]
    ybc = kron(eLe_i, ybcl) + kron(eRi_i, ybcr)
    yAe_uy_lr = kron(sparse(I, Npy + 2, Npy + 2), Bvux * A1D * Btemp) * ybc

    # apply bc in y-direction
    B2D, Btemp, ybcl, ybcu =
        bc_general_stag(Npy + 2, Npy, 2, bce.low, bce.up, hy[1], hy[end])
    ybc = kron(Btemp * ybcl, eLo) + kron(Btemp * ybcu, eUp)
    yAe_uy_lu = A2D * ybc

    A2Dx = A2D * kron(B2D, sparse(I, Npx, Npx))

    # calculate average in y-direction; no boundary conditions
    diag1 = weight * ones(Npy + 1)
    A1Dy = spdiagm(Npy + 1, Npy + 2, 0 => diag1, 1 => diag1)
    A2Dy = kron(A1Dy, sparse(I, Nux_in, Nux_in))

    Ae_uy = A2Dy * A2Dx
    yAe_uy = A2Dy * (yAe_uy_lu + yAe_uy_lr)

    # so nu at uy is given by:
    # Cmu * (Ak_uy * k + yAk_uy).^2 / (Ae_uy * e + yAe_uy)


    ## nu to vx positions
    diag1 = weight * ones(Npy + 1)
    A1D = spdiagm(Npy + 1, Npy + 2, 0 => diag1, 1 => diag1)

    ## k
    # boundary conditions for k; mapping from Npy (k) points to Npy+2 points
    B1D, Btemp, ybcl, ybcu =
        bc_general_stag(Npy + 2, Npy, 2, bck.low, bck.up, hy[1], hy[end])

    # map to Nvy_in points (like Iu_vx) with Buvy

    # extend to 2D
    A2D = kron(Buvy * A1D * B1D, sparse(I, Npx + 2, Npx + 2))
    kLo_i = [kLo[1]; kLo; kLo[end]]
    kUp_i = [kUp[1]; kUp; kUp[end]]
    ybc = kron(ybcl, kLo_i) + kron(ybcu, kUp_i)
    yAk_vx_lu = kron(Buvy * A1D * Btemp, sparse(I, Npx + 2, Npx + 2)) * ybc

    # apply boundary conditions also in x-direction:
    B2D, Btemp, ybcl, ybcr =
        bc_general_stag(Npx + 2, Npx, 2, bck.left, bck.right, hx[1], hx[end])
    ybc = kron(kLe, Btemp * ybcl) + kron(kRi, Btemp * ybcr)
    yAk_vx_lr = A2D * ybc

    A2Dy = A2D * kron(sparse(I, Npy, Npy), B2D)

    # calculate average in x-direction; no boundary conditions
    diag1 = weight * ones(Npx + 1)
    A1Dx = spdiagm(Npx + 1, Npx + 2, 0 => diag1, 1 => diag1)
    A2Dx = kron(sparse(I, Nvy_in, Nvy_in), A1Dx)

    Ak_vx = A2Dx * A2Dy
    yAk_vx = A2Dx * (yAk_vx_lr + yAk_vx_lu)


    ## epsilon
    # in a similar way but with different boundary conditions
    B1D, Btemp, ybcl, ybcu =
        bc_general_stag(Npy + 2, Npy, 2, bce.low, bce.up, hy[1], hy[end])

    # extend to 2D
    A2D = kron(Buvy * A1D * B1D, sparse(I, Npx + 2, Npx + 2))
    eLo_i = [eLo[1]; eLo; eLo[end]]
    eUp_i = [eUp[1]; eUp; eUp[end]]
    ybc = kron(ybcl, eLo_i) + kron(ybcu, eUp_i)
    yAe_vx_lu = kron(Buvy * A1D * Btemp, sparse(I, Npx + 2, Npx + 2)) * ybc

    # apply boundary conditions also in x-direction:
    B2D, Btemp, ybcl, ybcr =
        bc_general_stag(Npx + 2, Npx, 2, bce.left, bce.right, hx[1], hx[end])
    ybc = kron(eLe, Btemp * ybcl) + kron(eRi, Btemp * ybcr)
    yAe_vx_lr = A2D * ybc

    A2Dy = A2D * kron(sparse(I, Npy, Npy), B2D)

    # calculate average in x-direction; no boundary conditions
    diag1 = weight * ones(Npx + 1)
    A1Dx = spdiagm(Npx + 1, Npx + 2, 0 => diag1, 1 => diag1)
    A2Dx = kron(sparse(I, Nvy_in, Nvy_in), A1Dx)

    Ae_vx = A2Dx * A2Dy
    yAe_vx = A2Dx * (yAe_vx_lr + yAe_vx_lu)


    # so nu at vx is given by:
    # Cmu * (Ak_vx * k + yAk_vx).^2 / (Ae_vx * e + yAe_vx)


    ## nu to vy positions
    A1D = sparse(I, Npy + 2, Npy + 2)

    ## k
    # boundary conditions; mapping from Npy (k) points to Npy+2 (vy faces) points
    B1D, Btemp, ybcl, ybcu =
        bc_general_stag(Npy + 2, Npy, 2, bck.low, bck.up, hy[1], hy[end])

    # then map back to Nvy_in+1 points (vy-faces) with Bkvy

    # extend to 2D
    Ak_vy = kron(Bkvy * A1D * B1D, sparse(I, Nvx_in, Nvx_in))
    ybc = kron(ybcl, kLo) + kron(ybcu, kUp)
    yAk_vy = kron(Bkvy * A1D * Btemp, sparse(I, Nvx_in, Nvx_in)) * ybc


    ## epsilon
    # in a similar way but with different boundary conditions
    B1D, Btemp, ybcl, ybcu =
        bc_general_stag(Npy + 2, Npy, 2, bce.low, bce.up, hy[1], hy[end])
    # extend to 2D
    Ae_vy = kron(Bkvy * A1D * B1D, sparse(I, Nvx_in, Nvx_in))
    ybc = kron(ybcl, eLo) + kron(ybcu, eUp)
    yAe_vy = kron(Bkvy * A1D * Btemp, sparse(I, Nvx_in, Nvx_in)) * ybc


    # so nu at vy is given by:
    # Cmu * (Ak_vy * k + yAk_vy).^2 / (Ae_vy * e + yAe_vy)

    setup
end
