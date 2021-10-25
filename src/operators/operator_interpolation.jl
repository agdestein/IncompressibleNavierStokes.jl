"""
Construct averaging operators.
"""
function operator_interpolation!(setup)
    @unpack bc = setup
    @unpack Nx, Ny = setup.grid

    # Number of interior points and boundary points
    @unpack Nux_in, Nux_b, Nux_t, Nuy_in, Nuy_b, Nuy_t = setup.grid
    @unpack Nvx_in, Nvx_b, Nvx_t, Nvy_in, Nvy_b, Nvy_t = setup.grid
    @unpack hx, hy, hxi, hyi = setup.grid
    @unpack Buvy, Bvux = setup.grid
    @unpack order4 = setup.discretization

    if order4
        β = setup.discretization.β
        @unpack hxi3, hyi3, hx3, hy3 = setup.grid
    end

    weight = 1 / 2

    mat_hx = spdiagm(Nx, Nx, hxi)
    mat_hy = spdiagm(Ny, Ny, hyi)

    # Periodic boundary conditions
    if bc.u.x[1] == :periodic && bc.u.x[2] == :periodic
        mat_hx2 = spdiagm(Nx + 2, Nx + 2, [hx[end]; hx; hx[1]])
    else
        mat_hx2 = spdiagm(Nx + 2, Nx + 2, [hx[1]; hx; hx[end]])
    end

    if bc.v.y[1] == :periodic && bc.v.y[2] == :periodic
        mat_hy2 = spdiagm(Ny + 2, Ny + 2, [hy[end]; hy; hy[1]])
    else
        mat_hy2 = spdiagm(Ny + 2, Ny + 2, [hy[1]; hy; hy[end]])
    end

    ## Interpolation operators, u-component
    if order4
        mat_hx3 = spdiagm(Nx, Nx, hxi3)
        mat_hy3 = spdiagm(Ny, Ny, hyi3)

        weight1 = 1 / 2 * β
        weight2 = 1 / 2 * (1 - β)

        # Periodic boundary conditions
        if bc.u.x[1] == :periodic && bc.u.x[2] == :periodic
            mat_hx2 = spdiagm(Nx + 4, Nx + 4, [hx[end-1]; hx[end]; hx; hx[1]; hx[2]])
            mat_hx4 = spdiagm(Nx + 4, Nx + 4, [hx3[end-1]; hx3[end]; hxi3; hx3[1]; hx3[2]])
        else
            mat_hx2 = spdiagm(Nx + 4, Nx + 4, [hx[2]; hx[1]; hx; hx[end]; hx[end-1]])
            mat_hx4 = spdiagm(
                Nx + 4,
                Nx + 4,
                [
                    hx[1] + hx[2] + hx[3]
                    2 * hx[1] + hx[2]
                    hxi3
                    2 * hx[end] + hx[end-1]
                    hx[end] + hx[end-1] + hx[end-2]
                ],
            )
        end

        if bc.v.y[1] == :periodic && bc.v.y[2] == :periodic
            mat_hy2 = spdiagm(Ny + 4, Ny + 4, [hy[end-1]; hy[end]; hy; hy[1]; hy[2]])
            mat_hy4 = spdiagm(Ny + 4, Ny + 4, [hy3[end-1]; hy3[end]; hyi3; hy3[1]; hy3[2]])
        else
            mat_hy2 = spdiagm(Ny + 4, Ny + 4, [hy[2]; hy[1]; hy; hy[end]; hy[end-1]])
            mat_hy4 = spdiagm(
                Ny + 4,
                Ny + 4,
                [
                    hy[1] + hy[2] + hy[3]
                    2 * hy[1] + hy[2]
                    hyi3
                    2 * hy[end] + hy[end-1]
                    hy[end] + hy[end-1] + hy[end-2]
                ],
            )
        end

        ## Iu_ux
        diag1 = fill(weight1, Nux_t + 1)
        diag2 = fill(weight2, Nux_t + 1)
        I1D = spdiagm(Nux_t - 1, Nux_t + 2, 0 => diag2, 1 => diag1, 2 => diag1, 3 => diag2)
        # Boundary conditions
        Iu_ux_bc = bc_int2(
            Nux_t + 2,
            Nux_in,
            Nux_t + 2 - Nux_in,
            bc.u.x[1],
            bc.u.x[2],
            hx[1],
            hx[end],
        )

        # Extend to 2D
        Iu_ux = kron(mat_hy, I1D * Iu_ux_bc.B1D)
        Iu_ux_bc = (; Iu_ux_bc..., Bbc = kron(mat_hy, I1D * Iu_ux_bc.Btemp))

        ## Iu_ux3
        diag1 = fill(weight1, Nux_t + 4)
        diag2 = fill(weight2, Nux_t + 4)
        I1D3 =
            spdiagm(Nux_in + 3, Nux_t + 4, 0 => diag2, 1 => diag1, 2 => diag1, 3 => diag2)
        # Boundary conditions
        Iu_ux_bc3 = bc_int3(
            Nux_t + 4,
            Nux_in,
            Nux_t + 4 - Nux_in,
            bc.u.x[1],
            bc.u.x[2],
            hx[1],
            hx[end],
        )
        # Extend to 2D
        Iu_ux3 = kron(mat_hy3, I1D3 * Iu_ux_bc3.B1D)
        Iu_ux_bc3 = (; Iu_ux_bc3..., Bbc = kron(mat_hy3, I1D3 * Iu_ux_bc3.Btemp))

        ## Iv_uy
        diag1 = fill(weight1, Nvx_t)
        diag2 = fill(weight2, Nvx_t)
        I1D = spdiagm(Nvx_t - 1, Nvx_t + 2, 0 => diag2, 1 => diag1, 2 => diag1, 3 => diag2)
        # Restrict to u-points
        # The restriction is essentially 1D so it can be directly applied to I1D
        I1D = Bvux * I1D * mat_hx2
        I2D = kron(sparse(I, Nuy_t - 1, Nuy_t - 1), I1D)
        # Boundary conditions low/up
        Nb = Nuy_in + 1 - Nvy_in
        Iv_uy_bc_lu = bc_general(Nuy_in + 1, Nvy_in, Nb, bc.v.y[1], bc.v.y[2], hy[1], hy[end])
        Iv_uy_bc_lu =
            (; Iv_uy_bc_lu..., B2D = kron(Iv_uy_bc_lu.B1D, sparse(I, Nvx_in, Nvx_in)))
        Iv_uy_bc_lu =
            (; Iv_uy_bc_lu..., Bbc = kron(Iv_uy_bc_lu.Btemp, sparse(I, Nvx_in, Nvx_in)))
        # Boundary conditions left/right
        Iv_uy_bc_lr = bc_int_mixed_stag2(
            Nvx_t + 2,
            Nvx_in,
            Nvx_t + 2 - Nvx_in,
            bc.v.x[1],
            bc.v.x[2],
            hx[1],
            hx[end],
        )
        # Take I2D into left/right operators for convenience
        Iv_uy_bc_lr = (;
            Iv_uy_bc_lr...,
            B2D = I2D * kron(sparse(I, Nuy_t - 1, Nuy_t - 1), Iv_uy_bc_lr.B1D),
        )
        Iv_uy_bc_lr = (;
            Iv_uy_bc_lr...,
            Bbc = I2D * kron(sparse(I, Nuy_t - 1, Nuy_t - 1), Iv_uy_bc_lr.Btemp),
        )
        # Resulting operator:
        Iv_uy = Iv_uy_bc_lr.B2D * Iv_uy_bc_lu.B2D

        ## Iv_uy3
        diag1 = fill(weight1, Nvy_t)
        diag2 = fill(weight2, Nvy_t)
        I1D = spdiagm(Nvx_t - 1, Nvx_t + 2, 0 => diag2, 1 => diag1, 2 => diag1, 3 => diag2)
        # Restrict to u-points
        # The restriction is essentially 1D so it can be directly applied to I1D
        I1D = Bvux * I1D * mat_hx4
        I2D = kron(sparse(I, Nuy_t + 1, Nuy_t + 1), I1D)
        # Boundary conditions low/up
        Nb = Nuy_in + 3 - Nvy_in
        Iv_uy_bc_lu3 =
            bc_int_mixed2(Nuy_in + 3, Nvy_in, Nb, bc.v.y[1], bc.v.y[2], hy[1], hy[end])
        Iv_uy_bc_lu3 =
            (; Iv_uy_bc_lu3..., B2D = kron(Iv_uy_bc_lu3.B1D, sparse(I, Nvx_in, Nvx_in)))
        Iv_uy_bc_lu3 =
            (; Iv_uy_bc_lu3..., Bbc = kron(Iv_uy_bc_lu3.Btemp, sparse(I, Nvx_in, Nvx_in)))
        # Boundary conditions left/right
        Iv_uy_bc_lr3 = bc_int_mixed_stag3(
            Nvx_t + 2,
            Nvx_in,
            Nvx_t + 2 - Nvx_in,
            bc.v.x[1],
            bc.v.x[2],
            hx[1],
            hx[end],
        )
        # Take I2D into left/right operators for convenience
        Iv_uy_bc_lr3 = (;
            Iv_uy_bc_lr3...,
            B2D = I2D * kron(sparse(I, Nuy_t + 1, Nuy_t + 1), Iv_uy_bc_lr3.B1D),
        )
        Iv_uy_bc_lr3 = (;
            Iv_uy_bc_lr3...,
            Bbc = I2D * kron(sparse(I, Nuy_t + 1, Nuy_t + 1), Iv_uy_bc_lr3.Btemp),
        )
        # Resulting operator:
        Iv_uy3 = Iv_uy_bc_lr3.B2D * Iv_uy_bc_lu3.B2D

        ## Iu_vx
        diag1 = fill(weight1, Nuy_t)
        diag2 = fill(weight2, Nuy_t)
        I1D = spdiagm(Nuy_t - 1, Nuy_t + 2, 0 => diag2, 1 => diag1, 2 => diag1, 3 => diag2)
        # Restrict to v-points
        I1D = Buvy * I1D * mat_hy2
        I2D = kron(I1D, sparse(I, Nvx_t - 1, Nvx_t - 1))
        # Boundary conditions low/up
        Iu_vx_bc_lu = bc_int_mixed_stag2(
            Nuy_t + 2,
            Nuy_in,
            Nuy_t + 2 - Nuy_in,
            bc.u.y[1],
            bc.u.y[2],
            hy[1],
            hy[end],
        )
        Iu_vx_bc_lu = (;
            Iu_vx_bc_lu...,
            B2D = I2D * kron(Iu_vx_bc_lu.B1D, sparse(I, Nvx_t - 1, Nvx_t - 1)),
        )
        Iu_vx_bc_lu = (;
            Iu_vx_bc_lu...,
            Bbc = I2D * kron(Iu_vx_bc_lu.Btemp, sparse(I, Nvx_t - 1, Nvx_t - 1)),
        )

        # Boundary conditions left/right
        Nb = Nvx_in + 1 - Nux_in
        Iu_vx_bc_lr =
            bc_general(Nvx_in + 1, Nux_in, Nb, bc.u.x[1], bc.u.x[2], hx[1], hx[end])

        Iu_vx_bc_lr =
            (; Iu_vx_bc_lr..., B2D = kron(sparse(I, Nuy_in, Nuy_in), Iu_vx_bc_lr.B1D))
        Iu_vx_bc_lr =
            (; Iu_vx_bc_lr..., Bbc = kron(sparse(I, Nuy_in, Nuy_in), Iu_vx_bc_lr.Btemp))

        # Resulting operator:
        Iu_vx = Iu_vx_bc_lu.B2D * Iu_vx_bc_lr.B2D

        ## Iu_vx3
        diag1 = fill(weight1, Nuy_t)
        diag2 = fill(weight2, Nuy_t)
        I1D = spdiagm(Nuy_t - 1, Nuy_t + 2, 0 => diag2, 1 => diag1, 2 => diag1, 3 => diag2)
        # Restrict to v-points
        I1D = Buvy * I1D * mat_hy4
        I2D = kron(I1D, sparse(I, Nvx_t + 1, Nvx_t + 1))
        # Boundary conditions low/up
        Iu_vx_bc_lu3 = bc_int_mixed_stag3(
            Nuy_t + 2,
            Nuy_in,
            Nuy_t + 2 - Nuy_in,
            bc.u.y[1],
            bc.u.y[2],
            hy[1],
            hy[end],
        )
        Iu_vx_bc_lu3 = (;
            Iu_vx_bc_lu3...,
            B2D = I2D * kron(Iu_vx_bc_lu3.B1D, sparse(I, Nvx_t + 1, Nvx_t + 1)),
        )
        Iu_vx_bc_lu3 = (;
            Iu_vx_bc_lu3...,
            Bbc = I2D * kron(Iu_vx_bc_lu3.Btemp, sparse(I, Nvx_t + 1, Nvx_t + 1)),
        )

        # Boundary conditions left/right
        Nb = Nvx_in + 3 - Nux_in
        Iu_vx_bc_lr3 =
            bc_int_mixed2(Nvx_in + 3, Nux_in, Nb, bc.u.x[1], bc.u.x[2], hx[1], hx[end])

        Iu_vx_bc_lr3 =
            (; Iu_vx_bc_lr3..., B2D = kron(sparse(I, Nuy_in, Nuy_in), Iu_vx_bc_lr3.B1D))
        Iu_vx_bc_lr3 =
            (; Iu_vx_bc_lr3..., Bbc = kron(sparse(I, Nuy_in, Nuy_in), Iu_vx_bc_lr3.Btemp))

        # Resulting operator:
        Iu_vx3 = Iu_vx_bc_lu3.B2D * Iu_vx_bc_lr3.B2D

        ## Iv_vy
        diag1 = fill(weight1, Nvy_t + 1)
        diag2 = fill(weight2, Nvy_t + 1)
        I1D = spdiagm(Nvy_t - 1, Nvy_t + 2, 0 => diag2, 1 => diag1, 2 => diag1, 3 => diag2)

        # Boundary conditions
        Iv_vy_bc = bc_int2(
            Nvy_t + 2,
            Nvy_in,
            Nvy_t + 2 - Nvy_in,
            bc.v.y[1],
            bc.v.y[2],
            hy[1],
            hy[end],
        )

        # Extend to 2D
        Iv_vy = kron(I1D * Iv_vy_bc.B1D, mat_hx)
        Iv_vy_bc = (; Iv_vy_bc..., Bbc = kron(I1D * Iv_vy_bc.Btemp, mat_hx))

        ## Iv_vy3
        diag1 = fill(weight1, Nvx_t + 4)
        diag2 = fill(weight2, Nvx_t + 4)
        I1D3 =
            spdiagm(Nvy_in + 3, Nvy_t + 4, 0 => diag2, 1 => diag1, 2 => diag1, 3 => diag2)
        # Boundary conditions
        Iv_vy_bc3 = bc_int3(
            Nvy_t + 4,
            Nvy_in,
            Nvy_t + 4 - Nvy_in,
            bc.v.y[1],
            bc.v.y[2],
            hy[1],
            hy[end],
        )
        # Extend to 2D
        Iv_vy3 = kron(I1D3 * Iv_vy_bc3.B1D, mat_hx3)
        Iv_vy_bc3 = (; Iv_vy_bc3..., Bbc = kron(I1D3 * Iv_vy_bc3.Btemp, mat_hx3))
    else
        ## Iu_ux
        diag1 = fill(weight, Nux_t - 1)
        I1D = spdiagm(Nux_t - 1, Nux_t, 0 => diag1, 1 => diag1)
        # Boundary conditions
        Iu_ux_bc = bc_general(Nux_t, Nux_in, Nux_b, bc.u.x[1], bc.u.x[2], hx[1], hx[end])

        # Extend to 2D
        Iu_ux = kron(mat_hy, I1D * Iu_ux_bc.B1D)
        Iu_ux_bc = (; Iu_ux_bc..., Bbc = kron(mat_hy, I1D * Iu_ux_bc.Btemp))


        ## Iv_uy
        diag1 = fill(weight, Nvx_t - 1)
        I1D = spdiagm(Nvx_t - 1, Nvx_t, 0 => diag1, 1 => diag1)
        # The restriction is essentially 1D so it can be directly applied to I1D
        I1D = Bvux * I1D * mat_hx2
        I2D = kron(sparse(I, Nuy_t - 1, Nuy_t - 1), I1D)

        # Boundary conditions low/up
        Nb = Nuy_in + 1 - Nvy_in
        Iv_uy_bc_lu = bc_general(Nuy_in + 1, Nvy_in, Nb, bc.v.y[1], bc.v.y[2], hy[1], hy[end])
        Iv_uy_bc_lu =
            (; Iv_uy_bc_lu..., B2D = kron(Iv_uy_bc_lu.B1D, sparse(I, Nvx_in, Nvx_in)))
        Iv_uy_bc_lu =
            (; Iv_uy_bc_lu..., Bbc = kron(Iv_uy_bc_lu.Btemp, sparse(I, Nvx_in, Nvx_in)))

        # Boundary conditions left/right
        Iv_uy_bc_lr =
            bc_general_stag(Nvx_t, Nvx_in, Nvx_b, bc.v.x[1], bc.v.x[2], hx[1], hx[end])
        # Take I2D into left/right operators for convenience
        Iv_uy_bc_lr = (;
            Iv_uy_bc_lr...,
            B2D = I2D * kron(sparse(I, Nuy_t - 1, Nuy_t - 1), Iv_uy_bc_lr.B1D),
        )
        Iv_uy_bc_lr = (;
            Iv_uy_bc_lr...,
            Bbc = I2D * kron(sparse(I, Nuy_t - 1, Nuy_t - 1), Iv_uy_bc_lr.Btemp),
        )

        # Resulting operator:
        Iv_uy = Iv_uy_bc_lr.B2D * Iv_uy_bc_lu.B2D

        ## Interpolation operators, v-component

        ## Iu_vx
        diag1 = fill(weight, Nuy_t - 1)
        I1D = spdiagm(Nuy_t - 1, Nuy_t, 0 => diag1, 1 => diag1)
        I1D = Buvy * I1D * mat_hy2
        I2D = kron(I1D, sparse(I, Nvx_t - 1, Nvx_t - 1))

        # Boundary conditions low/up
        Iu_vx_bc_lu =
            bc_general_stag(Nuy_t, Nuy_in, Nuy_b, bc.u.y[1], bc.u.y[2], hy[1], hy[end])
        Iu_vx_bc_lu = (;
            Iu_vx_bc_lu...,
            B2D = I2D * kron(Iu_vx_bc_lu.B1D, sparse(I, Nvx_t - 1, Nvx_t - 1)),
        )
        Iu_vx_bc_lu = (;
            Iu_vx_bc_lu...,
            Bbc = I2D * kron(Iu_vx_bc_lu.Btemp, sparse(I, Nvx_t - 1, Nvx_t - 1)),
        )

        # Boundary conditions left/right
        Nb = Nvx_in + 1 - Nux_in
        Iu_vx_bc_lr =
            bc_general(Nvx_in + 1, Nux_in, Nb, bc.u.x[1], bc.u.x[2], hx[1], hx[end])

        Iu_vx_bc_lr =
            (; Iu_vx_bc_lr..., B2D = kron(sparse(I, Nuy_in, Nuy_in), Iu_vx_bc_lr.B1D))
        Iu_vx_bc_lr =
            (; Iu_vx_bc_lr..., Bbc = kron(sparse(I, Nuy_in, Nuy_in), Iu_vx_bc_lr.Btemp))

        # Resulting operator:
        Iu_vx = Iu_vx_bc_lu.B2D * Iu_vx_bc_lr.B2D

        ## Iv_vy
        diag1 = fill(weight, Nvy_t - 1)
        I1D = spdiagm(Nvy_t - 1, Nvy_t, 0 => diag1, 1 => diag1)
        # Boundary conditions
        Iv_vy_bc = bc_general(Nvy_t, Nvy_in, Nvy_b, bc.v.y[1], bc.v.y[2], hy[1], hy[end])

        # Extend to 2D
        Iv_vy = kron(I1D * Iv_vy_bc.B1D, mat_hx)
        Iv_vy_bc = (; Iv_vy_bc..., Bbc = kron(I1D * Iv_vy_bc.Btemp, mat_hx))
    end

    ## Store in setup structure
    @pack! setup.discretization = Iu_ux, Iv_uy, Iu_vx, Iv_vy
    @pack! setup.discretization = Iu_ux_bc, Iv_vy_bc
    @pack! setup.discretization = Iv_uy_bc_lr, Iv_uy_bc_lu, Iu_vx_bc_lr, Iu_vx_bc_lu
    if order4
        @pack! setup.discretization = Iu_ux3, Iv_uy3, Iu_vx3, Iv_vy3
        @pack! setup.discretization = Iu_ux_bc3, Iv_uy_bc_lr3, Iv_uy_bc_lu3, Iu_vx_bc_lr3, Iu_vx_bc_lu3, Iv_vy_bc3
    end

    setup
end
