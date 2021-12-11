"""
Construct averaging operators.
"""
function operator_interpolation!(setup)
    @unpack bc = setup
    @unpack Nx, Ny, Nz = setup.grid
    @unpack Nux_in, Nux_b, Nux_t, Nuy_in, Nuy_b, Nuy_t, Nuz_in, Nuz_b, Nuz_t  = setup.grid
    @unpack Nvx_in, Nvx_b, Nvx_t, Nvy_in, Nvy_b, Nvy_t, Nvz_in, Nvz_b, Nvz_t  = setup.grid
    @unpack Nwx_in, Nwx_b, Nwx_t, Nwy_in, Nwy_b, Nwy_t, Nwz_in, Nwz_b, Nwz_t  = setup.grid
    @unpack hx, hy, hz, hxi, hyi, hzi = setup.grid
    @unpack Buvy, Buwz, Bvux, Bvwz, Bwux, Bwvy = setup.grid

    weight = 1 / 2

    mat_hx = spdiagm(Nx, Nx, hxi)
    mat_hy = spdiagm(Ny, Ny, hyi)
    mat_hz = spdiagm(Nz, Nz, hzi)

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

    if bc.w.z[1] == :periodic && bc.w.z[2] == :periodic
        mat_hz2 = spdiagm(Nz + 2, Nz + 2, [hz[end]; hz; hz[1]])
    else
        mat_hz2 = spdiagm(Nz + 2, Nz + 2, [hz[1]; hz; hz[end]])
    end


    ## Interpolation operators, u-component

    ## Iu_ux
    diag1 = fill(weight, Nux_t - 1)
    I1D = spdiagm(Nux_t - 1, Nux_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Iu_ux_bc = bc_general(Nux_t, Nux_in, Nux_b, bc.u.x[1], bc.u.x[2], hx[1], hx[end])

    # Extend to 3D
    Iu_ux = mat_hz ⊗ mat_hy ⊗ (I1D * Iu_ux_bc.B1D)
    Iu_ux_bc = (; Iu_ux_bc..., Bbc = mat_hz ⊗ mat_hy ⊗ (I1D * Iu_ux_bc.Btemp))


    ## Iv_uy
    diag1 = fill(weight, Nvx_t - 1)
    I1D = spdiagm(Nvx_t - 1, Nvx_t, 0 => diag1, 1 => diag1)

    # The restriction is essentially 1D so it can be directly applied to I1D
    I1D = Bvux * I1D * mat_hx2
    I3D = kron(mat_hz, kron(I(Nuy_t - 1), I1D))

    # Boundary conditions low/up
    Nb = Nuy_in + 1 - Nvy_in
    Iv_uy_bc_lu = bc_general(Nuy_in + 1, Nvy_in, Nb, bc.v.y[1], bc.v.y[2], hy[1], hy[end])
    Iv_uy_bc_lu = (; Iv_uy_bc_lu..., B3D = I(Nz) ⊗ Iv_uy_bc_lu.B1D ⊗ I(Nvx_in))
    Iv_uy_bc_lu = (; Iv_uy_bc_lu..., Bbc = I(Nz) ⊗ Iv_uy_bc_lu.Btemp ⊗ I(Nvx_in))

    # Boundary conditions left/right
    Iv_uy_bc_lr =
        bc_general_stag(Nvx_t, Nvx_in, Nvx_b, bc.v.x[1], bc.v.x[2], hx[1], hx[end])

    # Take I2D into left/right operators for convenience
    Iv_uy_bc_lr = (; Iv_uy_bc_lr..., B3D = I3D * (I(Nz) ⊗ I(Nuy_t - 1) ⊗ Iv_uy_bc_lr.B1D))
    Iv_uy_bc_lr = (; Iv_uy_bc_lr..., Bbc = I3D * (I(Nz) ⊗ I(Nuy_t - 1) ⊗ Iv_uy_bc_lr.Btemp))

    # Resulting operator:
    Iv_uy = Iv_uy_bc_lr.B3D * Iv_uy_bc_lu.B3D


    ## Iw_uz
    diag1 = fill(weight, Nwx_t - 1)
    I1D = spdiagm(Nwx_t - 1, Nwx_t, 0 => diag1, 1 => diag1)
    I1D = Bwux * I1D * mat_hx2
    I3D = I(Nz + 1) ⊗ mat_hy ⊗ I1D

    # Boundary conditions left/right
    Iw_uz_bc_lr =
        bc_general_stag(Nwx_t, Nwx_in, Nwx_b, bc.w.x[1], bc.w.x[2], hx[1], hx[end])

    # Take I3D into left/right operators for convenience
    Iw_uz_bc_lr = (; Iw_uz_bc_lr..., B3D = I3D * (I(Nz + 1) ⊗ I(Ny) ⊗ Iw_uz_bc_lr.B1D))
    Iw_uz_bc_lr = (; Iw_uz_bc_lr..., Bbc = I3D * (I(Nz + 1) ⊗ I(Ny) ⊗ Iw_uz_bc_lr.Btemp))

    # Boundary conditions back/front
    Nb = Nuz_in + 1 - Nwz_in
    Iw_uz_bc_bf = bc_general(Nuz_in + 1, Nwz_in, Nb, bc.w.z[1], bc.w.z[2], hz[1], hz[end])
    Iw_uz_bc_bf = (; Iw_uz_bc_bf..., B3D = Iw_uz_bc_bf.B1D ⊗ I(Ny) ⊗ I(Nx))
    Iw_uz_bc_bf = (; Iw_uz_bc_bf..., Bbc = Iw_uz_bc_bf.Btemp ⊗ I(Ny) ⊗ I(Nx))

    # Resulting operator:
    Iw_uz = Iw_uz_bc_lr.B3D * Iw_uz_bc_bf.B3D



    ## Interpolation operators, v-component

    ## Iu_vx
    diag1 = fill(weight, Nuy_t - 1)
    I1D = spdiagm(Nuy_t - 1, Nuy_t, 0 => diag1, 1 => diag1)
    I1D = Buvy * I1D * mat_hy2
    I3D = mat_hz ⊗ I1D ⊗ I(Nx + 1)

    # Boundary conditions low/up
    Iu_vx_bc_lu =
        bc_general_stag(Nuy_t, Nuy_in, Nuy_b, bc.u.y[1], bc.u.y[2], hy[1], hy[end])
    Iu_vx_bc_lu = (; Iu_vx_bc_lu..., B3D = I3D * (I(Nz) ⊗ Iu_vx_bc_lu.B1D ⊗ I(Nx + 1)))
    Iu_vx_bc_lu = (; Iu_vx_bc_lu..., Bbc = I3D * (I(Nz) ⊗ Iu_vx_bc_lu.Btemp ⊗ I(Nx + 1)))

    # Boundary conditions left/right
    Nb = Nvx_in + 1 - Nux_in
    Iu_vx_bc_lr =
        bc_general(Nvx_in + 1, Nux_in, Nb, bc.u.x[1], bc.u.x[2], hx[1], hx[end])
    Iu_vx_bc_lr = (; Iu_vx_bc_lr..., B3D = I(Nz) ⊗ I(Ny) ⊗ Iu_vx_bc_lr.B1D)
    Iu_vx_bc_lr = (; Iu_vx_bc_lr..., Bbc = I(Nz) ⊗ I(Ny) ⊗ Iu_vx_bc_lr.Btemp)

    # Resulting operator:
    Iu_vx = Iu_vx_bc_lu.B3D * Iu_vx_bc_lr.B3D

    ## Iv_vy
    diag1 = fill(weight, Nvy_t - 1)
    I1D = spdiagm(Nvy_t - 1, Nvy_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Iv_vy_bc = bc_general(Nvy_t, Nvy_in, Nvy_b, bc.v.y[1], bc.v.y[2], hy[1], hy[end])

    # Extend to 3D
    Iv_vy = mat_hz ⊗ (I1D * Iv_vy_bc.B1D) ⊗ mat_hx
    Iv_vy_bc = (; Iv_vy_bc..., Bbc = mat_hz ⊗ (I1D * Iv_vy_bc.Btemp) ⊗ mat_hx)

    ## Iw_vz
    diag1 = fill(weight, Nwy_t - 1)
    I1D = spdiagm(Nwy_t - 1, Nwy_t, 0 => diag1, 1 => diag1)
    I1D = Bwvy * I1D * mat_hy2
    I3D = I(Nz + 1) ⊗ I1D ⊗ mat_hx

    # Boundary conditions low/up
    Iw_vz_bc_lu =
        bc_general_stag(Nwy_t, Nwy_in, Nwy_b, bc.w.y[1], bc.w.y[2], hy[1], hy[end])
    Iw_vz_bc_lu = (; Iw_vz_bc_lu..., B3D = I3D * (I(Nz + 1) ⊗ Iw_vz_bc_lu.B1D ⊗ I(Nx)))
    Iw_vz_bc_lu = (; Iw_vz_bc_lu..., Bbc = I3D * (I(Nz + 1) ⊗ Iw_vz_bc_lu.Btemp ⊗ I(Nx)))

    # Boundary conditions left/right
    Nb = Nvz_in + 1 - Nwz_in
    Iw_vz_bc_bf =
        bc_general(Nvz_in + 1, Nwz_in, Nb, bc.w.z[1], bc.w.z[2], hz[1], hz[end])
    Iw_vz_bc_bf = (; Iw_vz_bc_bf..., B3D = Iw_vz_bc_bf.B1D ⊗ I(Ny) ⊗ I(Nx))
    Iw_vz_bc_bf = (; Iw_vz_bc_bf..., Bbc = Iw_vz_bc_bf.Btemp ⊗ I(Ny) ⊗ I(Nx))

    # Resulting operator:
    Iw_vz = Iw_vz_bc_lu.B3D * Iw_vz_bc_bf.B3D



    ## Interpolation operators, w-component

    ## Iu_wx
    diag1 = fill(weight, Nuz_t - 1)
    I1D = spdiagm(Nuz_t - 1, Nuz_t, 0 => diag1, 1 => diag1)
    I1D = Buwz * I1D * mat_hz2
    I3D = I1D ⊗ mat_hy ⊗ I(Nx + 1)

    # Boundary conditions back/front
    Iu_wx_bc_bf =
        bc_general_stag(Nuz_t, Nuz_in, Nuz_b, bc.u.z[1], bc.u.z[2], hz[1], hz[end])
    Iu_wx_bc_bf = (; Iu_wx_bc_bf..., B3D = I3D * (Iu_wx_bc_bf.B1D ⊗ I(Ny) ⊗ I(Nx + 1)))
    Iu_wx_bc_bf = (; Iu_wx_bc_bf..., Bbc = I3D * (Iu_wx_bc_bf.Btemp ⊗ I(Ny) ⊗ I(Nx + 1)))

    # Boundary conditions left/right
    Nb = Nwx_in + 1 - Nux_in
    Iu_wx_bc_lr =
        bc_general(Nwx_in + 1, Nux_in, Nb, bc.u.x[1], bc.u.x[2], hx[1], hx[end])
    Iu_wx_bc_lr = (; Iu_wx_bc_lr..., B3D = I(Nz) ⊗ I(Ny) ⊗ Iu_wx_bc_lr.B1D)
    Iu_wx_bc_lr = (; Iu_wx_bc_lr..., Bbc = I(Nz) ⊗ I(Ny) ⊗ Iu_wx_bc_lr.Btemp)

    # Resulting operator:
    Iu_wx = Iu_wx_bc_bf.B3D * Iu_wx_bc_lr.B3D

    ## Iv_wy
    diag1 = fill(weight, Nwz_t - 1)
    I1D = spdiagm(Nwz_t - 1, Nwz_t, 0 => diag1, 1 => diag1)
    I1D = Bvwz * I1D * mat_hz2
    I3D = I1D ⊗ I(Ny + 1) ⊗ mat_hx

    # Boundary conditions back/front
    Iv_wy_bc_bf =
        bc_general_stag(Nvz_t, Nvz_in, Nvz_b, bc.v.y[1], bc.v.y[2], hz[1], hz[end])
    Iv_wy_bc_bf = (; Iv_wy_bc_bf..., B3D = I3D * (Iv_wy_bc_bf.B1D ⊗ I(Ny + 1) ⊗  I(Nx)))
    Iv_wy_bc_bf = (; Iv_wy_bc_bf..., Bbc = I3D * (Iv_wy_bc_bf.Btemp ⊗ I(Ny + 1) ⊗  I(Nx)))

    # Boundary conditions low/up
    Nb = Nwy_in + 1 - Nvy_in
    Iv_wy_bc_lu =
        bc_general(Nwy_in + 1, Nvy_in, Nb, bc.v.z[1], bc.v.z[2], hy[1], hy[end])
    Iv_wy_bc_lu = (; Iv_wy_bc_lu..., B3D = I(Nz) ⊗ Iv_wy_bc_lu.B1D ⊗ I(Nx))
    Iv_wy_bc_lu = (; Iv_wy_bc_lu..., Bbc = I(Nz) ⊗ Iv_wy_bc_lu.Btemp ⊗ I(Nx))

    # Resulting operator:
    Iv_wy = Iv_wy_bc_bf.B3D * Iv_wy_bc_lu.B3D

    ## Iw_wz
    diag1 = fill(weight, Nwz_t - 1)
    I1D = spdiagm(Nwz_t - 1, Nwz_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Iw_wz_bc = bc_general(Nwz_t, Nwz_in, Nwz_b, bc.w.z[1], bc.w.z[2], hz[1], hz[end])

    # Extend to 3D
    Iw_wz = (I1D * Iw_wz_bc.B1D) ⊗ mat_hy ⊗ mat_hx
    Iw_wz_bc = (; Iw_wz_bc..., Bbc = (I1D * Iw_wz_bc.Btemp) ⊗ mat_hy ⊗ mat_hx)


    ## Store in setup structure
    @pack! setup.discretization = Iu_ux, Iv_uy, Iw_uz
    @pack! setup.discretization = Iu_vx, Iv_vy, Iw_vz
    @pack! setup.discretization = Iu_wx, Iv_wy, Iw_wz
    @pack! setup.discretization = Iu_ux_bc, Iv_vy_bc, Iw_wz_bc
    @pack! setup.discretization = Iv_uy_bc_lr, Iv_uy_bc_lu, Iu_vx_bc_lr, Iu_vx_bc_lu
    @pack! setup.discretization = Iw_uz_bc_lr, Iw_uz_bc_bf, Iw_vz_bc_lu, Iw_vz_bc_bf
    @pack! setup.discretization = Iu_wx_bc_lr, Iu_wx_bc_bf, Iv_wy_bc_lu, Iv_wy_bc_bf

    setup
end
