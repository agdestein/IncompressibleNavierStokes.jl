"""
    operator_convection_diffusion!(setup)

Construct convection and diffusion operators.
"""
function operator_convection_diffusion! end

# 2D version
function operator_convection_diffusion!(setup::Setup{T,2}) where {T}
    (; grid, operators, bc, viscosity_model) = setup
    (; Nx, Ny) = grid
    (; Nux_in, Nux_b, Nux_t, Nuy_in, Nuy_b, Nuy_t) = grid
    (; Nvx_in, Nvx_b, Nvx_t, Nvy_in, Nvy_b, Nvy_t) = grid
    (; hx, hy, hxi, hyi, hxd, hyd) = grid
    (; gxi, gyi, gxd, gyd) = grid
    (; Buvy, Bvux) = grid
    (; order4, α) = grid
    (; Re) = viscosity_model

    if order4
        (; hxi3, hyi3, gxi3, gyi3, hxd13, hxd3, hyd13, hyd3) = grid
        (; gxd13, gxd3, gyd13, gyd3) = grid
        (; Ωux, Ωuy, Ωvx, Ωvy) = grid
        (; Ωux1, Ωux3, Ωuy1, Ωuy3, Ωvx1, Ωvx3, Ωvy1, Ωvy3) = grid
    end


    ## Convection (differencing) operator Cu

    # Calculates difference from pressure points to velocity points
    diag1 = ones(Nux_t - 2)
    D1D = spdiagm(Nux_t - 2, Nux_t - 1, 0 => -diag1, 1 => diag1)
    Cux = I(Nuy_in) ⊗ D1D
    if !order4
        Dux = Diagonal(hyi) ⊗ D1D
    end

    # Calculates difference from corner points to velocity points
    diag1 = ones(Nuy_t - 2)
    D1D = spdiagm(Nuy_t - 2, Nuy_t - 1, 0 => -diag1, 1 => diag1)
    Cuy = D1D ⊗ I(Nux_in)
    if !order4
        Duy = D1D ⊗ Diagonal(gxi)
    end

    # Cu = [Cux Cuy]
    # Du = [Dux Duy]

    ## Convection (differencing) operator Cv

    # Calculates difference from pressure points to velocity points
    diag1 = ones(Nvx_t - 2)
    D1D = spdiagm(Nvx_t - 2, Nvx_t - 1, 0 => -diag1, 1 => diag1)
    Cvx = I(Nvy_in) ⊗ D1D
    if !order4
        Dvx = Diagonal(gyi) ⊗ D1D
    end

    # Calculates difference from corner points to velocity points
    diag1 = ones(Nvy_t - 2)
    D1D = spdiagm(Nvy_t - 2, Nvy_t - 1, 0 => -diag1, 1 => diag1)
    Cvy = D1D ⊗ I(Nvx_in)
    if !order4
        Dvy = D1D ⊗ Diagonal(hxi)
    end

    # Cv = [Cvx Cvy]
    # Dv = [Dvx Dvy]

    if order4
        ## Fourth order operators
        ## Convection (differencing) operator Cu

        # Calculates difference from pressure points to velocity points
        diag1 = ones(Nux_t - 2)
        D1D = spdiagm(Nux_t - 2, Nux_t + 1, 1 => -diag1, 2 => diag1)
        Dux = Diagonal(hyi) ⊗ D1D

        # The "second order" Cux is unchanged
        # The "second order" Dux changes, because we also use the "second
        # Order" flux at "fourth order" ghost points (Dux should have the same
        # Size as Dux3)

        # Calculates difference from pressure points to velocity points
        diag1 = ones(Nux_t - 2)
        D1D3 = spdiagm(Nux_t - 2, Nux_t + 1, 0 => -diag1, 3 => diag1)
        Cux3 = I(Ny) ⊗ D1D3
        Dux3 = Diagonal(hyi3) ⊗ D1D3

        # Calculates difference from corner points to velocity points
        diag1 = ones(Nuy_t - 2)
        D1D = spdiagm(Nuy_t - 2, Nuy_t + 1, 1 => -diag1, 2 => diag1)
        Duy = D1D ⊗ Diagonal(gxi)

        # Calculates difference from corner points to velocity points
        diag1 = ones(Nuy_t - 2)
        D1D3 = spdiagm(Nuy_t - 2, Nuy_t + 1, 0 => -diag1, 3 => diag1)

        # Uncomment for new BC (functions/new)
        if bc.u.y[1] == :dirichlet
            D1D3[1, 1] = 1
            D1D3[1, 2] = -2
        end
        if bc.u.y[2] == :dirichlet
            D1D3[end, end - 1] = 2
            D1D3[end, end] = -1
        end
        Cuy3 = D1D3 ⊗ I(Nux_in)
        Duy3 = D1D3 ⊗ Diagonal(gxi3)

        ## Convection (differencing) operator Cv

        # Calculates difference from pressure points to velocity points
        diag1 = ones(Nvx_t - 2)
        D1D = spdiagm(Nvx_t - 2, Nvx_t + 1, 1 => -diag1, 2 => diag1)
        Dvx = Diagonal(gyi) ⊗ D1D

        # Calculates difference from pressure points to velocity points
        diag1 = ones(Nvx_t - 2)
        D1D3 = spdiagm(Nvx_t - 2, Nvx_t + 1, 0 => -diag1, 3 => diag1)

        # Uncomment for new BC (functions/new)
        if bc.v.x[1] == :dirichlet
            D1D3[1, 1] = 1
            D1D3[1, 2] = -2
        end
        if bc.v.x[2] == :dirichlet
            D1D3[end, end - 1] = 2
            D1D3[end, end] = -1
        end
        Cvx3 = I(Nvy_in) ⊗ D1D3
        Dvx3 = Diagonal(gyi3) ⊗ D1D3

        # Calculates difference from corner points to velocity points
        diag1 = ones(Nvy_t - 2)
        D1D = spdiagm(Nvy_t - 2, Nvy_t + 1, 1 => -diag1, 2 => diag1)
        Dvy = D1D ⊗ Diagonal(hxi)

        # Calculates difference from corner points to velocity points
        diag1 = ones(Nvy_t - 2)
        D1D3 = spdiagm(Nvy_t - 2, Nvy_t + 1, 0 => -diag1, 3 => diag1)
        Cvy3 = D1D3 ⊗ I(Nvx_in)
        Dvy3 = D1D3 ⊗ Diagonal(hxi3)

        ## Su_ux: evaluate ux
        diag1 = 1 ./ hxd13
        S1D = spdiagm(Nux_in + 3, Nux_t + 4, 1 => -diag1, 2 => diag1)

        # Boundary conditions
        Su_ux_bc = bc_diff3(
            Nux_t + 4,
            Nux_in,
            Nux_t + 4 - Nux_in,
            bc.u.x[1],
            bc.u.x[2],
            hx[1],
            hx[end],
        )

        # Extend to 2D
        Su_ux = Diagonal(Ωux1) * (I(Ny) ⊗ (S1D * Su_ux_bc.B1D))
        Su_ux_bc = (;
            Su_ux_bc...,
            Bbc = Diagonal(Ωux1) * (I(Ny) ⊗ (S1D * Su_ux_bc.Btemp)),
        )

        diag1 = 1 ./ hxd3
        S1D3 = spdiagm(Nux_in + 3, Nux_t + 4, 0 => -diag1, 3 => diag1)

        # Boundary conditions
        Su_ux_bc3 = bc_diff3(
            Nux_t + 4,
            Nux_in,
            Nux_t + 4 - Nux_in,
            bc.u.x[1],
            bc.u.x[2],
            hx[1],
            hx[end],
        )

        # Extend to 2D
        Su_ux3 = Diagonal(Ωux3) * (I(Nuy_in) ⊗ (S1D3 * Su_ux_bc3.B1D))
        Su_ux_bc3 = (;
            Su_ux_bc3...,
            Bbc = Diagonal(Ωux3) * (I(Nuy_in) ⊗ (S1D3 * Su_ux_bc3.Btemp)),
        )

        ## Su_uy: evaluate uy
        diag1 = 1 ./ gyd13
        S1D = spdiagm(Nuy_in + 3, Nuy_t + 4, 1 => -diag1, 2 => diag1)

        # Boundary conditions
        Su_uy_bc = bc_diff_stag3(
            Nuy_t + 4,
            Nuy_in,
            Nuy_t + 4 - Nuy_in,
            bc.u.y[1],
            bc.u.y[2],
            hy[1],
            hy[end],
        )

        # Extend to 2D
        Su_uy = Diagonal(Ωuy1) * ((S1D * Su_uy_bc.B1D) ⊗ I(Nux_in))
        Su_uy_bc = (;
            Su_uy_bc...,
            Bbc = Diagonal(Ωuy1) * ((S1D * Su_uy_bc.Btemp) ⊗ I(Nux_in)),
        )

        diag1 = 1 ./ gyd3
        S1D3 = spdiagm(Nuy_in + 3, Nuy_t + 4, 0 => -diag1, 3 => diag1)

        # Boundary conditions
        Su_uy_bc3 = bc_diff_stag3(
            Nuy_t + 4,
            Nuy_in,
            Nuy_t + 4 - Nuy_in,
            bc.u.y[1],
            bc.u.y[2],
            hy[1],
            hy[end],
        )

        # Extend to 2D
        Su_uy3 = Diagonal(Ωuy3) * ((S1D3 * Su_uy_bc3.B1D) ⊗ I(Nux_in))
        Su_uy_bc3 = (;
            Su_uy_bc3...,
            Bbc = Diagonal(Ωuy3) * ((S1D3 * Su_uy_bc3.Btemp) ⊗ I(Nux_in)),
        )

        ## Sv_vx: evaluate vx
        diag1 = 1 ./ gxd13
        S1D = spdiagm(Nvx_in + 3, Nvx_t + 4, 1 => -diag1, 2 => diag1)

        # Boundary conditions
        Sv_vx_bc = bc_diff_stag3(
            Nvx_t + 4,
            Nvx_in,
            Nvx_t + 4 - Nvx_in,
            bc.v.x[1],
            bc.v.x[2],
            hx[1],
            hx[end],
        )

        # Extend to 2D
        Sv_vx = Diagonal(Ωvx1) * (I(Nvy_in) ⊗ (S1D * Sv_vx_bc.B1D))
        Sv_vx_bc = (;
            Sv_vx_bc...,
            Bbc = Diagonal(Ωvx1) * (I(Nvy_in) ⊗ (S1D * Sv_vx_bc.Btemp)),
        )

        diag1 = 1 ./ gxd3
        S1D3 = spdiagm(Nvx_in + 3, Nvx_t + 4, 0 => -diag1, 3 => diag1)

        # Boundary conditions
        Sv_vx_bc3 = bc_diff_stag3(
            Nvx_t + 4,
            Nvx_in,
            Nvx_t + 4 - Nvx_in,
            bc.v.x[1],
            bc.v.x[2],
            hx[1],
            hx[end],
        )
        # Extend to 2D
        Sv_vx3 = Diagonal(Ωvx3) * (I(Nvy_in) ⊗ (S1D3 * Sv_vx_bc3.B1D))
        Sv_vx_bc3 = (;
            Sv_vx_bc3...,
            Bbc = Diagonal(Ωvx3) * (I(Nvy_in) ⊗ (S1D3 * Sv_vx_bc3.Btemp)),
        )

        ## Sv_vy: evaluate vy
        diag1 = 1 ./ hyd13
        S1D = spdiagm(Nvy_in + 3, Nvy_t + 4, 1 => -diag1, 2 => diag1)

        # Boundary conditions
        Sv_vy_bc = bc_diff3(
            Nvy_t + 4,
            Nvy_in,
            Nvy_t + 4 - Nvy_in,
            bc.v.y[1],
            bc.v.y[2],
            hy[1],
            hy[end],
        )

        # Extend to 2D
        Sv_vy = Diagonal(Ωvy1) * ((S1D * Sv_vy_bc.B1D) ⊗ I(Nvx_in))
        Sv_vy_bc = (;
            Sv_vy_bc...,
            Bbc = Diagonal(Ωvy1) * ((S1D * Sv_vy_bc.Btemp) ⊗ I(Nvx_in)),
        )

        diag1 = 1 ./ hyd3
        S1D3 = spdiagm(Nvy_in + 3, Nvy_t + 4, 0 => -diag1, 3 => diag1)

        # Boundary conditions
        Sv_vy_bc3 = bc_diff3(
            Nvy_t + 4,
            Nvy_in,
            Nvy_t + 4 - Nvy_in,
            bc.v.y[1],
            bc.v.y[2],
            hy[1],
            hy[end],
        )
        # Extend to 2D
        Sv_vy3 = Diagonal(Ωvy3) * ((S1D3 * Sv_vy_bc3.B1D) ⊗ I(Nvx_in))
        Sv_vy_bc3 = (;
            Sv_vy_bc3...,
            Bbc = Diagonal(Ωvy3) * ((S1D3 * Sv_vy_bc3.Btemp) ⊗ I(Nvx_in)),
        )
    else
        ## Diffusion operator (stress tensor), u-component: similar to averaging, but with mesh sizes

        ## Su_ux: evaluate ux
        diag1 = 1 ./ hxd
        S1D = spdiagm(Nux_t - 1, Nux_t, 0 => -diag1, 1 => diag1)

        # Boundary conditions
        Su_ux_bc = bc_general(Nux_t, Nux_in, Nux_b, bc.u.x[1], bc.u.x[2], hx[1], hx[end])

        # Extend to 2D
        Su_ux = I(Ny) ⊗ (S1D * Su_ux_bc.B1D)
        Su_ux_bc = (; Su_ux_bc..., Bbc = I(Ny) ⊗ (S1D * Su_ux_bc.Btemp))

        ## Su_uy: evaluate uy
        diag1 = 1 ./ gyd
        S1D = spdiagm(Nuy_t - 1, Nuy_t, 0 => -diag1, 1 => diag1)

        # Boundary conditions
        Su_uy_bc = bc_diff_stag(Nuy_t, Nuy_in, Nuy_b, bc.u.y[1], bc.u.y[2], hy[1], hy[end])

        # Extend to 2D
        Su_uy = (S1D * Su_uy_bc.B1D) ⊗ I(Nux_in)
        Su_uy_bc = (; Su_uy_bc..., Bbc = (S1D * Su_uy_bc.Btemp) ⊗ I(Nux_in))

        ## Sv_uy: evaluate vx at uy; same as Iv_uy except for mesh sizes and -diag diag
        diag1 = 1 ./ gxd
        S1D = spdiagm(Nvx_t - 1, Nvx_t, 0 => -diag1, 1 => diag1)

        # The restriction is essentially 1D so it can be directly applied to I1D
        S1D = Bvux * S1D
        S2D = I(Nuy_t - 1) ⊗ S1D

        # Boundary conditions low/up
        Nb = Nuy_in + 1 - Nvy_in
        Sv_uy_bc_lu = bc_general(Nuy_in + 1, Nvy_in, Nb, bc.v.y[1], bc.v.y[2], hy[1], hy[end])
        Sv_uy_bc_lu = (; Sv_uy_bc_lu..., B2D = Sv_uy_bc_lu.B1D ⊗ I(Nvx_in))
        Sv_uy_bc_lu = (; Sv_uy_bc_lu..., Bbc = Sv_uy_bc_lu.Btemp ⊗ I(Nvx_in))

        # Boundary conditions left/right
        Sv_uy_bc_lr =
            bc_general_stag(Nvx_t, Nvx_in, Nvx_b, bc.v.x[1], bc.v.x[2], hx[1], hx[end])

        # Take I2D into left/right operators for convenience
        Sv_uy_bc_lr = (;
            Sv_uy_bc_lr...,
            B2D = S2D * (I(Nuy_t - 1) ⊗ Sv_uy_bc_lr.B1D),
        )
        Sv_uy_bc_lr = (;
            Sv_uy_bc_lr...,
            Bbc = S2D * (I(Nuy_t - 1) ⊗ Sv_uy_bc_lr.Btemp),
        )

        # Resulting operator:
        Sv_uy = Sv_uy_bc_lr.B2D * Sv_uy_bc_lu.B2D

        ## Diffusion operator (stress tensor), v-component: similar to averaging!

        ## Su_vx: evaluate uy at vx. Same as Iu_vx except for mesh sizes and -diag diag
        diag1 = 1 ./ gyd
        S1D = spdiagm(Nuy_t - 1, Nuy_t, 0 => -diag1, 1 => diag1)
        S1D = Buvy * S1D
        S2D = S1D ⊗ I(Nvx_t - 1)

        # Boundary conditions low/up
        Su_vx_bc_lu =
            bc_general_stag(Nuy_t, Nuy_in, Nuy_b, bc.u.y[1], bc.u.y[2], hy[1], hy[end])
        Su_vx_bc_lu = (; Su_vx_bc_lu..., B2D = S2D * (Su_vx_bc_lu.B1D ⊗ I(Nvx_t - 1)))
        Su_vx_bc_lu = (; Su_vx_bc_lu..., Bbc = S2D * (Su_vx_bc_lu.Btemp ⊗ I(Nvx_t - 1)))

        # Boundary conditions left/right
        Nb = Nvx_in + 1 - Nux_in
        Su_vx_bc_lr =
            bc_general(Nvx_in + 1, Nux_in, Nb, bc.u.x[1], bc.u.x[2], hx[1], hx[end])

        Su_vx_bc_lr = (; Su_vx_bc_lr..., B2D = I(Nuy_in) ⊗ Su_vx_bc_lr.B1D)
        Su_vx_bc_lr = (; Su_vx_bc_lr..., Bbc = I(Nuy_in) ⊗ Su_vx_bc_lr.Btemp)

        # Resulting operator:
        Su_vx = Su_vx_bc_lu.B2D * Su_vx_bc_lr.B2D

        ## Sv_vx: evaluate vx
        diag1 = 1 ./ gxd
        S1D = spdiagm(Nvx_t - 1, Nvx_t, 0 => -diag1, 1 => diag1)

        # Boundary conditions
        Sv_vx_bc = bc_diff_stag(Nvx_t, Nvx_in, Nvx_b, bc.v.x[1], bc.v.x[2], hx[1], hx[end])

        # Extend to 2D
        Sv_vx = I(Nvy_in) ⊗ (S1D * Sv_vx_bc.B1D)

        Sv_vx_bc = (; Sv_vx_bc..., Bbc = I(Nvy_in) ⊗ (S1D * Sv_vx_bc.Btemp))

        ## Sv_vy: evaluate vy
        diag1 = 1 ./ hyd
        S1D = spdiagm(Nvy_t - 1, Nvy_t, 0 => -diag1, 1 => diag1)

        # Boundary conditions
        Sv_vy_bc = bc_general(Nvy_t, Nvy_in, Nvy_b, bc.v.y[1], bc.v.y[2], hy[1], hy[end])

        # Extend to 2D
        Sv_vy = (S1D * Sv_vy_bc.B1D) ⊗ I(Nx)
        Sv_vy_bc = (; Sv_vy_bc..., Bbc = (S1D * Sv_vy_bc.Btemp) ⊗ I(Nx))
    end

    ## Assemble operators
    if viscosity_model isa LaminarModel
        if order4
            Diffux_div = (α * Dux - Dux3) * Diagonal(1 ./ Ωux)
            Diffuy_div = (α * Duy - Duy3) * Diagonal(1 ./ Ωuy)
            Diffvx_div = (α * Dvx - Dvx3) * Diagonal(1 ./ Ωvx)
            Diffvy_div = (α * Dvy - Dvy3) * Diagonal(1 ./ Ωvy)
            Diffu =
                1 / Re * Diffux_div * (α * Su_ux - Su_ux3) +
                1 / Re * Diffuy_div * (α * Su_uy - Su_uy3)
            Diffv =
                1 / Re * Diffvx_div * (α * Sv_vx - Sv_vx3) +
                1 / Re * Diffvy_div * (α * Sv_vy - Sv_vy3)
        else
            Diffu = 1 / Re * (Dux * Su_ux + Duy * Su_uy)
            Diffv = 1 / Re * (Dvx * Sv_vx + Dvy * Sv_vy)
        end
        Diff = blockdiag(Diffu, Diffv)
    end

    @pack! operators = Cux, Cuy, Cvx, Cvy
    @pack! operators = Su_ux, Su_uy
    @pack! operators = Sv_vx, Sv_vy
    @pack! operators = Su_ux_bc, Su_uy_bc, Sv_vx_bc, Sv_vy_bc
    @pack! operators = Dux, Duy, Dvx, Dvy

    if viscosity_model isa LaminarModel
        @pack! operators = Diff
    else
        @pack! operators = Sv_uy, Su_vx
    end

    if order4
        @pack! operators = Cux3, Cuy3, Cvx3, Cvy3
        @pack! operators = Su_ux_bc3, Su_uy_bc3, Sv_vx_bc3, Sv_vy_bc3
        @pack! operators = Diffux_div, Diffuy_div, Diffvx_div, Diffvy_div
    else
        @pack! operators = Su_vx_bc_lr, Su_vx_bc_lu, Sv_uy_bc_lr, Sv_uy_bc_lu
    end

    setup
end

# 3D version
function operator_convection_diffusion!(setup::Setup{T,3}) where {T}
    (; grid, operators, bc, viscosity_model) = setup
    (; Nx, Ny, Nz) = grid
    (; Nux_in, Nux_b, Nux_t, Nuy_in, Nuy_b, Nuy_t, Nuz_in, Nuz_b, Nuz_t) = grid
    (; Nvx_in, Nvx_b, Nvx_t, Nvy_in, Nvy_b, Nvy_t, Nvz_in, Nvz_b, Nvz_t) = grid
    (; Nwx_in, Nwx_b, Nwx_t, Nwy_in, Nwy_b, Nwy_t, Nwz_in, Nwz_b, Nwz_t) = grid
    (; hx, hy, hz, hxi, hyi, hzi, hxd, hyd, hzd) = grid
    (; gxi, gyi, gzi, gxd, gyd, gzd) = grid
    (; Bvux, Bwux, Buvy, Bwvy, Buwz, Bvwz) = grid
    (; Re) = viscosity_model

    ## Convection (differencing) operator Cu

    # Calculates difference from pressure points to velocity points
    diag1 = ones(Nux_t - 2)
    M1D = spdiagm(Nux_t - 2, Nux_t - 1, 0 => -diag1, 1 => diag1)
    Cux = I(Nz) ⊗ I(Nuy_in) ⊗ M1D
    Dux = Diagonal(hzi) ⊗ Diagonal(hyi) ⊗ M1D

    # Calculates difference from corner points to velocity points
    diag1 = ones(Nuy_t - 2)
    M1D = spdiagm(Nuy_t - 2, Nuy_t - 1, 0 => -diag1, 1 => diag1)
    Cuy = I(Nz) ⊗ M1D ⊗ I(Nux_in)
    Duy = Diagonal(hzi) ⊗ M1D ⊗ Diagonal(gxi)

    # Calculates difference from corner points to velocity points
    diag1 = ones(Nuz_t - 2)
    M1D = spdiagm(Nuz_t - 2, Nuz_t - 1, 0 => -diag1, 1 => diag1)
    Cuz = M1D ⊗ I(Ny) ⊗ I(Nux_in)
    Duz = M1D ⊗ Diagonal(hyi) ⊗ Diagonal(gxi)

    # Cu = [Cux Cuy Cuz]
    # Du = [Dux Duy Duz]

    ## Convection (differencing) operator Cv

    # Calculates difference from pressure points to velocity points
    diag1 = ones(Nvx_t - 2)
    M1D = spdiagm(Nvx_t - 2, Nvx_t - 1, 0 => -diag1, 1 => diag1)
    Cvx = I(Nz) ⊗ I(Nvy_in) ⊗ M1D
    Dvx = Diagonal(hzi) ⊗ Diagonal(gyi) ⊗ M1D

    # Calculates difference from corner points to velocity points
    diag1 = ones(Nvy_t - 2)
    M1D = spdiagm(Nvy_t - 2, Nvy_t - 1, 0 => -diag1, 1 => diag1)
    Cvy = I(Nz) ⊗ M1D ⊗ I(Nx)
    Dvy = Diagonal(hzi) ⊗ M1D ⊗ Diagonal(hxi)

    # Calculates difference from corner points to velocity points
    diag1 = ones(Nvz_t - 2)
    M1D = spdiagm(Nvz_t - 2, Nvz_t - 1, 0 => -diag1, 1 => diag1)
    Cvz = M1D ⊗ I(Nvy_in) ⊗ I(Nx)
    Dvz = M1D ⊗ Diagonal(gyi) ⊗ Diagonal(hxi)

    # Cv = [Cvx Cvy Cvz]
    # Dv = [Dvx Dvy Dvz]

    ## Convection (differencing) operator Cw

    # Calculates difference from pressure points to velocity points
    diag1 = ones(Nwx_t - 2)
    M1D = spdiagm(Nwx_t - 2, Nwx_t - 1, 0 => -diag1, 1 => diag1)
    Cwx = I(Nwz_in) ⊗ I(Ny) ⊗ M1D
    Dwx = Diagonal(gzi) ⊗ Diagonal(hyi) ⊗ M1D

    # Calculates difference from corner points to velocity points
    diag1 = ones(Nwy_t - 2)
    M1D = spdiagm(Nwy_t - 2, Nwy_t - 1, 0 => -diag1, 1 => diag1)
    Cwy = I(Nwz_in) ⊗ M1D ⊗ I(Nx)
    Dwy = Diagonal(gzi) ⊗ M1D ⊗ Diagonal(hxi)

    # Calculates difference from corner points to velocity points
    diag1 = ones(Nwz_t - 2)
    M1D = spdiagm(Nwz_t - 2, Nwz_t - 1, 0 => -diag1, 1 => diag1)
    Cwz = M1D ⊗ I(Ny) ⊗ I(Nx)
    Dwz = M1D ⊗ Diagonal(hyi) ⊗ Diagonal(hxi)

    # Cw = [Cwx Cwy Cwz]
    # Dw = [Dwx Dwy Dwz]

    ## Diffusion operator (stress tensor), u-component: similar to averaging, but with mesh sizes

    ## Su_ux: evaluate ux
    diag1 = 1 ./ hxd
    S1D = spdiagm(Nux_t - 1, Nux_t, 0 => -diag1, 1 => diag1)

    # Boundary conditions
    Su_ux_bc = bc_general(Nux_t, Nux_in, Nux_b, bc.u.x[1], bc.u.x[2], hx[1], hx[end])

    # Extend to 3D
    Su_ux = I(Nuz_in) ⊗ I(Nuy_in) ⊗ (S1D * Su_ux_bc.B1D)
    Su_ux_bc = (; Su_ux_bc..., Bbc = I(Nuz_in) ⊗ I(Nuy_in) ⊗ (S1D * Su_ux_bc.Btemp))

    ## Su_uy: evaluate uy
    diag1 = 1 ./ gyd
    S1D = spdiagm(Nuy_t - 1, Nuy_t, 0 => -diag1, 1 => diag1)

    # Boundary conditions
    Su_uy_bc = bc_diff_stag(Nuy_t, Nuy_in, Nuy_b, bc.u.y[1], bc.u.y[2], hy[1], hy[end])

    # Extend to 3D
    Su_uy = I(Nuz_in) ⊗ (S1D * Su_uy_bc.B1D) ⊗ I(Nux_in)
    Su_uy_bc = (; Su_uy_bc..., Bbc = I(Nuz_in) ⊗ (S1D * Su_uy_bc.Btemp) ⊗ I(Nux_in))

    ## Su_uz: evaluate uz
    diag1 = 1 ./ gzd
    S1D = spdiagm(Nuz_t - 1, Nuz_t, 0 => -diag1, 1 => diag1)

    # Boundary conditions
    Su_uz_bc = bc_diff_stag(Nuz_t, Nuz_in, Nuz_b, bc.u.z[1], bc.u.z[2], hz[1], hz[end])

    # Extend to 3D
    Su_uz = (S1D * Su_uz_bc.B1D) ⊗ I(Nuy_in) ⊗ I(Nux_in)
    Su_uz_bc = (; Su_uz_bc..., Bbc = (S1D * Su_uz_bc.Btemp) ⊗ I(Nuy_in) ⊗ I(Nux_in))


    ## Sv_uy: evaluate vx at uy; same as Iv_uy except for mesh sizes and -diag diag
    diag1 = 1 ./ gxd
    S1D = spdiagm(Nvx_t - 1, Nvx_t, 0 => -diag1, 1 => diag1)

    # The restriction is essentially 1D so it can be directly applied to I1D
    S1D = Bvux * S1D
    S3D = I(Nz) ⊗ I(Nuy_t - 1) ⊗ S1D

    # Boundary conditions left/right
    Sv_uy_bc_lr =
        bc_general_stag(Nvx_t, Nvx_in, Nvx_b, bc.v.x[1], bc.v.x[2], hx[1], hx[end])

    # Take I3D into left/right operators for convenience
    Sv_uy_bc_lr = (; Sv_uy_bc_lr..., B3D = S3D * (I(Nz) ⊗ I(Nuy_t - 1) ⊗ Sv_uy_bc_lr.B1D))
    Sv_uy_bc_lr = (; Sv_uy_bc_lr..., Bbc = S3D * (I(Nz) ⊗ I(Nuy_t - 1) ⊗ Sv_uy_bc_lr.Btemp))

    # Boundary conditions low/up
    Nb = Nuy_in + 1 - Nvy_in
    Sv_uy_bc_lu =
        bc_general(Nuy_in + 1, Nvy_in, Nb, bc.v.y[1], bc.v.y[2], hy[1], hy[end])
    Sv_uy_bc_lu = (; Sv_uy_bc_lu..., B3D = I(Nz) ⊗ Sv_uy_bc_lu.B1D ⊗ I(Nvx_in))
    Sv_uy_bc_lu = (; Sv_uy_bc_lu..., Bbc = I(Nz) ⊗ Sv_uy_bc_lu.Btemp ⊗ I(Nvx_in))

    # Resulting operator:
    Sv_uy = Sv_uy_bc_lr.B3D * Sv_uy_bc_lu.B3D

    ## Sw_uz: evaluate wx at uz; same as Iw_uz except for mesh sizes and -diag diag
    diag1 = 1 ./ gxd
    S1D = spdiagm(Nwx_t - 1, Nwx_t, 0 => -diag1, 1 => diag1)

    # The restriction is essentially 1D so it can be directly applied to I1D
    S1D = Bwux * S1D
    S3D = I(Nz + 1) ⊗ I(Ny) ⊗ S1D

    # Boundary conditions left/right
    Sw_uz_bc_lr =
        bc_general_stag(Nwx_t, Nwx_in, Nwx_b, bc.w.x[1], bc.w.x[2], hx[1], hx[end])

    # Take I3D into left/right operators for convenience
    Sw_uz_bc_lr = (; Sw_uz_bc_lr..., B3D = S3D * (I(Nz + 1) ⊗ I(Ny) ⊗ Sw_uz_bc_lr.B1D))
    Sw_uz_bc_lr = (; Sw_uz_bc_lr..., Bbc = S3D * (I(Nz + 1) ⊗ I(Ny) ⊗ Sw_uz_bc_lr.Btemp))

    # Boundary conditions back/front
    Nb = Nuz_in + 1 - Nwz_in
    Sw_uz_bc_bf =
        bc_general(Nuz_in + 1, Nwz_in, Nb, bc.w.z[1], bc.w.z[2], hz[1], hz[end])
    Sw_uz_bc_bf = (; Sw_uz_bc_bf..., B3D = Sw_uz_bc_bf.B1D ⊗ I(Ny) ⊗ I(Nx))
    Sw_uz_bc_bf = (; Sw_uz_bc_bf..., Bbc = Sw_uz_bc_bf.Btemp ⊗ I(Ny) ⊗ I(Nx))

    # Resulting operator:
    Sw_uz = Sw_uz_bc_lr.B3D * Sw_uz_bc_bf.B3D


    ## Diffusion operator (stress tensor), v-component: similar to averaging!

    ## Su_vx: evaluate uy at vx. Same as Iu_vx except for mesh sizes and -diag diag
    diag1 = 1 ./ gyd
    S1D = spdiagm(Nuy_t - 1, Nuy_t, 0 => -diag1, 1 => diag1)
    S1D = Buvy * S1D
    S3D = I(Nz) ⊗ S1D ⊗ I(Nx + 1)

    # Boundary conditions left/right
    Nb = Nvx_in + 1 - Nux_in
    Su_vx_bc_lr =
        bc_general(Nvx_in + 1, Nux_in, Nb, bc.u.x[1], bc.u.x[2], hx[1], hx[end])
    Su_vx_bc_lr = (; Su_vx_bc_lr..., B3D = I(Nz) ⊗ I(Ny) ⊗ Su_vx_bc_lr.B1D)
    Su_vx_bc_lr = (; Su_vx_bc_lr..., Bbc = I(Nz) ⊗ I(Ny) ⊗ Su_vx_bc_lr.Btemp)

    # Boundary conditions low/up
    Su_vx_bc_lu =
        bc_general_stag(Nuy_t, Nuy_in, Nuy_b, bc.u.y[1], bc.u.y[2], hy[1], hy[end])
    Su_vx_bc_lu = (; Su_vx_bc_lu..., B3D = S3D * (I(Nz) ⊗ Su_vx_bc_lu.B1D ⊗ I(Nx + 1)))
    Su_vx_bc_lu = (; Su_vx_bc_lu..., Bbc = S3D * (I(Nz) ⊗ Su_vx_bc_lu.Btemp ⊗ I(Nx + 1)))

    # Resulting operator:
    Su_vx = Su_vx_bc_lu.B3D * Su_vx_bc_lr.B3D

    ## Sv_vx: evaluate vx
    diag1 = 1 ./ gxd
    S1D = spdiagm(Nvx_t - 1, Nvx_t, 0 => -diag1, 1 => diag1)

    # Boundary conditions
    Sv_vx_bc = bc_diff_stag(Nvx_t, Nvx_in, Nvx_b, bc.v.x[1], bc.v.x[2], hx[1], hx[end])

    # Extend to 3D
    Sv_vx = I(Nvz_in) ⊗ I(Nvy_in) ⊗ (S1D * Sv_vx_bc.B1D)
    Sv_vx_bc = (; Sv_vx_bc..., Bbc = I(Nvz_in) ⊗ I(Nvy_in) ⊗ (S1D * Sv_vx_bc.Btemp))

    ## Sv_vy: evaluate vy
    diag1 = 1 ./ hyd
    S1D = spdiagm(Nvy_t - 1, Nvy_t, 0 => -diag1, 1 => diag1)

    # Boundary conditions
    Sv_vy_bc = bc_general(Nvy_t, Nvy_in, Nvy_b, bc.v.y[1], bc.v.y[2], hy[1], hy[end])

    # Extend to 3D
    Sv_vy = I(Nvz_in) ⊗ (S1D * Sv_vy_bc.B1D) ⊗ I(Nx)
    Sv_vy_bc = (; Sv_vy_bc..., Bbc = I(Nvz_in) ⊗ (S1D * Sv_vy_bc.Btemp) ⊗ I(Nx))

    ## Sv_vz: evaluate v at vz location
    diag1 = 1 ./ gzd
    S1D = spdiagm(Nvz_t - 1, Nvz_t, 0 => -diag1, 1 => diag1)

    # Boundary conditions
    Sv_vz_bc = bc_diff_stag(Nvz_t, Nvz_in, Nvz_b, bc.v.z[1], bc.v.z[2], hz[1], hz[end])

    # Extend to 3D
    Sv_vz = (S1D * Sv_vz_bc.B1D) ⊗ I(Nvy_in) ⊗ I(Nvx_in)
    Sv_vz_bc = (; Sv_vz_bc..., Bbc = (S1D * Sv_vz_bc.Btemp) ⊗ I(Nvy_in) ⊗ I(Nvx_in))


    ## Sw_vz: evaluate wy at vz location
    diag1 = 1 ./ gyd
    S1D = spdiagm(Nwy_t - 1, Nwy_t, 0 => -diag1, 1 => diag1)
    S1D = Bwvy * S1D
    S3D = I(Nz + 1) ⊗ S1D ⊗ I(Nx)

    # Boundary conditions low/up
    Sw_vz_bc_lu =
        bc_general_stag(Nwy_t, Nwy_in, Nwy_b, bc.w.y[1], bc.w.y[2], hy[1], hy[end])
    Sw_vz_bc_lu = (; Sw_vz_bc_lu..., B3D = S3D * (I(Nz + 1) ⊗ Sw_vz_bc_lu.B1D ⊗ I(Nx)))
    Sw_vz_bc_lu = (; Sw_vz_bc_lu..., Bbc = S3D * (I(Nz + 1) ⊗ Sw_vz_bc_lu.Btemp ⊗ I(Nx)))

    # Boundary conditions back/front
    Nb = Nvz_in + 1 - Nwz_in
    Sw_vz_bc_bf =
        bc_general(Nvz_in + 1, Nwz_in, Nb, bc.w.z[1], bc.w.z[2], hz[1], hz[end])
    Sw_vz_bc_bf = (; Sw_vz_bc_bf..., B3D = Sw_vz_bc_bf.B1D ⊗ I(Ny) ⊗ I(Nx))
    Sw_vz_bc_bf = (; Sw_vz_bc_bf..., Bbc = Sw_vz_bc_bf.Btemp ⊗ I(Ny) ⊗ I(Nx))

    # Resulting operator:
    Sw_vz = Sw_vz_bc_lu.B3D * Sw_vz_bc_bf.B3D






    ## Diffusion operator (stress tensor), w-component: similar to averaging, but with mesh sizes

    ## Sw_wx: evaluate w at wx location
    diag1 = 1 ./ gxd
    S1D = spdiagm(Nwx_t - 1, Nwx_t, 0 => -diag1, 1 => diag1)

    # Boundary conditions
    Sw_wx_bc = bc_diff_stag(Nwx_t, Nwx_in, Nwx_b, bc.w.x[1], bc.w.x[2], hx[1], hx[end])

    # Extend to 3D
    Sw_wx = I(Nwz_in) ⊗ I(Nwy_in) ⊗ (S1D * Sw_wx_bc.B1D)
    Sw_wx_bc = (; Sw_wx_bc..., Bbc = I(Nwz_in) ⊗ I(Nwy_in) ⊗ (S1D * Sw_wx_bc.Btemp))

    ## Sw_wy: evaluate w at wy location
    diag1 = 1 ./ gyd
    S1D = spdiagm(Nwy_t - 1, Nwy_t, 0 => -diag1, 1 => diag1)

    # Boundary conditions
    Sw_wy_bc = bc_diff_stag(Nwy_t, Nwy_in, Nwy_b, bc.w.y[1], bc.w.y[2], hy[1], hy[end])

    # Extend to 3D
    Sw_wy = I(Nwz_in) ⊗ (S1D * Sw_wy_bc.B1D) ⊗ I(Nwx_in)
    Sw_wy_bc = (; Sw_wy_bc..., Bbc = I(Nwz_in) ⊗ (S1D * Sw_wy_bc.Btemp) ⊗ I(Nwx_in))

    ## Sw_wz: evaluate w at wz location
    diag1 = 1 ./ hzd
    S1D = spdiagm(Nwz_t - 1, Nwz_t, 0 => -diag1, 1 => diag1)

    # Boundary conditions
    Sw_wz_bc = bc_diff_stag(Nwz_t, Nwz_in, Nwz_b, bc.w.z[1], bc.w.z[2], hz[1], hz[end])

    # Extend to 3D
    Sw_wz = (S1D * Sw_wz_bc.B1D) ⊗ I(Nwy_in) ⊗ I(Nwx_in)
    Sw_wz_bc = (; Sw_wz_bc..., Bbc = (S1D * Sw_wz_bc.Btemp) ⊗ I(Nwy_in) ⊗ I(Nwx_in))

    ## Su_wx: evaluate uz at wx
    diag1 = 1 ./ gzd
    S1D = spdiagm(Nuz_t - 1, Nuz_t, 0 => -diag1, 1 => diag1)

    # The restriction is essentially 1D so it can be directly applied to I1D
    S1D = Buwz * S1D
    S3D = S1D ⊗ I(Ny) ⊗ I(Nx + 1)

    # Boundary conditions back/front
    Su_wx_bc_bf =
        bc_general_stag(Nuz_t, Nuz_in, Nuz_b, bc.u.z[1], bc.u.z[2], hz[1], hz[end])

    # Take I3D into left/right operators for convenience
    Su_wx_bc_bf = (; Su_wx_bc_bf..., B3D = S3D * (Su_wx_bc_bf.B1D ⊗ I(Ny) ⊗ I(Nx + 1)))
    Su_wx_bc_bf = (; Su_wx_bc_bf..., Bbc = S3D * (Su_wx_bc_bf.Btemp ⊗ I(Ny) ⊗ I(Nx + 1)))

    # Boundary conditions left/right
    Nb = Nwx_in + 1 - Nux_in
    Su_wx_bc_lr =
        bc_general(Nwx_in + 1, Nux_in, Nb, bc.u.x[1], bc.u.x[2], hx[1], hx[end])
    Su_wx_bc_lr = (; Su_wx_bc_lr..., B3D = I(Nz) ⊗ I(Ny) ⊗ Su_wx_bc_lr.B1D)
    Su_wx_bc_lr = (; Su_wx_bc_lr..., Bbc = I(Nz) ⊗ I(Ny) ⊗ Su_wx_bc_lr.Btemp)

    # Resulting operator:
    Su_wx = Su_wx_bc_bf.B3D * Su_wx_bc_lr.B3D

    ## Sv_wy: evaluate vz at wy
    diag1 = 1 ./ gzd
    S1D = spdiagm(Nvz_t - 1, Nvz_t, 0 => -diag1, 1 => diag1)

    # The restriction is essentially 1D so it can be directly applied to I1D
    S1D = Bvwz * S1D
    S3D = S1D ⊗ I(Ny + 1) ⊗ I(Nx)

    # Boundary conditions back/front
    Sv_wy_bc_bf =
        bc_general_stag(Nvz_t, Nvz_in, Nvz_b, bc.v.z[1], bc.v.z[2], hz[1], hz[end])

    # Take I3D into left/right operators for convenience
    Sv_wy_bc_bf = (; Sv_wy_bc_bf..., B3D = S3D * (Sv_wy_bc_bf.B1D ⊗ I(Ny + 1) ⊗ I(Nx)))
    Sv_wy_bc_bf = (; Sv_wy_bc_bf..., Bbc = S3D * (Sv_wy_bc_bf.Btemp ⊗ I(Ny + 1) ⊗ I(Nx)))

    # Boundary conditions low/up
    Nb = Nwy_in + 1 - Nvy_in
    Sv_wy_bc_lu =
        bc_general(Nwy_in + 1, Nvy_in, Nb, bc.v.y[1], bc.v.y[2], hy[1], hy[end])
    Sv_wy_bc_lu = (; Sv_wy_bc_lu..., B3D = I(Nz) ⊗ Sv_wy_bc_lu.B1D ⊗ I(Nx))
    Sv_wy_bc_lu = (; Sv_wy_bc_lu..., Bbc = I(Nz) ⊗ Sv_wy_bc_lu.Btemp ⊗ I(Nx))

    # Resulting operator:
    Sv_wy = Sv_wy_bc_bf.B3D * Sv_wy_bc_lu.B3D

    ## Assemble operators
    if viscosity_model isa LaminarModel
        Diffu = 1 / Re * (Dux * Su_ux + Duy * Su_uy + Duz * Su_uz)
        Diffv = 1 / Re * (Dvx * Sv_vx + Dvy * Sv_vy + Dvz * Sv_vz)
        Diffw = 1 / Re * (Dwx * Sw_wx + Dwy * Sw_wy + Dwz * Sw_wz)
        Diff = blockdiag(Diffu, Diffv, Diffw)
    end

    @pack! operators = Cux, Cuy, Cuz, Cvx, Cvy, Cvz, Cwx, Cwy, Cwz
    @pack! operators = Su_ux, Su_uy, Su_uz
    @pack! operators = Sv_vx, Sv_vy, Sv_vz
    @pack! operators = Sw_wx, Sw_wy, Sw_wz
    @pack! operators = Su_ux_bc, Su_uy_bc, Su_uz_bc
    @pack! operators = Sv_vx_bc, Sv_vy_bc, Sv_vz_bc
    @pack! operators = Sw_wx_bc, Sw_wy_bc, Sw_wz_bc
    @pack! operators = Dux, Duy, Duz, Dvx, Dvy, Dvz, Dwx, Dwy, Dwz

    if viscosity_model isa LaminarModel
        @pack! operators = Diff
    else
        @pack! operators = Sv_uy, Su_vx, Sw_uz, Su_wx, Sw_vz, Sv_wy
    end

    @pack! operators = Sv_uy_bc_lr, Sv_uy_bc_lu, Sw_uz_bc_lr, Sw_uz_bc_bf
    @pack! operators = Su_vx_bc_lr, Su_vx_bc_lu, Sw_vz_bc_lu, Sw_vz_bc_bf
    @pack! operators = Su_wx_bc_lr, Su_wx_bc_bf, Sv_wy_bc_lu, Sv_wy_bc_bf

    setup
end
