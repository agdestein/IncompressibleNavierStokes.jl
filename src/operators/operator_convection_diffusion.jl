"""
Construct convection and diffusion operators.
"""
function operator_convection_diffusion!(setup)
    # Boundary conditions
    bc = setup.bc

    # Number of interior points and boundary points
    @unpack Nx, Ny = setup.grid
    @unpack Nu, Nux_in, Nux_b, Nux_t, Nuy_in, Nuy_b, Nuy_t = setup.grid
    @unpack Nv, Nvx_in, Nvx_b, Nvx_t, Nvy_in, Nvy_b, Nvy_t = setup.grid
    @unpack hx, hy, hxi, hyi, hxd, hyd = setup.grid
    @unpack gxi, gyi, gxd, gyd = setup.grid
    @unpack Buvy, Bvux = setup.grid
    @unpack Ω⁻¹ = setup.grid
    @unpack time_stepper, Δt = setup.time

    @unpack order4 = setup.discretization

    if order4
        α = setup.discretization.α
        @unpack hxi3, hyi3, gxi3, gyi3, hxd13, hxd3, hyd13, hyd3 = setup.grid
        @unpack gxd13, gxd3, gyd13, gyd3 = setup.grid
        @unpack Ωux, Ωuy, Ωvx, Ωvy = setup.grid
        @unpack Ωux1, Ωux3, Ωuy1, Ωuy3, Ωvx1, Ωvx3, Ωvy1, Ωvy3 = setup.grid
    end

    @unpack model = setup
    @unpack Re = setup.fluid

    ## Convection (differencing) operator Cu

    # Calculates difference from pressure points to velocity points
    diag1 = ones(Nux_t - 2)
    D1D = spdiagm(Nux_t - 2, Nux_t - 1, 0 => -diag1, 1 => diag1)
    Cux = kron(sparse(I, Nuy_in, Nuy_in), D1D)
    if !order4
        Dux = kron(Diagonal(hyi), D1D)
    end

    # Calculates difference from corner points to velocity points
    diag1 = ones(Nuy_t - 2)
    D1D = spdiagm(Nuy_t - 2, Nuy_t - 1, 0 => -diag1, 1 => diag1)
    Cuy = kron(D1D, sparse(I, Nux_in, Nux_in))
    if !order4
        Duy = kron(D1D, Diagonal(gxi))
    end

    # Cu = [Cux Cuy]
    # Du = [Dux Duy]

    ## Convection (differencing) operator Cv

    # Calculates difference from pressure points to velocity points
    diag1 = ones(Nvx_t - 2)
    D1D = spdiagm(Nvx_t - 2, Nvx_t - 1, 0 => -diag1, 1 => diag1)
    Cvx = kron(sparse(I, Nvy_in, Nvy_in), D1D)
    if !order4
        Dvx = kron(Diagonal(gyi), D1D)
    end

    # Calculates difference from corner points to velocity points
    diag1 = ones(Nvy_t - 2)
    D1D = spdiagm(Nvy_t - 2, Nvy_t - 1, 0 => -diag1, 1 => diag1)
    Cvy = kron(D1D, sparse(I, Nvx_in, Nvx_in))
    if !order4
        Dvy = kron(D1D, Diagonal(hxi))
    end

    # Cv = [Cvx Cvy]
    # Dv = [Dvx Dvy]

    if order4
        ## Fourth order operators
        ## Convection (differencing) operator Cu

        # Calculates difference from pressure points to velocity points
        diag1 = ones(Nux_t)
        D1D = spdiagm(Nux_t - 2, Nux_t + 1, 1 => -diag1, 2 => diag1)
        Dux = kron(Diagonal(hyi), D1D)

        # The "second order" Cux is unchanged
        # The "second order" Dux changes, because we also use the "second
        # Order" flux at "fourth order" ghost points (Dux should have the same
        # Size as Dux3)

        # Calculates difference from pressure points to velocity points
        diag1 = ones(Nux_t)
        D1D3 = spdiagm(Nux_t - 2, Nux_t + 1, 0 => -diag1, 3 => diag1)
        Cux3 = kron(sparse(I, Ny, Ny), D1D3)
        Dux3 = kron(Diagonal(hyi3), D1D3)

        # Calculates difference from corner points to velocity points
        diag1 = ones(Nuy_t)
        D1D = spdiagm(Nuy_t - 2, Nuy_t + 1, 1 => -diag1, 2 => diag1)
        Duy = kron(D1D, Diagonal(gxi))

        # Calculates difference from corner points to velocity points
        diag1 = ones(Nuy_t)
        D1D3 = spdiagm(Nuy_t - 2, Nuy_t + 1, 0 => -diag1, 3 => diag1)

        # Uncomment for new BC (functions/new)
        if bc.u.low == :dirichlet
            D1D3[1, 1] = 1
            D1D3[1, 2] = -2
        end
        if bc.u.up == :dirichlet
            D1D3[end, end-1] = 2
            D1D3[end, end] = -1
        end
        Cuy3 = kron(D1D3, sparse(I, Nux_in, Nux_in))
        Duy3 = kron(D1D3, Diagonal(gxi3))

        ## Convection (differencing) operator Cv

        # Calculates difference from pressure points to velocity points
        diag1 = ones(Nvx_t)
        D1D = spdiagm(Nvx_t - 2, Nvx_t + 1, 1 => -diag1, 2 => diag1)
        Dvx = kron(Diagonal(gyi), D1D)

        # Calculates difference from pressure points to velocity points
        diag1 = ones(Nvx_t)
        D1D3 = spdiagm(Nvx_t - 2, Nvx_t + 1, 0 => -diag1, 3 => diag1)

        # Uncomment for new BC (functions/new)
        if bc.v.left == :dirichlet
            D1D3[1, 1] = 1
            D1D3[1, 2] = -2
        end
        if bc.v.right == :dirichlet
            D1D3[end, end-1] = 2
            D1D3[end, end] = -1
        end
        Cvx3 = kron(sparse(I, Nvy_in, Nvy_in), D1D3)
        Dvx3 = kron(Diagonal(gyi3), D1D3)

        # Calculates difference from corner points to velocity points
        diag1 = ones(Nvy_t, 1)
        D1D = spdiagm(Nvy_t - 2, Nvy_t + 1, 1 => -diag1, 2 => diag1)
        Dvy = kron(D1D, Diagonal(hxi))

        # Calculates difference from corner points to velocity points
        diag1 = ones(Nvy_t, 1)
        D1D3 = spdiagm(Nvy_t - 2, Nvy_t + 1, 0 => -diag1, 3 => diag1)
        Cvy3 = kron(D1D3, sparse(I, Nvx_in, Nvx_in))
        Dvy3 = kron(D1D3, Diagonal(hxi3))

        ## Su_ux: evaluate ux
        diag1 = 1 ./ hxd13
        S1D = spdiagm(Nux_in + 3, Nux_t + 4, 1 => -diag1, 2 => diag1)

        # Boundary conditions
        Su_ux_bc = bc_diff3(
            Nux_t + 4,
            Nux_in,
            Nux_t + 4 - Nux_in,
            bc.u.left,
            bc.u.right,
            hx[1],
            hx[end],
        )

        # Extend to 2D
        Su_ux = Diagonal(Ωux1) * kron(sparse(I, Ny, Ny), S1D * Su_ux_bc.B1D)
        Su_ux_bc = (;
            Su_ux_bc...,
            Bbc = Diagonal(Ωux1) * kron(sparse(I, Ny, Ny), S1D * Su_ux_bc.Btemp),
        )

        diag1 = 1 ./ hxd3
        S1D3 = spdiagm(Nux_in + 3, Nux_t + 4, 0 => -diag1, 3 => diag1)

        # Boundary conditions
        Su_ux_bc3 = bc_diff3(
            Nux_t + 4,
            Nux_in,
            Nux_t + 4 - Nux_in,
            bc.u.left,
            bc.u.right,
            hx[1],
            hx[end],
        )

        # Extend to 2D
        Su_ux3 = Diagonal(Ωux3) * kron(sparse(I, Nuy_in, Nuy_in), S1D3 * Su_ux_bc3.B1D)
        Su_ux_bc3.Bbc =
            Diagonal(Ωux3) * kron(sparse(I, Nuy_in, Nuy_in), S1D3 * Su_ux_bc3.Btemp)

        ## Su_uy: evaluate uy
        diag1 = 1 ./ gyd13
        S1D = spdiagm(Nuy_in + 3, Nuy_t + 4, 1 => -diag1, 2 => diag1)

        # Boundary conditions
        Su_uy_bc = bc_diff_stag3(
            Nuy_t + 4,
            Nuy_in,
            Nuy_t + 4 - Nuy_in,
            bc.u.low,
            bc.u.up,
            hy[1],
            hy[end],
        )

        # Extend to 2D
        Su_uy = Diagonal(Ωuy1) * kron(S1D * Su_uy_bc.B1D, sparse(I, Nux_in, Nux_in))
        Su_uy_bc.Bbc =
            Diagonal(Ωuy1) * kron(S1D * Su_uy_bc.Btemp, sparse(I, Nux_in, Nux_in))

        diag1 = 1 ./ gyd3
        S1D3 = spdiagm(Nuy_in + 3, Nuy_t + 4, 0 => -diag1, 3 => diag1)

        # Boundary conditions
        Su_uy_bc3 = bc_diff_stag3(
            Nuy_t + 4,
            Nuy_in,
            Nuy_t + 4 - Nuy_in,
            bc.u.low,
            bc.u.up,
            hy[1],
            hy[end],
        )

        # Extend to 2D
        Su_uy3 = Diagonal(Ωuy3) * kron(S1D3 * Su_uy_bc3.B1D, sparse(I, Nux_in, Nux_in))
        Su_uy_bc3.Bbc =
            Diagonal(Ωuy3) * kron(S1D3 * Su_uy_bc3.Btemp, sparse(I, Nux_in, Nux_in))

        ## Sv_vx: evaluate vx
        diag1 = 1 ./ gxd13
        S1D = spdiagm(Nvx_in + 3, Nvx_t + 4, 1 => -diag1, 2 => diag1)

        # Boundary conditions
        Sv_vx_bc = bc_diff_stag3(
            Nvx_t + 4,
            Nvx_in,
            Nvx_t + 4 - Nvx_in,
            bc.v.left,
            bc.v.right,
            hx[1],
            hx[end],
        )

        # Extend to 2D
        Sv_vx = Diagonal(Ωvx1) * kron(sparse(I, Nvy_in, Nvy_in), S1D * Sv_vx_bc.B1D)
        Sv_vx_bc.Bbc =
            Diagonal(Ωvx1) * kron(sparse(I, Nvy_in, Nvy_in), S1D * Sv_vx_bc.Btemp)

        diag1 = 1 ./ gxd3
        S1D3 = spdiagm(Nvx_in + 3, Nvx_t + 4, 0 => -diag1, 3 => diag1)

        # Boundary conditions
        Sv_vx_bc3 = bc_diff_stag3(
            Nvx_t + 4,
            Nvx_in,
            Nvx_t + 4 - Nvx_in,
            bc.v.left,
            bc.v.right,
            hx[1],
            hx[end],
        )
        # Extend to 2D
        Sv_vx3 = Diagonal(Ωvx3) * kron(sparse(I, Nvy_in, Nvy_in), S1D3 * Sv_vx_bc3.B1D)
        Sv_vx_bc3.Bbc =
            Diagonal(Ωvx3) * kron(sparse(I, Nvy_in, Nvy_in), S1D3 * Sv_vx_bc3.Btemp)

        ## Sv_vy: evaluate vy
        diag1 = 1 ./ hyd13
        S1D = spdiagm(Nvy_in + 3, Nvy_t + 4, 1 => -diag1, 2 => diag1)

        # Boundary conditions
        Sv_vy_bc = bc_diff3(
            Nvy_t + 4,
            Nvy_in,
            Nvy_t + 4 - Nvy_in,
            bc.v.low,
            bc.v.up,
            hy[1],
            hy[end],
        )

        # Extend to 2D
        Sv_vy = Diagonal(Ωvy1) * kron(S1D * Sv_vy_bc.B1D, sparse(I, Nvx_in, Nvx_in))
        Sv_vy_bc.Bbc =
            Diagonal(Ωvy1) * kron(S1D * Sv_vy_bc.Btemp, sparse(I, Nvx_in, Nvx_in))

        diag1 = 1 ./ hyd3
        S1D3 = spdiagm(Nvy_in + 3, Nvy_t + 4, 0 => -diag1, 3 => diag1)

        # Boundary conditions
        Sv_vy_bc3 = bc_diff3(
            Nvy_t + 4,
            Nvy_in,
            Nvy_t + 4 - Nvy_in,
            bc.v.low,
            bc.v.up,
            hy[1],
            hy[end],
        )
        # Extend to 2D
        Sv_vy3 = Diagonal(Ωvy3) * kron(S1D3 * Sv_vy_bc3.B1D, sparse(I, Nvx_in, Nvx_in))
        Sv_vy_bc3.Bbc =
            Diagonal(Ωvy3) * kron(S1D3 * Sv_vy_bc3.Btemp, sparse(I, Nvx_in, Nvx_in))
    else
        ## Diffusion operator (stress tensor), u-component: similar to averaging, but with mesh sizes

        ## Su_ux: evaluate ux
        diag1 = 1 ./ hxd
        S1D = spdiagm(Nux_t - 1, Nux_t, 0 => -diag1, 1 => diag1)

        # Boundary conditions
        Su_ux_bc = bc_general(Nux_t, Nux_in, Nux_b, bc.u.left, bc.u.right, hx[1], hx[end])

        # Extend to 2D
        Su_ux = kron(sparse(I, Ny, Ny), S1D * Su_ux_bc.B1D)
        Su_ux_bc = (; Su_ux_bc..., Bbc = kron(sparse(I, Ny, Ny), S1D * Su_ux_bc.Btemp))

        ## Su_uy: evaluate uy
        diag1 = 1 ./ gyd
        S1D = spdiagm(Nuy_t - 1, Nuy_t, 0 => -diag1, 1 => diag1)

        # Boundary conditions
        Su_uy_bc = bc_diff_stag(Nuy_t, Nuy_in, Nuy_b, bc.u.low, bc.u.up, hy[1], hy[end])

        # Extend to 2D
        Su_uy = kron(S1D * Su_uy_bc.B1D, sparse(I, Nux_in, Nux_in))
        Su_uy_bc =
            (; Su_uy_bc..., Bbc = kron(S1D * Su_uy_bc.Btemp, sparse(I, Nux_in, Nux_in)))

        ## Sv_uy: evaluate vx at uy; same as Iv_uy except for mesh sizes and -diag diag
        diag1 = 1 ./ gxd
        S1D = spdiagm(Nvx_t - 1, Nvx_t, 0 => -diag1, 1 => diag1)

        # The restriction is essentially 1D so it can be directly applied to I1D
        S1D = Bvux * S1D
        S2D = kron(sparse(I, Nuy_t - 1, Nuy_t - 1), S1D)

        # Boundary conditions low/up
        Nb = Nuy_in + 1 - Nvy_in
        Sv_uy_bc_lu = bc_general(Nuy_in + 1, Nvy_in, Nb, bc.v.low, bc.v.up, hy[1], hy[end])
        Sv_uy_bc_lu =
            (; Sv_uy_bc_lu..., B2D = kron(Sv_uy_bc_lu.B1D, sparse(I, Nvx_in, Nvx_in)))
        Sv_uy_bc_lu =
            (; Sv_uy_bc_lu..., Bbc = kron(Sv_uy_bc_lu.Btemp, sparse(I, Nvx_in, Nvx_in)))

        # Boundary conditions left/right
        Sv_uy_bc_lr =
            bc_general_stag(Nvx_t, Nvx_in, Nvx_b, bc.v.left, bc.v.right, hx[1], hx[end])

        # Take I2D into left/right operators for convenience
        Sv_uy_bc_lr = (;
            Sv_uy_bc_lr...,
            B2D = S2D * kron(sparse(I, Nuy_t - 1, Nuy_t - 1), Sv_uy_bc_lr.B1D),
        )
        Sv_uy_bc_lr = (;
            Sv_uy_bc_lr...,
            Bbc = S2D * kron(sparse(I, Nuy_t - 1, Nuy_t - 1), Sv_uy_bc_lr.Btemp),
        )

        # Resulting operator:
        Sv_uy = Sv_uy_bc_lr.B2D * Sv_uy_bc_lu.B2D

        ## Diffusion operator (stress tensor), v-component: similar to averaging!

        ## Su_vx: evaluate uy at vx. Same as Iu_vx except for mesh sizes and -diag diag
        diag1 = 1 ./ gyd
        S1D = spdiagm(Nuy_t - 1, Nuy_t, 0 => -diag1, 1 => diag1)
        S1D = Buvy * S1D
        S2D = kron(S1D, sparse(I, Nvx_t - 1, Nvx_t - 1))

        # Boundary conditions low/up
        Su_vx_bc_lu =
            bc_general_stag(Nuy_t, Nuy_in, Nuy_b, bc.u.low, bc.u.up, hy[1], hy[end])
        Su_vx_bc_lu = (;
            Su_vx_bc_lu...,
            B2D = S2D * kron(Su_vx_bc_lu.B1D, sparse(I, Nvx_t - 1, Nvx_t - 1)),
        )
        Su_vx_bc_lu = (;
            Su_vx_bc_lu...,
            Bbc = S2D * kron(Su_vx_bc_lu.Btemp, sparse(I, Nvx_t - 1, Nvx_t - 1)),
        )

        # Boundary conditions left/right
        Nb = Nvx_in + 1 - Nux_in
        Su_vx_bc_lr =
            bc_general(Nvx_in + 1, Nux_in, Nb, bc.u.left, bc.u.right, hx[1], hx[end])

        Su_vx_bc_lr =
            (; Su_vx_bc_lr..., B2D = kron(sparse(I, Nuy_in, Nuy_in), Su_vx_bc_lr.B1D))
        Su_vx_bc_lr =
            (; Su_vx_bc_lr..., Bbc = kron(sparse(I, Nuy_in, Nuy_in), Su_vx_bc_lr.Btemp))

        # Resulting operator:
        Su_vx = Su_vx_bc_lu.B2D * Su_vx_bc_lr.B2D

        ## Sv_vx: evaluate vx
        diag1 = 1 ./ gxd
        S1D = spdiagm(Nvx_t - 1, Nvx_t, 0 => -diag1, 1 => diag1)

        # Boundary conditions
        Sv_vx_bc = bc_diff_stag(Nvx_t, Nvx_in, Nvx_b, bc.v.left, bc.v.right, hx[1], hx[end])

        # Extend to 2D
        Sv_vx = kron(sparse(I, Nvy_in, Nvy_in), S1D * Sv_vx_bc.B1D)

        Sv_vx_bc =
            (; Sv_vx_bc..., Bbc = kron(sparse(I, Nvy_in, Nvy_in), S1D * Sv_vx_bc.Btemp))

        ## Sv_vy: evaluate vy
        diag1 = 1 ./ hyd
        S1D = spdiagm(Nvy_t - 1, Nvy_t, 0 => -diag1, 1 => diag1)

        # Boundary conditions
        Sv_vy_bc = bc_general(Nvy_t, Nvy_in, Nvy_b, bc.v.low, bc.v.up, hy[1], hy[end])

        # Extend to 2D
        Sv_vy = kron(S1D * Sv_vy_bc.B1D, sparse(I, Nx, Nx))
        Sv_vy_bc = (; Sv_vy_bc..., Bbc = kron(S1D * Sv_vy_bc.Btemp, sparse(I, Nx, Nx)))
    end

    ## Assemble operators
    if model isa LaminarModel
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

    @pack! setup.discretization = Cux, Cuy, Cvx, Cvy
    @pack! setup.discretization = Su_ux, Su_uy
    @pack! setup.discretization = Sv_vx, Sv_vy
    @pack! setup.discretization = Su_ux_bc, Su_uy_bc, Sv_vx_bc, Sv_vy_bc
    @pack! setup.discretization = Dux, Duy, Dvx, Dvy

    if model isa LaminarModel
        @pack! setup.discretization = Diff
    else
        @pack! setup.discretization = Sv_uy, Su_vx
    end

    if order4
        @pack! setup.discretization = Cux3, Cuy3, Cvx3, Cvy3
        @pack! setup.discretization = Su_ux_bc3, Su_uy_bc3, Sv_vx_bc3, Sv_vy_bc3
        @pack! setup.discretization = Diffux_div, Diffuy_div, Diffvx_div, Diffvy_div
    else
        @pack! setup.discretization = Su_vx_bc_lr, Su_vx_bc_lu, Sv_uy_bc_lr, Sv_uy_bc_lu
    end

    setup
end
