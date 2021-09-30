"""
Construct divergence and gradient operator
"""
function operator_divergence!(setup)

    # boundary conditions
    bc = setup.bc

    # number of interior points and boundary points
    @unpack Nx, Ny, Npx, Npy = setup.grid
    @unpack Nux_in, Nux_b, Nux_t, Nuy_in, Nuy_b, Nuy_t = setup.grid
    @unpack Nvx_in, Nvx_b, Nvx_t, Nvy_in, Nvy_b, Nvy_t = setup.grid
    @unpack Nu, Nv, Np = setup.grid
    @unpack hx, hy = setup.grid
    @unpack Ω⁻¹ = setup.grid

    order4 = setup.discretization.order4

    if order4
        α = setup.discretization.α
        hxi3 = setup.grid.hxi3
        hyi3 = setup.grid.hyi3
    end

    is_steady = setup.case.is_steady
    visc = setup.case.visc

    ## Divergence operator M

    # note that the divergence matrix M is not square
    mat_hx = spdiagm(hx)
    mat_hy = spdiagm(hy)

    # for fourth order: mat_hx3 is defined in operator_interpolation

    ## Mx
    # building blocks consisting of diagonal matrices where the diagonal is
    # equal to constant per block (hy(block)) and changing for next block to
    # hy(block+1)
    diag1 = ones(Nux_t - 1)
    M1D = spdiagm(Nux_t - 1, Nux_t, 0 => -diag1, 1 => diag1)

    # we only need derivative at inner pressure points, so we map the resulting
    # boundary matrix (restrict)
    diagpos = 0
    if bc.u.right == "pres" && bc.u.left == "pres"
        diagpos = 1
    end
    if bc.u.right != "pres" && bc.u.left == "pres"
        diagpos = 1
    end
    if bc.u.right == "pres" && bc.u.left != "pres"
        diagpos = 0
    end
    if bc.u.right == "per" && bc.u.left == "per"
        # like pressure left
        diagpos = 1
    end

    BMx = spdiagm(Npx, Nux_t - 1, diagpos => ones(Npx))
    M1D = BMx * M1D

    # extension to 2D to be used in post-processing files
    Bup = kron(sparse(I, Nuy_in, Nuy_in), BMx)

    # boundary conditions
    Mx_bc = bc_general(Nux_t, Nux_in, Nux_b, bc.u.left, bc.u.right, hx[1], hx[end])
    Mx_bc = (; Mx_bc..., Bbc = kron(mat_hy, M1D * Mx_bc.Btemp))

    # extend to 2D
    Mx = kron(mat_hy, M1D * Mx_bc.B1D)

    if order4
        mat_hy3 = spdiagm(hyi3)
        diag1 = ones(Nux_t - 1)
        M1D3 = spdiagm(Nux_t - 1, Nux_t + 2, 0 => -diag1, 3 => diag1)
        M1D3 = BMx * M1D3
        Mx_bc3 = bc_div2(
            Nux_t + 2,
            Nux_in,
            Nux_t + 2 - Nux_in,
            bc.u.left,
            bc.u.right,
            hx[1],
            hx[end],
        )
        Mx3 = kron(mat_hy3, M1D3 * Mx_bc3.B1D)
        Mx_bc3 = (; Mx_bc3..., Bbc = kron(mat_hy3, M1D3 * Mx_bc3.Btemp))
    end

    ## My
    # same as Mx but reversing indices and kron arguments
    diag1 = ones(Nvy_t - 1)
    M1D = spdiagm(Nvy_t - 1, Nvy_t, 0 => -diag1, 1 => diag1)

    # we only need derivative at inner pressure points, so we map the resulting
    # boundary matrix (restriction)
    diagpos = 0
    if bc.v.up == "pres" && bc.v.low == "pres"
        diagpos = 1
    end
    if bc.v.up != "pres" && bc.v.low == "pres"
        diagpos = 1
    end
    if bc.v.up == "pres" && bc.v.low != "pres"
        diagpos = 0
    end
    if bc.v.up == "per" && bc.v.low == "per"
        # like pressure low
        diagpos = 1
    end

    BMy = spdiagm(Npy, Nvy_t - 1, diagpos => ones(Npy))
    M1D = BMy * M1D
    # extension to 2D to be used in post-processing files
    Bvp = kron(BMy, sparse(I, Nvx_in, Nvx_in))

    # boundary conditions
    My_bc = bc_general(Nvy_t, Nvy_in, Nvy_b, bc.v.low, bc.v.up, hy[1], hy[end])
    My_bc = (; My_bc..., Bbc = kron(M1D * My_bc.Btemp, mat_hx))

    # extend to 2D
    My = kron(M1D * My_bc.B1D, mat_hx)

    if order4
        mat_hx3 = spdiagm(Nx, Nx, hxi3)
        diag1 = ones(Nvy_t - 1)
        M1D3 = spdiagm(Nvy_t - 1, Nvy_t + 2, 0 => -diag1, 3 => diag1)
        M1D3 = BMy * M1D3
        My_bc3 = bc_div2(
            Nvy_t + 2,
            Nvy_in,
            Nvy_t + 2 - Nvy_in,
            bc.v.low,
            bc.v.up,
            hy[1],
            hy[end],
        )
        My3 = kron(M1D3 * My_bc3.B1D, mat_hx3)
        My_bc3 = (; My_bc3..., Bbc = kron(M1D3 * My_bc3.Btemp, mat_hx3))
    end

    ## resulting divergence matrix
    if order4
        Mx = α * Mx - Mx3
        My = α * My - My3
    end
    M = [Mx My]

    ## Gradient operator G

    # like in the continuous case, grad = -div^T
    # note that this also holds for outflow boundary conditions, if the stress
    # on the ouflow boundary is properly taken into account in y_p (often this
    # stress will be zero)
    Gx = -Mx'
    Gy = -My'

    G = [Gx; Gy]

    ## store in setup structure
    setup.discretization.M = M
    setup.discretization.Mx = Mx
    setup.discretization.My = My
    setup.discretization.Mx_bc = Mx_bc
    setup.discretization.My_bc = My_bc
    setup.discretization.G = G
    setup.discretization.Gx = Gx
    setup.discretization.Gy = Gy

    setup.discretization.Bup = Bup
    setup.discretization.Bvp = Bvp

    if order4
        setup.discretization.Mx3 = Mx3
        setup.discretization.My3 = My3
        setup.discretization.Mx_bc3 = Mx_bc3
        setup.discretization.My_bc3 = My_bc3
    end

    ## Pressure matrix for pressure correction method;
    # also used to make initial data divergence free or compute additional poisson solve
    if !is_steady && visc != "keps"
        # Note that the matrix for the pressure is constant in time.
        # Only the right hand side vector changes, so the pressure matrix
        # can be set up outside the time-stepping-loop.

        # Laplace = div grad
        A = M * spdiagm(Ω⁻¹) * G
        setup.discretization.A = A

        # ROM does not require Poisson solve for simple BC
        # for rom_bc > 0, we need Poisson solve to determine the V_bc field
        if setup.rom.use_rom && setup.rom.rom_bc == 0 && setup.rom.rom_type == "POD"
            return setup
        end

        # LU decomposition
        setup.discretization.A_fact = factorize(A)

        # check if all the row sums of the pressure matrix are zero, which
        # should be the case if there are no pressure boundary conditions
        if bc.v.low != "pres" &&
           bc.v.up != "pres" &&
           bc.u.right != "pres" &&
           bc.u.left != "pres"
            if maximum(abs.(A * ones(Np))) > 1e-10
                @warn "Pressure matrix: not all rowsums are zero!"
            end
        end
    end

    setup
end
