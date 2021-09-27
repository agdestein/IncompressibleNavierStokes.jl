function operator_divergence!(setup)
    # construct divergence and gradient operator

    # boundary conditions
    BC = setup.BC

    # number of interior points and boundary points
    @unpack Nx, Ny, Npx, Npy = setup.grid
    @unpack Nux_in, Nux_b, Nux_t, Nuy_in, Nuy_b, Nuy_t = setup.grid
    @unpack Nvx_in, Nvx_b, Nvx_t, Nvy_in, Nvy_b, Nvy_t = setup.grid
    @unpack Nu, Nv, Np = setup.grid
    @unpack hx, hy = setup.grid
    @unpack Om_inv = setup.grid

    order4 = setup.discretization.order4

    if order4
        α = setup.discretization.α
        hxi3 = setup.grid.hxi3
        hyi3 = setup.grid.hyi3
    end

    steady = setup.case.steady
    visc = setup.case.visc

    ## Divergence operator M

    # note that the divergence matrix M is not square
    mat_hx = spdiagm(Nx, Nx, hx)
    mat_hy = spdiagm(Ny, Ny, hy)

    # for fourth order: mat_hx3 is defined in operator_interpolation

    ## Mx
    # building blocks consisting of diagonal matrices where the diagonal is
    # equal to constant per block (hy(block)) and changing for next block to
    # hy(block+1)
    diag1 = ones(Nux_t)
    M1D = spdiagm(Nux_t - 1, Nux_t, 0 => -diag1, 1 => diag1)

    # we only need derivative at inner pressure points, so we map the resulting
    # boundary matrix (restrict)
    diagpos = 0
    if BC.u.right == "pres" && BC.u.left == "pres"
        diagpos = 1
    end
    if BC.u.right != "pres" && BC.u.left == "pres"
        diagpos = 1
    end
    if BC.u.right == "pres" && BC.u.left != "pres"
        diagpos = 0
    end
    if BC.u.right == "per" && BC.u.left == "per"
        # like pressure left
        diagpos = 1
    end

    BMx = spdiagm(Npx, Nux_t - 1, diagpos => ones(Npx))
    M1D = BMx * M1D

    # extension to 2D to be used in post-processing files
    Bup = kron(sparse(I, Nuy_in, Nuy_in), BMx)

    # boundary conditions
    Mx_BC = BC_general(Nux_t, Nux_in, Nux_b, BC.u.left, BC.u.right, hx[1], hx[end])
    Mx_BC.Bbc = kron(mat_hy, M1D * Mx_BC.Btemp)

    # extend to 2D
    Mx = kron(mat_hy, M1D * Mx_BC.B1D)

    if order4
        mat_hy3 = spdiagm(hyi3, 0, Ny, Ny)
        diag1 = ones(Nux_t + 1, 1)
        M1D3 = spdiagm([-diag1 diag1], [0 3], Nux_t - 1, Nux_t + 2)
        M1D3 = BMx * M1D3
        Mx_BC3 = BC_div2(
            Nux_t + 2,
            Nux_in,
            Nux_t + 2 - Nux_in,
            BC.u.left,
            BC.u.right,
            hx[1],
            hx[end],
        )
        Mx3 = kron(mat_hy3, M1D3 * Mx_BC3.B1D)
        Mx_BC3.Bbc = kron(mat_hy3, M1D3 * Mx_BC3.Btemp)
    end

    ## My
    # same as Mx but reversing indices and kron arguments
    diag1 = ones(Nvy_t, 1)
    M1D = spdiagm([-diag1 diag1], [0 1], Nvy_t - 1, Nvy_t)

    # we only need derivative at inner pressure points, so we map the resulting
    # boundary matrix (restriction)
    diagpos = 0
    if BC.v.up == "pres" && BC.v.low == "pres"
        diagpos = 1
    end
    if BC.v.up != "pres" && BC.v.low == "pres"
        diagpos = 1
    end
    if BC.v.up == "pres" && BC.v.low != "pres"
        diagpos = 0
    end
    if BC.v.up == "per" && BC.v.low == "per"
        # like pressure low
        diagpos = 1
    end

    BMy = spdiagm(Npy, Nvy_t - 1, diagpos => ones(Npy))
    M1D = BMy * M1D
    # extension to 2D to be used in post-processing files
    Bvp = kron(BMy, sparse(I, Nvx_in, Nvx_in))

    # boundary conditions
    My_BC = BC_general(Nvy_t, Nvy_in, Nvy_b, BC.v.low, BC.v.up, hy[1], hy[end])
    My_BC.Bbc = kron(M1D * My_BC.Btemp, mat_hx)

    # extend to 2D
    My = kron(M1D * My_BC.B1D, mat_hx)

    if order4
        mat_hx3 = spdiagm(Nx, Nx, hxi3)
        diag1 = ones(Nvy_t + 1)
        M1D3 = spdiagm(Nvy_t - 1, Nvy_t + 2, 0 => -diag1, 3 => diag1)
        M1D3 = BMy * M1D3
        My_BC3 = BC_div2(
            Nvy_t + 2,
            Nvy_in,
            Nvy_t + 2 - Nvy_in,
            BC.v.low,
            BC.v.up,
            hy[1],
            hy[end],
        )
        My3 = kron(M1D3 * My_BC3.B1D, mat_hx3)
        My_BC3.Bbc = kron(M1D3 * My_BC3.Btemp, mat_hx3)
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
    setup.discretization.Mx_BC = Mx_BC
    setup.discretization.My_BC = My_BC
    setup.discretization.G = G
    setup.discretization.Gx = Gx
    setup.discretization.Gy = Gy

    setup.discretization.Bup = Bup
    setup.discretization.Bvp = Bvp

    if order4
        setup.discretization.Mx3 = Mx3
        setup.discretization.My3 = My3
        setup.discretization.Mx_BC3 = Mx_BC3
        setup.discretization.My_BC3 = My_BC3
    end

    ## Pressure matrix for pressure correction method;
    # also used to make initial data divergence free or compute additional poisson solve

    if !steady && visc != "keps"
        poisson = setup.solversettings.poisson

        #   Note that the matrix for the pressure is constant in time.
        #   Only the right hand side vector changes, so the pressure matrix
        #   can be set up outside the time-stepping-loop.

        #   Laplace = div grad
        A = M * spdiagm(Nu + Nv, Nu + Nv, Om_inv) * G
        setup.discretization.A = A

        # ROM does not require Poisson solve for simple BC
        # for rom_bc>0, we need Poisson solve to determine the V_bc field
        if setup.rom.use_rom && setup.rom.rom_bc == 0 && setup.rom.rom_type == "POD"
            return setup
        end

        # LU decomposition
        if poisson == 3
            if exist(["cg." mexext], "file") == 3
                [B, d] = spdiagm(A)
                ndia = (length(d) + 1) / 2
                dia = d[ndia:end]
                B = B[:, ndia:-1:1]

                setup.solversettings.ndia = ndia
                setup.solversettings.dia = dia
                setup.solversettings.B = B
            else
                println(
                    "No correct CG mex file available, switching to Matlab implementation",
                )
                poisson = 4
            end
        end
        if poisson == 4
            # preconditioner
            A_pc = make_cholinc(A)
            setup.solversettings.A_pc = A_pc
        end
        if poisson == 2
            setup.type = "nofill"
            [L, U] = ilu(A, setup)
            setup.solversettings.L = L
            setup.solversettings.U = U
        end
        if poisson == 1
            setup.solversettings.decomp = factorize(A)
        end
        if poisson == 5
            # open socket only once
            # system("petscmpiexec -n 2 ./solvers/petsc_poisson_par -viewer_socket_port 5600 -pc_type hypre -pc_hypre_type boomeramg &";
            PS = PetscOpenSocket(5600)
            PetscBinaryWrite(PS, -A)
        end
        if poisson == 6
            if BC.v.low == "per" &&
               BC.v.up == "per" &&
               BC.u.left == "per" &&
               BC.u.left == "per"
                tol = 1e-14
                if maximum(abs.(diff(hx))) > tol || maximum(abs.(diff(hy))) > tol
                    error(
                        "grid needs to be uniform to use Fourier transform for pressure matrix",
                    )
                else
                    dx = hx[1]
                    dy = hy[1]

                    # Fourier transform of the discretization; assuming uniform
                    # grid, although dx, dy and dz do not need to be the same
                    [I, J] = ndgrid(0:(Npx-1), 0:(Npy-1))
                    # scale with dx*dy*dz, since we solve the PPE in integrated
                    # form
                    Ahat =
                        (dx * dy) *
                        4 *
                        (
                            (1 / (dx^2)) * sin(I * pi / Npx) .^ 2 +
                            (1 / (dy^2)) * sin(J * pi / Npy) .^ 2
                        )
                    Ahat[1, 1] = 1 # pressure is determined up to constant, fix at 0
                    setup.solversettings.Ahat = Ahat
                end
            else
                error(
                    "Fourier transform for pressure Poisson only implemented for periodic boundary conditions",
                )
            end
        end

        # check if all the row sums of the pressure matrix are zero, which
        # should be the case if there are no pressure boundary conditions
        if BC.v.low != "pres" &&
           BC.v.up != "pres" &&
           BC.u.right != "pres" &&
           BC.u.left != "pres"
            if max(abs(A * ones(Np, 1))) > 1e-10
                @warn "Pressure matrix: not all rowsums are zero!"
            end
        end
    end

    setup
end
