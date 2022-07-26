"""
    operator_divergence!(setup)

Construct divergence and gradient operator.
"""
function operator_divergence! end

# 2D version
function operator_divergence!(setup::Setup{T,2}) where {T}
    (; grid, operators, pressure_solver) = setup
    (; Npx, Npy) = grid
    (; Nux_in, Nux_b, Nux_t, Nuy_in) = grid
    (; Nvx_in, Nvy_in, Nvy_b, Nvy_t) = grid
    (; hx, hy) = grid
    (; Ω⁻¹) = grid

    ## Divergence operator M

    # Note that the divergence matrix M is not square
    mat_hx = Diagonal(hx)
    mat_hy = Diagonal(hy)

    # For fourth order: mat_hx3 is defined in operator_interpolation

    ## Mx
    # Building blocks consisting of diagonal matrices where the diagonal is
    # equal to constant per block (hy(block)) and changing for next block to
    # hy(block+1)
    diag1 = ones(Nux_t - 1)
    M1D = spdiagm(Nux_t - 1, Nux_t, 0 => -diag1, 1 => diag1)

    # We only need derivative at inner pressure points, so we map the resulting
    # boundary matrix (restrict)
    diagpos = 1

    BMx = spdiagm(Npx, Nux_t - 1, diagpos => ones(Npx))
    M1D = BMx * M1D

    # Extension to 2D to be used in post-processing files
    Bup = I(Nuy_in) ⊗ BMx

    # Boundary conditions
    Mx_bc = bc_general(Nux_t, Nux_in, Nux_b, hx[1], hx[end])
    Mx_bc = (; Mx_bc..., Bbc = mat_hy ⊗ (M1D * Mx_bc.Btemp))

    # Extend to 2D
    Mx = mat_hy ⊗ (M1D * Mx_bc.B1D)

    ## My (same as Mx but reversing indices and kron arguments)
    diag1 = ones(Nvy_t - 1)
    M1D = spdiagm(Nvy_t - 1, Nvy_t, 0 => -diag1, 1 => diag1)

    # We only need derivative at inner pressure points, so we map the resulting
    # boundary matrix (restriction)
    diagpos = 1

    BMy = spdiagm(Npy, Nvy_t - 1, diagpos => ones(Npy))
    M1D = BMy * M1D

    # Extension to 2D to be used in post-processing files
    Bvp = BMy ⊗ I(Nvx_in)

    # Boundary conditions
    My_bc = bc_general(Nvy_t, Nvy_in, Nvy_b, hy[1], hy[end])
    My_bc = (; My_bc..., Bbc = (M1D * My_bc.Btemp) ⊗ mat_hx)

    # Extend to 2D
    My = (M1D * My_bc.B1D) ⊗ mat_hx

    ## Resulting divergence matrix
    M = [Mx My]


    ## Gradient operator G

    # Like in the continuous case, grad = -div^T
    # Note that this also holds for outflow boundary conditions
    G = -M'

    ## Store in setup structure
    @pack! operators = M, G
    @pack! operators = Bup, Bvp

    ## Pressure matrix for pressure correction method;
    # Also used to make initial data divergence free or compute additional poisson solve
    # Note that the matrix for the pressure is constant in time.
    # Only the right hand side vector changes, so the pressure matrix can be set up outside the time-stepping-loop.

    # Laplace = div grad
    A = M * Diagonal(Ω⁻¹) * G
    @pack! operators = A

    initialize!(pressure_solver, setup, A)

    # Check if all the row sums of the pressure matrix are zero
    if any(≉(0; atol = 1e-10), sum(A; dims = 2))
        @warn "Pressure matrix: not all rowsums are zero!"
    end

    setup
end

# 3D version
function operator_divergence!(setup::Setup{T,3}) where {T}
    (; grid, operators, pressure_solver) = setup
    (; Nux_in, Nux_b, Nux_t, Nuy_in, Nuz_in) = grid
    (; Nvx_in, Nvy_in, Nvy_b, Nvy_t, Nvz_in) = grid
    (; Nwx_in, Nwy_in, Nwz_in, Nwz_b, Nwz_t) = grid
    (; Npx, Npy, Npz) = grid
    (; hx, hy, hz) = grid
    (; Ω⁻¹) = grid

    ## Divergence operator M

    # Note that the divergence matrix M is not square
    mat_hx = Diagonal(hx)
    mat_hy = Diagonal(hy)
    mat_hz = Diagonal(hz)

    ## Mx
    # Building blocks consisting of diagonal matrices where the diagonal is
    # Equal to constant per block (hy(block)) and changing for next block to
    # Hy(block+1)
    diag1 = ones(Nux_t - 1)
    M1D = spdiagm(Nux_t - 1, Nux_t, 0 => -diag1, 1 => diag1)

    # We only need derivative at inner pressure points, so we map the resulting
    # Boundary matrix (restrict)
    diagpos = 1
    BMx = spdiagm(Npx, Nux_t - 1, diagpos => ones(Npx))
    M1D = BMx * M1D

    # Extension to 3D to be used in post-processing files
    Bup = I(Nuz_in) ⊗ I(Nuy_in) ⊗ BMx

    # Boundary conditions
    Mx_bc = bc_general(Nux_t, Nux_in, Nux_b, hx[1], hx[end])
    Mx_bc = (; Mx_bc..., Bbc = mat_hz ⊗ mat_hy ⊗ (M1D * Mx_bc.Btemp))

    # Extend to 3D
    Mx = mat_hz ⊗ mat_hy ⊗ (M1D * Mx_bc.B1D)

    ## My
    # Same as Mx but reversing indices and kron arguments
    diag1 = ones(Nvy_t - 1)
    M1D = spdiagm(Nvy_t - 1, Nvy_t, 0 => -diag1, 1 => diag1)

    # We only need derivative at inner pressure points, so we map the resulting
    # Boundary matrix (restriction)
    diagpos = 1
    BMy = spdiagm(Npy, Nvy_t - 1, diagpos => ones(Npy))
    M1D = BMy * M1D

    # Extension to 3D to be used in post-processing files
    Bvp = I(Nvz_in) ⊗ BMy ⊗ I(Nvx_in)

    # Boundary conditions
    My_bc = bc_general(Nvy_t, Nvy_in, Nvy_b, hy[1], hy[end])
    My_bc = (; My_bc..., Bbc = mat_hz ⊗ (M1D * My_bc.Btemp) ⊗ mat_hx)

    # Extend to 3D
    My = mat_hz ⊗ (M1D * My_bc.B1D) ⊗ mat_hx

    ## Mz
    # Same as Mx but reversing indices and kron arguments
    diag1 = ones(Nwz_t - 1)
    M1D = spdiagm(Nwz_t - 1, Nwz_t, 0 => -diag1, 1 => diag1)

    # We only need derivative at inner pressure points, so we map the resulting
    # Boundary matrix (restriction)
    diagpos = 1

    BMz = spdiagm(Npz, Nwz_t - 1, diagpos => ones(Npz))
    M1D = BMz * M1D

    # Extension to 3D to be used in post-processing files
    Bwp = BMz ⊗ I(Nwy_in) ⊗ I(Nwx_in)

    # Boundary conditions
    Mz_bc = bc_general(Nwz_t, Nwz_in, Nwz_b, hz[1], hz[end])
    Mz_bc = (; Mz_bc..., Bbc = (M1D * Mz_bc.Btemp) ⊗ mat_hy ⊗ mat_hx)

    # Extend to 3D
    Mz = (M1D * Mz_bc.B1D) ⊗ mat_hy ⊗ mat_hx

    ## Resulting divergence matrix
    M = [Mx My Mz]

    ## Gradient operator G

    # Like in the continuous case, grad = -div^T
    # Note that this also holds for outflow boundary conditions
    G = -M'

    ## Store in setup structure
    @pack! operators = M, G
    @pack! operators = Bup, Bvp, Bwp

    ## Pressure matrix for pressure correction method;
    # Also used to make initial data divergence free or compute additional
    # poisson solve.
    # Note that the matrix for the pressure is constant in time.
    # Only the right hand side vector changes, so the pressure matrix can be set up
    # outside the time-stepping-loop.

    # Laplace = div grad
    A = M * Diagonal(Ω⁻¹) * G
    @pack! operators = A

    initialize!(pressure_solver, setup, A)

    # Check if all the row sums of the pressure matrix are zero
    if any(≉(0; atol = 1e-10), sum(A; dims = 2))
        @warn "Pressure matrix: not all rowsums are zero!"
    end

    setup
end
