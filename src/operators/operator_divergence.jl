"""
    operator_divergence(grid, boundary_conditions)

Construct divergence and gradient operator.
"""
function operator_divergence end

# 2D version
function operator_divergence(grid::Grid{T,2}, boundary_conditions) where {T}
    (; Npx, Npy) = grid
    (; Nux_in, Nux_b, Nux_t, Nuy_in) = grid
    (; Nvx_in, Nvy_in, Nvy_b, Nvy_t) = grid
    (; hx, hy) = grid
    (; Ω⁻¹) = grid
    (; order4, α) = grid

    bc = boundary_conditions

    if order4
        (; hxi3, hyi3) = grid
    end

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
    diagpos = 0
    bc.u.x[2] == :pressure && bc.u.x[1] == :pressure && (diagpos = 1)
    bc.u.x[2] != :pressure && bc.u.x[1] == :pressure && (diagpos = 1)
    bc.u.x[2] == :pressure && bc.u.x[1] != :pressure && (diagpos = 0)
    bc.u.x[2] == :periodic && bc.u.x[1] == :periodic && (diagpos = 1)

    BMx = spdiagm(Npx, Nux_t - 1, diagpos => ones(Npx))
    M1D = BMx * M1D

    # Extension to 2D to be used in post-processing files
    Bup = I(Nuy_in) ⊗ BMx

    # Boundary conditions
    Mx_bc = bc_general(Nux_t, Nux_in, Nux_b, bc.u.x[1], bc.u.x[2], hx[1], hx[end])
    Mx_bc = (; Mx_bc..., Bbc = mat_hy ⊗ (M1D * Mx_bc.Btemp))

    # Extend to 2D
    Mx = mat_hy ⊗ (M1D * Mx_bc.B1D)

    if order4
        mat_hy3 = Diagonal(hyi3)
        diag1 = ones(Nux_t - 1)
        M1D3 = spdiagm(Nux_t - 1, Nux_t + 2, 0 => -diag1, 3 => diag1)
        M1D3 = BMx * M1D3
        Mx_bc3 = bc_div2(
            Nux_t + 2,
            Nux_in,
            Nux_t + 2 - Nux_in,
            bc.u.x[1],
            bc.u.x[2],
            hx[1],
            hx[end],
        )
        Mx3 = mat_hy3 ⊗ (M1D3 * Mx_bc3.B1D)
        Mx_bc3 = (; Mx_bc3..., Bbc = mat_hy3 ⊗ (M1D3 * Mx_bc3.Btemp))
    end

    ## My (same as Mx but reversing indices and kron arguments)
    diag1 = ones(Nvy_t - 1)
    M1D = spdiagm(Nvy_t - 1, Nvy_t, 0 => -diag1, 1 => diag1)

    # We only need derivative at inner pressure points, so we map the resulting
    # boundary matrix (restriction)
    diagpos = 0
    bc.v.y[2] == :pressure && bc.v.y[1] == :pressure && (diagpos = 1)
    bc.v.y[2] != :pressure && bc.v.y[1] == :pressure && (diagpos = 1)
    bc.v.y[2] == :pressure && bc.v.y[1] != :pressure && (diagpos = 0)
    bc.v.y[2] == :periodic && bc.v.y[1] == :periodic && (diagpos = 1)

    BMy = spdiagm(Npy, Nvy_t - 1, diagpos => ones(Npy))
    M1D = BMy * M1D

    # Extension to 2D to be used in post-processing files
    Bvp = BMy ⊗ I(Nvx_in)

    # Boundary conditions
    My_bc = bc_general(Nvy_t, Nvy_in, Nvy_b, bc.v.y[1], bc.v.y[2], hy[1], hy[end])
    My_bc = (; My_bc..., Bbc = (M1D * My_bc.Btemp) ⊗ mat_hx)

    # Extend to 2D
    My = (M1D * My_bc.B1D) ⊗ mat_hx

    if order4
        mat_hx3 = Diagonal(hxi3)
        diag1 = ones(Nvy_t - 1)
        M1D3 = spdiagm(Nvy_t - 1, Nvy_t + 2, 0 => -diag1, 3 => diag1)
        M1D3 = BMy * M1D3
        My_bc3 = bc_div2(
            Nvy_t + 2,
            Nvy_in,
            Nvy_t + 2 - Nvy_in,
            bc.v.y[1],
            bc.v.y[2],
            hy[1],
            hy[end],
        )
        My3 = (M1D3 * My_bc3.B1D) ⊗ mat_hx3
        My_bc3 = (; My_bc3..., Bbc = (M1D3 * My_bc3.Btemp) ⊗ mat_hx3)
    end

    ## Resulting divergence matrix
    if order4
        Mx = α * Mx - Mx3
        My = α * My - My3
    end
    M = [Mx My]


    ## Gradient operator G

    # Like in the continuous case, grad = -div^T
    # Note that this also holds for outflow boundary conditions, if the stress
    # on the ouflow boundary is properly taken into account in y_p (often this
    # stress will be zero)
    G = -M'


    ## Pressure matrix for pressure correction method;
    # Also used to make initial data divergence free or compute additional poisson solve
    # Note that the matrix for the pressure is constant in time.
    # Only the right hand side vector changes, so the pressure matrix can be set up outside the time-stepping-loop.

    # Laplace = div grad
    A = M * Diagonal(Ω⁻¹) * G

    # Check if all the row sums of the pressure matrix are zero, which
    # should be the case if there are no pressure boundary conditions
    if all(≠(:pressure), (bc.u.x..., bc.v.y...))
         if any(≉(0; atol = 1e-10), sum(A; dims = 2))
            @warn "Pressure matrix: not all rowsums are zero!"
        end
    end

    ## Group operators
    operators = (; M, Mx_bc, My_bc, G, Bup, Bvp, A)

    if order4
        operators = (; operators..., Mx3, My3, Mx_bc3, My_bc3)
    end

    operators
end

# 3D version
function operator_divergence(grid::Grid{T,3}, boundary_conditions) where {T}
    (; Nux_in, Nux_b, Nux_t, Nuy_in, Nuz_in) = grid
    (; Nvx_in, Nvy_in, Nvy_b, Nvy_t, Nvz_in) = grid
    (; Nwx_in, Nwy_in, Nwz_in, Nwz_b, Nwz_t) = grid
    (; Npx, Npy, Npz) = grid
    (; hx, hy, hz) = grid
    (; Ω⁻¹) = grid

    bc = boundary_conditions

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
    diagpos = 0
    bc.u.x[2] == :pressure && bc.u.x[1] == :pressure && (diagpos = 1)
    bc.u.x[2] != :pressure && bc.u.x[1] == :pressure && (diagpos = 1)
    bc.u.x[2] == :pressure && bc.u.x[1] != :pressure && (diagpos = 0)
    bc.u.x[2] == :periodic && bc.u.x[1] == :periodic && (diagpos = 1)
    BMx = spdiagm(Npx, Nux_t - 1, diagpos => ones(Npx))
    M1D = BMx * M1D

    # Extension to 3D to be used in post-processing files
    Bup = I(Nuz_in) ⊗ I(Nuy_in) ⊗ BMx

    # Boundary conditions
    Mx_bc = bc_general(Nux_t, Nux_in, Nux_b, bc.u.x[1], bc.u.x[2], hx[1], hx[end])
    Mx_bc = (; Mx_bc..., Bbc = mat_hz ⊗ mat_hy ⊗ (M1D * Mx_bc.Btemp))

    # Extend to 3D
    Mx = mat_hz ⊗ mat_hy ⊗ (M1D * Mx_bc.B1D)

    ## My
    # Same as Mx but reversing indices and kron arguments
    diag1 = ones(Nvy_t - 1)
    M1D = spdiagm(Nvy_t - 1, Nvy_t, 0 => -diag1, 1 => diag1)

    # We only need derivative at inner pressure points, so we map the resulting
    # Boundary matrix (restriction)
    diagpos = 0
    bc.v.y[2] == :pressure && bc.v.y[1] == :pressure && (diagpos = 1)
    bc.v.y[2] != :pressure && bc.v.y[1] == :pressure && (diagpos = 1)
    bc.v.y[2] == :pressure && bc.v.y[1] != :pressure && (diagpos = 0)
    bc.v.y[2] == :periodic && bc.v.y[1] == :periodic && (diagpos = 1)
    BMy = spdiagm(Npy, Nvy_t - 1, diagpos => ones(Npy))
    M1D = BMy * M1D

    # Extension to 3D to be used in post-processing files
    Bvp = I(Nvz_in) ⊗ BMy ⊗ I(Nvx_in)

    # Boundary conditions
    My_bc = bc_general(Nvy_t, Nvy_in, Nvy_b, bc.v.y[1], bc.v.y[2], hy[1], hy[end])
    My_bc = (; My_bc..., Bbc = mat_hz ⊗ (M1D * My_bc.Btemp) ⊗ mat_hx)

    # Extend to 3D
    My = mat_hz ⊗ (M1D * My_bc.B1D) ⊗ mat_hx

    ## Mz
    # Same as Mx but reversing indices and kron arguments
    diag1 = ones(Nwz_t - 1)
    M1D = spdiagm(Nwz_t - 1, Nwz_t, 0 => -diag1, 1 => diag1)

    # We only need derivative at inner pressure points, so we map the resulting
    # Boundary matrix (restriction)
    diagpos = 0
    bc.w.z[1] == :pressure && bc.w.z[2] == :pressure && (diagpos = 1)
    bc.w.z[1] == :pressure && bc.w.z[2] != :pressure && (diagpos = 1)
    bc.w.z[1] != :pressure && bc.w.z[2] == :pressure && (diagpos = 0)
    bc.w.z[1] == :periodic && bc.w.z[2] == :periodic && (diagpos = 1)

    BMz = spdiagm(Npz, Nwz_t - 1, diagpos => ones(Npz))
    M1D = BMz * M1D

    # Extension to 3D to be used in post-processing files
    Bwp = BMz ⊗ I(Nwy_in) ⊗ I(Nwx_in)

    # Boundary conditions
    Mz_bc = bc_general(Nwz_t, Nwz_in, Nwz_b, bc.w.z[1], bc.w.z[2], hz[1], hz[end])
    Mz_bc = (; Mz_bc..., Bbc = (M1D * Mz_bc.Btemp) ⊗ mat_hy ⊗ mat_hx)

    # Extend to 3D
    Mz = (M1D * Mz_bc.B1D) ⊗ mat_hy ⊗ mat_hx

    ## Resulting divergence matrix
    M = [Mx My Mz]

    ## Gradient operator G

    # Like in the continuous case, grad = -div^T
    # Note that this also holds for outflow boundary conditions, if the stress
    # on the ouflow boundary is properly taken into account in y_p (often this
    # stress will be zero)
    G = -M'

    ## Pressure matrix for pressure correction method;
    # Also used to make initial data divergence free or compute additional poisson solve
    # if !is_steady(problem) && !isa(viscosity_model, KEpsilonModel)
    # Note that the matrix for the pressure is constant in time.
    # Only the right hand side vector changes, so the pressure matrix can be set up
    # outside the time-stepping-loop.

    # Laplace = div grad
    A = M * Diagonal(Ω⁻¹) * G

    # Check if all the row sums of the pressure matrix are zero, which
    # should be the case if there are no pressure boundary conditions
    if all(≠(:pressure), (bc.u.x..., bc.v.y..., bc.w.z...))
         if any(≉(0; atol = 1e-10), sum(A; dims = 2))
            @warn "Pressure matrix: not all rowsums are zero!"
        end
    end

    ## Group operators
    (; M, Mx_bc, My_bc, Mz_bc, G, Bup, Bvp, Bwp, A)
end
