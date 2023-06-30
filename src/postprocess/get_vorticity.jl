"""
    get_vorticity(setup, V, t)

Get vorticity from velocity field.
"""
function get_vorticity end

get_vorticity(setup, V, t) = get_vorticity(setup.grid.dimension, setup, V, t)

# 2D version
function get_vorticity(::Dimension{2}, setup, V, t)
    (; grid, boundary_conditions) = setup
    (; Nx, Ny) = grid

    if all(==(:periodic), (boundary_conditions.u.x[1], boundary_conditions.v.y[1]))
        Nωx = Nx + 1
        Nωy = Ny + 1
    else
        Nωx = Nx - 1
        Nωy = Ny - 1
    end

    ω = zeros(Nωx, Nωy)

    vorticity!(ω, setup, V, t)
end

# 3D version
function get_vorticity(::Dimension{3}, setup, V, t)
    (; grid, boundary_conditions) = setup
    (; Nx, Ny, Nz) = grid

    if all(
        ==(:periodic),
        (
            boundary_conditions.u.x[1],
            boundary_conditions.v.y[1],
            boundary_conditions.w.z[1],
        ),
    )
        Nωx = Nx + 1
        Nωy = Ny + 1
        Nωz = Nz + 1
    else
        Nωx = Nx - 1
        Nωy = Ny - 1
        Nωz = Nz - 1
    end

    ω = zeros(Nωx, Nωy, Nωz)

    vorticity!(ω, setup, V, t)
end

"""
    vorticity!(ω, V, t, setup)

Compute vorticity values at pressure midpoints.
This should be consistent with `operator_postprocessing.jl`.
"""
function vorticity! end

vorticity!(ω, setup, V, t) = vorticity!(setup.grid.dimension, ω, setup, V, t)

# 2D version
function vorticity!(::Dimension{2}, ω, setup, V, t)
    (; grid, operators, boundary_conditions) = setup
    (; indu, indv, Nux_in, Nvy_in, Nx, Ny) = grid
    (; Wv_vx, Wu_uy) = operators

    uₕ = @view V[indu]
    vₕ = @view V[indv]
    ω_flat = reshape(ω, length(ω))

    if boundary_conditions.u.x[1] == :periodic && boundary_conditions.v.y[1] == :periodic
        uₕ_in = uₕ
        vₕ_in = vₕ
    else
        # Velocity at inner points
        diagpos = 0
        boundary_conditions.u.x[1] == :pressure && (diagpos = 1)
        boundary_conditions.u.x[1] == :periodic && (diagpos = 1)
        B1D = spdiagm(Nx - 1, Nux_in, diagpos => ones(Nx - 1))
        B2D = I(Ny) ⊗ B1D
        uₕ_in = B2D * uₕ

        diagpos = 0
        boundary_conditions.v.y[1] == :pressure && (diagpos = 1)
        boundary_conditions.v.y[1] == :periodic && (diagpos = 1)
        B1D = spdiagm(Ny - 1, Nvy_in, diagpos => ones(Ny - 1))
        B2D = B1D ⊗ I(Nx)
        vₕ_in = B2D * vₕ
    end

    # ω_flat .= Wv_vx * vₕ_in - Wu_uy * uₕ_in
    mul!(ω_flat, Wv_vx, vₕ_in) # a = b * c
    mul!(ω_flat, Wu_uy, uₕ_in, -1, 1) # a = -b * c + a

    ω
end

# 3D version
function vorticity!(::Dimension{3}, ω, setup, V, t)
    (; grid, operators, boundary_conditions) = setup
    (; indu, indv, indw, Nux_in, Nvy_in, Nwz_in, Nx, Ny, Nz) = grid
    (; Wu_uy, Wu_uz, Wv_vx, Wv_vz, Ww_wx, Ww_wy) = operators

    uₕ = @view V[indu]
    vₕ = @view V[indv]
    wₕ = @view V[indw]
    ω_flat = reshape(ω, length(ω))

    if boundary_conditions.u.x[1] == :periodic &&
       boundary_conditions.v.y[1] == :periodic &&
       boundary_conditions.w.z[1] == :periodic
        uₕ_in = uₕ
        vₕ_in = vₕ
        wₕ_in = wₕ
    else
        # Velocity at inner points
        diagpos = 0
        boundary_conditions.u.x[1] == :pressure && (diagpos = 1)
        boundary_conditions.u.x == (:periodic, :periodic) && (diagpos = 1)
        B1D = spdiagm(Nx - 1, Nux_in, diagpos => ones(Nx - 1))
        B2D = I(Nz) ⊗ I(Ny) ⊗ B1D
        uₕ_in = B2D * uₕ

        diagpos = 0
        boundary_conditions.v.y[1] == :pressure && (diagpos = 1)
        boundary_conditions.v.y == (:periodic, :periodic) && (diagpos = 1)
        B1D = spdiagm(Ny - 1, Nvy_in, diagpos => ones(Ny - 1))
        B2D = I(Nz) ⊗ B1D ⊗ I(Nx)
        vₕ_in = B2D * vₕ

        diagpos = 0
        boundary_conditions.w.z[1] == :pressure && (diagpos = 1)
        boundary_conditions.w.z == (:periodic, :periodic) && (diagpos = 1)
        B1D = spdiagm(Nz - 1, Nwz_in, diagpos => ones(Nz - 1))
        B2D = B1D ⊗ I(Ny) ⊗ I(Nx)
        wₕ_in = B2D * wₕ
    end

    # ωx_flat .= Ww_wy * wₕ_in - Wy_yz * vₕ_in
    # ωy_flat .= Wu_uz * uₕ_in - Ww_wx * wₕ_in
    # ωz_flat .= Wv_vx * vₕ_in - Wu_uy * uₕ_in

    ω_flat .= .√(
        (Ww_wy * wₕ_in .- Wv_vz * vₕ_in) .^ 2 .+ (Wu_uz * uₕ_in .- Ww_wx * wₕ_in) .^ 2 .+ (Wv_vx * vₕ_in .- Wu_uy * uₕ_in) .^ 2,
    )

    ω
end
