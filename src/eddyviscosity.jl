#### Implementations of eddy viscosity closure models
#### Contains:
#### - WALE (only 3D):          `wale_closure!`
#### - Smagorinsky (2D and 3D): `smagorinsky_closure!`
#### Use these models in INS, by including them in the right hand side function.
#### See `examples/ChannelFlow3D.jl` for an example.
#### provide model coefficients in `params` as a parameter to solve_unsteady() !!!

getgrid(setup) = (; setup.Δ, setup.Δu)

strain!(S, u, setup) = apply!(strain_kernel!, setup, S, u, getgrid(setup))

@kernel function strain_kernel!(O::CartesianIndex{2}, S, u, grid)
    I = @index(Global, Cartesian)
    I = I + O
    (; Δ, Δu) = grid
    ex, ey = unit_cartesian_indices(2)
    Δux, Δuy = Δu[1][I[1]], Δ[2][I[2]]
    Δvx, Δvy = Δ[1][I[1]], Δu[2][I[2]]
    ∂u∂x = (u[I, 1] - u[I-ex, 1]) / Δux
    ∂u∂y = (u[I+ey, 1] - u[I, 1]) / Δuy
    ∂v∂x = (u[I+ex, 2] - u[I, 2]) / Δvx
    ∂v∂y = (u[I, 2] - u[I-ey, 2]) / Δvy
    S.xx[I] = ∂u∂x
    S.yy[I] = ∂v∂y
    S.xy[I] = (∂u∂y + ∂v∂x) / 2
end

@kernel function strain_kernel!(O::CartesianIndex{3}, S, u, grid)
    I = @index(Global, Cartesian)
    I = I + O
    (; Δ, Δu) = grid
    ex, ey, ez = unit_cartesian_indices(3)
    Δux, Δuy, Δuz = Δu[1][I[1]], Δ[2][I[2]], Δ[3][I[3]]
    Δvx, Δvy, Δvz = Δ[1][I[1]], Δu[2][I[2]], Δ[3][I[3]]
    Δwx, Δwy, Δwz = Δ[1][I[1]], Δ[2][I[2]], Δu[3][I[3]]
    ∂u∂x = (u[I, 1] - u[I-ex, 1]) / Δux
    ∂u∂y = (u[I+ey, 1] - u[I, 1]) / Δuy
    ∂u∂z = (u[I+ez, 1] - u[I, 1]) / Δuz
    ∂v∂x = (u[I+ex, 2] - u[I, 2]) / Δvx
    ∂v∂y = (u[I, 2] - u[I-ey, 2]) / Δvy
    ∂v∂z = (u[I+ez, 2] - u[I, 2]) / Δvz
    ∂w∂x = (u[I+ex, 3] - u[I, 3]) / Δwx
    ∂w∂y = (u[I+ey, 3] - u[I, 3]) / Δwy
    ∂w∂z = (u[I, 3] - u[I-ez, 3]) / Δwz
    S.xx[I] = ∂u∂x
    S.yy[I] = ∂v∂y
    S.zz[I] = ∂w∂z
    S.xy[I] = (∂u∂y + ∂v∂x) / 2
    S.xz[I] = (∂u∂z + ∂w∂x) / 2
    S.yz[I] = (∂v∂z + ∂w∂y) / 2
end

gradient_tensor!(G, u, setup) = apply!(gradient_tensor_kernel!, setup, G, u, getgrid(setup))

@kernel function gradient_tensor_kernel!(O::CartesianIndex{3}, G, u, grid)
    I = @index(Global, Cartesian)
    I = I + O
    (; Δ, Δu) = grid
    ex, ey, ez = unit_cartesian_indices(3)
    Δux, Δuy, Δuz = Δ[1][I[1]], Δu[2][I[2]], Δu[3][I[3]]
    Δvx, Δvy, Δvz = Δu[1][I[1]], Δ[2][I[2]], Δu[3][I[3]]
    Δwx, Δwy, Δwz = Δu[1][I[1]], Δu[2][I[2]], Δ[3][I[3]]
    G.xx[I] = (u[I, 1] - u[I-ex, 1]) / Δux
    G.xy[I] = (u[I+ey, 1] - u[I, 1]) / Δuy
    G.xz[I] = (u[I+ez, 1] - u[I, 1]) / Δuz
    G.yx[I] = (u[I+ex, 2] - u[I, 2]) / Δvx
    G.yy[I] = (u[I, 2] - u[I-ey, 2]) / Δvy
    G.yz[I] = (u[I+ez, 2] - u[I, 2]) / Δvz
    G.zx[I] = (u[I+ex, 3] - u[I, 3]) / Δwx
    G.zy[I] = (u[I+ey, 3] - u[I, 3]) / Δwy
    G.zz[I] = (u[I, 3] - u[I-ez, 3]) / Δwz
end

"""
Compute symmetric part of tensor.
Overwrite the upper diagonal of the tensor itself.
The lower diagonal is not modified - don't use it!
"""
function symmetrize!(G)
    G.xy .= (G.xy .+ G.yx) ./ 2
    G.xz .= (G.xz .+ G.zx) ./ 2
    G.yz .= (G.yz .+ G.zy) ./ 2
end

smagorinsky_viscosity!(visc, S, θ, setup) =
    apply!(smagorinsky_viscosity_kernel!, setup, visc, S, θ, getgrid(setup))

@kernel function smagorinsky_viscosity_kernel!(O::CartesianIndex{2}, visc, S, θ, grid)
    I = @index(Global, Cartesian)
    I = I + O
    (; Δ) = grid
    ex, ey = unit_cartesian_indices(2)
    d = gridsize_vol(grid, I)
    Sxx2 = S.xx[I]^2
    Syy2 = S.yy[I]^2
    Sxy2 = (S.xy[I]^2 + S.xy[I-ex]^2 + S.xy[I-ey]^2 + S.xy[I-ex-ey]^2) / 4
    visc[I] = θ^2 * d^2 * sqrt(2 * (Sxx2 + Syy2) + 4 * Sxy2)
end

@kernel function smagorinsky_viscosity_kernel!(O::CartesianIndex{3}, visc, S, θ, grid)
    I = @index(Global, Cartesian)
    I = I + O
    (; Δ) = grid
    ex, ey, ez = unit_cartesian_indices(3)
    d = gridsize_vol(grid, I)
    Sxx2 = S.xx[I]^2
    Syy2 = S.yy[I]^2
    Szz2 = S.zz[I]^2
    Sxy2 = (S.xy[I]^2 + S.xy[I-ex]^2 + S.xy[I-ey]^2 + S.xy[I-ex-ey]^2) / 4
    Sxz2 = (S.xz[I]^2 + S.xz[I-ex]^2 + S.xz[I-ez]^2 + S.xz[I-ex-ez]^2) / 4
    Syz2 = (S.yz[I]^2 + S.yz[I-ey]^2 + S.yz[I-ez]^2 + S.yz[I-ey-ez]^2) / 4
    visc[I] = θ^2 * d^2 * sqrt(2 * (Sxx2 + Syy2 + Szz2) + 4 * (Sxy2 + Sxz2 + Syz2))
end

apply_eddy_viscosity!(σ, visc, setup) = apply!(apply_eddy_viscosity_kernel!, setup, σ, visc)

# Strain is already stored in σ, multiply by eddy-viscosity scaling
@kernel function apply_eddy_viscosity_kernel!(O::CartesianIndex{2}, σ, visc)
    I = @index(Global, Cartesian)
    I = I + O
    ex, ey = unit_cartesian_indices(2)

    # Get linear interpolation weights
    Δx, Δy = Δ
    Δxa, Δxb = Δx[I], Δx[I+ex]
    Δya, Δyb = Δy[I], Δy[I+ey]
    ax, bx = Δxb / (Δxa + Δxb), Δxa / (Δxa + Δxb)
    ay, by = Δyb / (Δya + Δyb), Δya / (Δya + Δyb)

    # Interpolate viscosity to off-diagonal location
    visc_xy = ax * ay * visc[I] + bx * ay * visc[I+ex] + ax * by * visc[I+ey] + bx * by * visc[I+ex+ey]

    σ.xx[I] = -2 * visc[I] * σ.xx[I]
    σ.yy[I] = -2 * visc[I] * σ.yy[I]
    σ.xy[I] = -2 * visc_xy * σ.xy[I]
end

@kernel function apply_eddy_viscosity_kernel!(O::CartesianIndex{3}, σ, visc, Δ)
    I = @index(Global, Cartesian)
    I = I + O
    ex, ey, ez = unit_cartesian_indices(3)

    # Get linear interpolation weights
    Δx, Δy, Δz = Δ
    Δxa, Δxb = Δx[I], Δx[I+ex]
    Δya, Δyb = Δy[I], Δy[I+ey]
    Δza, Δzb = Δz[I], Δz[I+ez]
    ax, bx = Δxb / (Δxa + Δxb), Δxa / (Δxa + Δxb)
    ay, by = Δyb / (Δya + Δyb), Δya / (Δya + Δyb)
    az, bz = Δzb / (Δza + Δzb), Δza / (Δza + Δzb)

    # Interpolate viscosities to off-diagonal locations
    visc_xy = ax * ay * visc[I] + bx * ay * visc[I+ex] + ax * by * visc[I+ey] + bx * by * visc[I+ex+ey]
    visc_xz = ax * az * visc[I] + bx * az * visc[I+ex] + ax * bz * visc[I+ez] + bx * bz * visc[I+ex+ez]
    visc_yz = ay * az * visc[I] + by * az * visc[I+ey] + ay * bz * visc[I+ez] + by * bz * visc[I+ey+ez]

    σ.xx[I] = -2 * visc[I] * σ.xx[I]
    σ.yy[I] = -2 * visc[I] * σ.yy[I]
    σ.zz[I] = -2 * visc[I] * σ.zz[I]
    σ.xy[I] = -2 * visc_xy * σ.xy[I]
    σ.xz[I] = -2 * visc_xz * σ.xz[I]
    σ.yz[I] = -2 * visc_yz * σ.yz[I]
end

divoftensor!(c, σ, setup) = apply!(divoftensor_kernel!, setup, c, σ, getgrid(setup))

@kernel function divoftensor_kernel!(O::CartesianIndex{2}, f, σ, grid)
    I = @index(Global, Cartesian)
    I = I + O
    (; Δ, Δu) = grid
    ex, ey = unit_cartesian_indices(2)
    Δpx, Δpy = Δ[1][I[1]], Δ[2][I[2]]
    Δux, Δuy = Δu[1][I[1]], Δu[2][I[2]]
    ∂σxx∂x = (σ.xx[I+ex] - σ.xx[I]) / Δux
    ∂σxy∂y = (σ.xy[I] - σ.xy[I-ey]) / Δpy
    ∂σyx∂x = (σ.xy[I] - σ.xy[I-ex]) / Δpx
    ∂σyy∂y = (σ.yy[I+ey] - σ.yy[I]) / Δuy
    f[I, 1] -= ∂σxx∂x + ∂σxy∂y
    f[I, 2] -= ∂σyx∂x + ∂σyy∂y
end

@kernel function divoftensor_kernel!(O::CartesianIndex{3}, f, σ, grid)
    I = @index(Global, Cartesian)
    I = I + O
    (; Δ, Δu) = grid
    ex, ey, ez = unit_cartesian_indices(3)
    Δpx, Δpy, Δpz = Δ[1][I[1]], Δ[2][I[2]], Δ[3][I[3]]
    Δux, Δuy, Δuz = Δu[1][I[1]], Δu[2][I[2]], Δu[3][I[3]]
    ∂σxx∂x = (σ.xx[I+ex] - σ.xx[I]) / Δux
    ∂σxy∂y = (σ.xy[I] - σ.xy[I-ey]) / Δpy
    ∂σxz∂z = (σ.xz[I] - σ.xz[I-ez]) / Δpz
    ∂σyx∂x = (σ.xy[I] - σ.xy[I-ex]) / Δpx
    ∂σyy∂y = (σ.yy[I+ey] - σ.yy[I]) / Δuy
    ∂σyz∂z = (σ.yz[I] - σ.yz[I-ez]) / Δpz
    ∂σzx∂x = (σ.xz[I] - σ.xz[I-ex]) / Δpx
    ∂σzy∂y = (σ.yz[I] - σ.yz[I-ey]) / Δpy
    ∂σzz∂z = (σ.zz[I+ez] - σ.zz[I]) / Δuz
    f[I, 1] -= ∂σxx∂x + ∂σxy∂y + ∂σxz∂z
    f[I, 2] -= ∂σyx∂x + ∂σyy∂y + ∂σyz∂z
    f[I, 3] -= ∂σzx∂x + ∂σzy∂y + ∂σzz∂z
end

"Apply WALE viscosity."
wale_viscosity!(visc, G_split, θ, setup) =
    apply!(wale_viscosity_kernel!, setup, visc, G_split, θ, getgrid(setup))

"""
Collocate staggered tensor to the center of the cell.
Put the tensor in a statically sized `SMatrix`.
"""
function collocate_tensor end
function collocate_tensor(σ, I::CartesianIndex{2})
    ex, ey = unit_cartesian_indices(2)
    SMatrix{2,2,eltype(σ.xx),4}(
        σ.xx[I],
        (σ.yx[I] + σ.yx[I-ex] + σ.yx[I-ey] + σ.yx[I-ex-ey]) / 4,
        (σ.xy[I] + σ.xy[I-ex] + σ.xy[I-ey] + σ.xy[I-ex-ey]) / 4,
        σ.yy[I],
    )
end
function collocate_tensor(σ, I::CartesianIndex{3})
    ex, ey, ez = unit_cartesian_indices(3)
    SMatrix{3,3,eltype(σ.xx),9}(
        σ.xx[I],
        (σ.yx[I] + σ.yx[I-ex] + σ.yx[I-ey] + σ.yx[I-ex-ey]) / 4,
        (σ.zx[I] + σ.zx[I-ex] + σ.zx[I-ez] + σ.zx[I-ex-ez]) / 4,
        (σ.xy[I] + σ.xy[I-ex] + σ.xy[I-ey] + σ.xy[I-ex-ey]) / 4,
        σ.yy[I],
        (σ.zy[I] + σ.zy[I-ey] + σ.zy[I-ez] + σ.zy[I-ey-ez]) / 4,
        (σ.xz[I] + σ.xz[I-ex] + σ.xz[I-ez] + σ.xz[I-ex-ez]) / 4,
        (σ.yz[I] + σ.yz[I-ey] + σ.yz[I-ez] + σ.yz[I-ey-ez]) / 4,
        σ.zz[I],
    )
end

@kernel function wale_viscosity_kernel!(O::CartesianIndex{3}, visc, G_split, θ, grid)
    I = @index(Global, Cartesian)
    I = I + O
    (; Δ) = grid
    T = eltype(G_split.xx)
    ex, ey, ez = unit_cartesian_indices(3)
    G = collocate_tensor(G_split, I)
    d = gridsize_vol(grid, I)
    S = (G + G') / 2
    G2 = G * G
    Sd = (G2 + G2') / 2 - tr(G2) * one(G2) / 3
    visc[I] =
        (θ * d)^2 * dot(Sd, Sd)^T(3 / 2) /
        (dot(S, S)^T(5 / 2) + dot(Sd, Sd)^T(5/4) + eps(T))
end

function zero_out_wall!(p, setup)
    d = setup.dimension()
    for i = 1:d
        bc = setup.boundary_conditions.u[i]
        bc[1] isa DirichletBC && fill!(view(p, ntuple(j -> i == j ? 1 : (:), d)...), 0)
        bc[2] isa DirichletBC &&
            fill!(view(p, ntuple(j -> i == j ? size(p, i) : (:), d)...), 0)
    end
end

"Apply WALE closure model."
function wale_closure!(f, u, θ, cache, setup)
    (; visc, G) = cache
    fill!(visc, 0)
    for g in G
        fill!(g, 0)
    end
    gradient_tensor!(G, u, setup)
    for g in G
        zero_out_wall!(g, setup)
        apply_bc_p!(g, zero(eltype(u)), setup)
    end
    wale_viscosity!(visc, G, θ, setup)
    zero_out_wall!(visc, setup)
    apply_bc_p!(visc, zero(eltype(u)), setup)
    symmetrize!(G)
    apply_eddy_viscosity!(G, visc, setup)
    for g in G
        zero_out_wall!(g, setup)
        apply_bc_p!(g, zero(eltype(u)), setup)
    end
    divoftensor!(f, G, setup)
end

"Apply Smagorinsky closure model."
function smagorinsky_closure!(f, u, θ, cache, setup)
    (; visc, S) = cache
    fill!(visc, 0)
    for S in S
        fill!(S, 0)
    end
    strain!(S, u, setup)
    for S in S
        zero_out_wall!(S, setup)
        apply_bc_p!(S, zero(eltype(u)), setup)
    end
    smagorinsky_viscosity!(visc, S, θ, setup)
    zero_out_wall!(visc, setup)
    apply_bc_p!(visc, zero(eltype(u)), setup)
    apply_eddy_viscosity!(S, visc, setup)
    for s in S
        zero_out_wall!(s, setup)
        apply_bc_p!(s, zero(eltype(u)), setup)
    end
    divoftensor!(f, S, setup)
end

get_cache(::typeof(wale_closure!), setup) =
    (; visc = scalarfield(setup), G = tensorfield(setup))

get_cache(::typeof(smagorinsky_closure!), setup) =
    (; visc = scalarfield(setup), S = symmetric_tensorfield(setup))
