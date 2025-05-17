#### Implementations of eddy viscosity closure models
#### Contains:
#### - Wale                                 wale_closure                    only for 3D
#### - Smagorinsky                          smagorinsky_closure             only for 3D
#### Use these models in INS, by including them in the right hand side function.
#### See `examples/ChannelFlow3D.jl` for an example.
#### provide `params` as a parameter to solve_unsteady() !!!

strain!(S, u, setup) = apply!(strain_kernel!, setup, S, u, setup.grid)

@kernel function strain_kernel!(S, u, grid, O::CartesianIndex{2})
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

@kernel function strain_kernel!(S, u, grid, O::CartesianIndex{3})
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

gradient_tensor!(G, u, setup) = apply!(gradient_tensor_kernel!, setup, G, u, setup.grid)

@kernel function gradient_tensor_kernel!(G, u, grid, O::CartesianIndex{3})
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
Compute symmetric part of gradient.
Overwrite the gradient itself.
"""
function strain_from_gradient!(G)
    G.xy .= (G.xy .+ G.yx) ./ 2
    G.xz .= (G.xz .+ G.zx) ./ 2
    G.yz .= (G.yz .+ G.zy) ./ 2
end

smagorinsky_viscosity!(visc, S, θ, setup) =
    apply!(smagorinsky_viscosity_kernel!, setup, visc, S, θ, setup.grid)

@kernel function smagorinsky_viscosity_kernel!(visc, S, θ, grid, O::CartesianIndex{2})
    I = @index(Global, Cartesian)
    I = I + O
    (; Δ) = grid
    ex, ey = unit_cartesian_indices(2)
    d = gridsize(Δ, I)
    Sxx2 = S.xx[I]^2
    Syy2 = S.yy[I]^2
    Sxy2 = (S.xy[I]^2 + S.xy[I-ex]^2 + S.xy[I-ey]^2 + S.xy[I-ex-ey]^2) / 4
    visc[I] = θ^2 * d^2 * sqrt(2 * (Sxx2 + Syy2) + 4 * Sxy2)
end

@kernel function smagorinsky_viscosity_kernel!(visc, S, θ, grid, O::CartesianIndex{3})
    I = @index(Global, Cartesian)
    I = I + O
    (; Δ) = grid
    ex, ey, ez = unit_cartesian_indices(3)
    d = gridsize(Δ, I)
    Sxx2 = S.xx[I]^2
    Syy2 = S.yy[I]^2
    Szz2 = S.zz[I]^2
    Sxy2 = (S.xy[I]^2 + S.xy[I-ex]^2 + S.xy[I-ey]^2 + S.xy[I-ex-ey]^2) / 4
    Sxz2 = (S.xz[I]^2 + S.xz[I-ex]^2 + S.xz[I-ez]^2 + S.xz[I-ex-ez]^2) / 4
    Syz2 = (S.yz[I]^2 + S.yz[I-ey]^2 + S.yz[I-ez]^2 + S.yz[I-ey-ez]^2) / 4
    visc[I] = θ^2 * d^2 * sqrt(2 * (Sxx2 + Syy2 + Szz2) + 4 * (Sxy2 + Sxz2 + Syz2))
end

apply_eddy_viscosity!(σ, visc, setup) = apply!(apply_eddy_viscosity_kernel!, setup, σ, visc)

@kernel function apply_eddy_viscosity_kernel!(σ, visc, O::CartesianIndex{2})
    I = @index(Global, Cartesian)
    I = I + O
    ex, ey = unit_cartesian_indices(2)
    # TODO: Add interpolation weights here
    visc_xy = (visc[I] + visc[I+ex] + visc[I+ey] + visc[I+ex+ey]) / 4
    σ.xx[I] = -2 * visc[I] * σ.xx[I]
    σ.yy[I] = -2 * visc[I] * σ.yy[I]
    σ.xy[I] = -2 * visc_xy * σ.xy[I]
end

@kernel function apply_eddy_viscosity_kernel!(σ, visc, O::CartesianIndex{3})
    I = @index(Global, Cartesian)
    I = I + O
    ex, ey, ez = unit_cartesian_indices(3)
    # TODO: Add interpolation weights here
    visc_xy = (visc[I] + visc[I+ex] + visc[I+ey] + visc[I+ex+ey]) / 4
    visc_xz = (visc[I] + visc[I+ex] + visc[I+ez] + visc[I+ex+ez]) / 4
    visc_yz = (visc[I] + visc[I+ey] + visc[I+ez] + visc[I+ey+ez]) / 4
    σ.xx[I] = -2 * visc[I] * σ.xx[I]
    σ.yy[I] = -2 * visc[I] * σ.yy[I]
    σ.zz[I] = -2 * visc[I] * σ.zz[I]
    σ.xy[I] = -2 * visc_xy * σ.xy[I]
    σ.xz[I] = -2 * visc_xz * σ.xz[I]
    σ.yz[I] = -2 * visc_yz * σ.yz[I]
end

divoftensor!(c, σ, setup) = apply!(divoftensor_kernel!, setup, c, σ, setup.grid)

@kernel function divoftensor_kernel!(f, σ, grid, O::CartesianIndex{2})
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

@kernel function divoftensor_kernel!(f, σ, grid, O::CartesianIndex{3})
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

wale_viscosity!(visc, G_split, θ, setup) =
    apply!(wale_viscosity_kernel!, setup, visc, G_split, θ, setup.grid)

@kernel function wale_viscosity_kernel!(visc, G_split, θ, grid, O::CartesianIndex{3})
    I = @index(Global, Cartesian)
    I = I + O
    (; Δ) = grid
    T = eltype(G_split.xx)
    ex, ey, ez = unit_cartesian_indices(3)
    G = SMatrix{3,3,eltype(G_split.xx),9}(
        G_split.xx[I],
        (G_split.yx[I] + G_split.yx[I-ex] + G_split.yx[I-ey] + G_split.yx[I-ex-ey]) / 4,
        (G_split.zx[I] + G_split.zx[I-ex] + G_split.zx[I-ez] + G_split.zx[I-ex-ez]) / 4,
        (G_split.xy[I] + G_split.xy[I-ex] + G_split.xy[I-ey] + G_split.xy[I-ex-ey]) / 4,
        G_split.yy[I],
        (G_split.zy[I] + G_split.zy[I-ey] + G_split.zy[I-ez] + G_split.zy[I-ey-ez]) / 4,
        (G_split.xz[I] + G_split.xz[I-ex] + G_split.xz[I-ez] + G_split.xz[I-ex-ez]) / 4,
        (G_split.yz[I] + G_split.yz[I-ey] + G_split.yz[I-ez] + G_split.yz[I-ey-ez]) / 4,
        G_split.zz[I],
    )
    d = gridsize_vol(Δ, I)
    S = (G + G') / 2
    G2 = G * G
    Sd = (G2 + G2') / 2 - tr(G2) * one(G2) / 3
    visc[I] =
        (θ * d)^2 * dot(Sd, Sd)^T(3 / 2) /
        (dot(S, S)^T(5 / 2) + dot(Sd, Sd)^T(5/4) + eps(T))
end

smagorinsky_viscosity!(visc, G, θ, setup) =
    apply!(smagorinsky_viscosity_kernel!, setup, visc, G, θ, setup.grid)

@kernel function smagorinsky_viscosity_kernel!(
    visc,
    G_split,
    θ,
    grid,
    I0::CartesianIndex{3},
)
    I = @index(Global, Cartesian)
    I = I + I0
    (; Δ) = grid
    T = eltype(G_split.xx)
    ex, ey, ez = unit_cartesian_indices(3)
    G = SMatrix{3,3,eltype(G_split.xx),9}(
        G_split.xx[I],
        (G_split.yx[I] + G_split.yx[I-ex] + G_split.yx[I-ey] + G_split.yx[I-ex-ey]) / 4,
        (G_split.zx[I] + G_split.zx[I-ex] + G_split.zx[I-ez] + G_split.zx[I-ex-ez]) / 4,
        (G_split.xy[I] + G_split.xy[I-ex] + G_split.xy[I-ey] + G_split.xy[I-ex-ey]) / 4,
        G_split.yy[I],
        (G_split.zy[I] + G_split.zy[I-ey] + G_split.zy[I-ez] + G_split.zy[I-ey-ez]) / 4,
        (G_split.xz[I] + G_split.xz[I-ex] + G_split.xz[I-ez] + G_split.xz[I-ex-ez]) / 4,
        (G_split.yz[I] + G_split.yz[I-ey] + G_split.yz[I-ez] + G_split.yz[I-ey-ez]) / 4,
        G_split.zz[I],
    )
    S = (G + G') / 2
    d = gridsize_vol(Δ, I)
    visc[I] = θ^2 * d^2 * sqrt(2 * dot(S, S))
end

function zero_out_wall!(p, setup)
    d = setup.grid.dimension()
    for i = 1:d
        bc = setup.boundary_conditions[i]
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
    strain_from_gradient!(G)
    apply_eddy_viscosity!(G, visc, setup)
    for g in G
        zero_out_wall!(g, setup)
        apply_bc_p!(g, zero(eltype(u)), setup)
    end
    divoftensor!(f, G, setup)
end

"Apply Smagorinsky closure model."
function smagorinsky_closure!(f, u, θ, cache, setup)
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
    smagorinsky_viscosity!(visc, G, θ, setup)
    zero_out_wall!(visc, setup)

    apply_bc_p!(visc, zero(eltype(u)), setup)
    strain_from_gradient!(G)
    apply_eddy_viscosity!(G, visc, setup)
    for g in G
        zero_out_wall!(g, setup)
        apply_bc_p!(g, zero(eltype(u)), setup)
    end
    divoftensor!(f, G, setup)
end

function get_cache(::Union{typeof(wale_closure),typeof(smagorinsky_closure)}, setup)
    (; dimension, x, N) = setup.grid
    D = dimension()
    T = eltype(x[1])
    visc = scalarfield(setup)
    G = if D == 2
        (;
            xx = scalarfield(setup),
            yx = scalarfield(setup),
            xy = scalarfield(setup),
            yy = scalarfield(setup),
        )
    elseif D == 3
        (;
            xx = scalarfield(setup),
            yx = scalarfield(setup),
            zx = scalarfield(setup),
            xy = scalarfield(setup),
            yy = scalarfield(setup),
            zy = scalarfield(setup),
            xz = scalarfield(setup),
            yz = scalarfield(setup),
            zz = scalarfield(setup),
        )
    end
    (; visc, G)
end
