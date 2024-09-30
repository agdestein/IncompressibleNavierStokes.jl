function strain_natural!(S, u, θ, setup)
    (; grid, workgroupsize) = setup
    (; dimension, Np, Ip, Δ, Δu) = grid
    @kernel function S!(S, u, I0::CartesianIndex{2})
        I = @index(Global, Cartesian)
        I = I + I0
        u, v = u[1], u[2]
        ex, ey = unit_cartesian_indices(2)
        Δux, Δuy = Δu[1][I[1]], Δ[I[2]]
        Δvx, Δvy = Δ[I[1]], Δu[2][I[2]]
        ∂u∂x = (u[I] - u[I-ex]) / Δux
        ∂u∂y = (u[I+ey] - u[I]) / Δuy
        ∂v∂x = (v[I+ex] - v[I]) / Δvx
        ∂v∂y = (v[I] - v[I-ey]) / Δvy
        S.xx[I] = ∂u∂x
        S.yy[I] = ∂v∂y
        S.xy[I] = (∂u∂y + ∂v∂x) / 2
    end
    @kernel function S!(S, u, I0::CartesianIndex{3})
        I = @index(Global, Cartesian)
        I = I + I0
        u, v, w = u[1], u[2], u[3]
        ex, ey, ez = unit_cartesian_indices(3)
        Δux, Δuy, Δuz = Δu[1][I[1]], Δ[I[2]], Δ[I[3]]
        Δvx, Δvy, Δvz = Δ[I[1]], Δu[2][I[2]], Δ[I[3]]
        Δwx, Δwy, Δwz = Δ[I[1]], Δ[I[2]], Δu[3][I[3]]
        ∂u∂x = (u[I] - u[I-ex]) / Δux
        ∂u∂y = (u[I+ey] - u[I]) / Δuy
        ∂u∂z = (u[I+ez] - u[I]) / Δuz
        ∂v∂x = (v[I+ex] - v[I]) / Δvx
        ∂v∂y = (v[I] - v[I-ey]) / Δvy
        ∂v∂z = (v[I+ez] - v[I]) / Δvz
        ∂w∂x = (w[I+ex] - w[I]) / Δwx
        ∂w∂y = (w[I+ey] - w[I]) / Δwy
        ∂w∂z = (w[I] - w[I-ez]) / Δwz
        S.xx[I] = ∂u∂x
        S.yy[I] = ∂v∂y
        S.zz[I] = ∂w∂z
        S.xy[I] = (∂u∂y + ∂v∂x) / 2
        S.xz[I] = (∂u∂z + ∂w∂x) / 2
        S.yz[I] = (∂v∂z + ∂w∂y) / 2
    end
    I0 = first(Ip)
    I0 -= oneunit(I0)
    S!(get_backend(u[1]), workgroupsize)(S, u, I0; ndrange = Np)
    S
end

function smagorinsky_viscosity!(visc, S, θ, setup)
    (; grid, workgroupsize) = setup
    (; dimension, Np, Ip, Δ, Δu) = grid
    @kernel function visc!(visc, S, I0::CartesianIndex{2})
        I = @index(Global, Cartesian)
        I = I + I0
        ex, ey = unit_cartesian_indices(2)
        d = gridsize(Δ, I)
        Sxx2 = S.xx[I]^2
        Syy2 = S.yy[I]^2
        Sxy2 = (S.xy[I]^2 + S.xy[I-e(1)]^2 + S.xy[I-e(2)]^2 + S.xy[I-e(1)-e(2)]^2) / 4
        visc[I] = θ^2 * d^2 * sqrt(2 * (Sxx2 + Syy2) + 4 * Sxy2)
    end
    @kernel function visc!(visc, S, I0::CartesianIndex{3})
        I = @index(Global, Cartesian)
        I = I + I0
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
    I0 = first(Ip)
    I0 -= oneunit(I0)
    visc!(get_backend(u[1]), workgroupsize)(visc, S, I0; ndrange = Np)
    visc
end

function apply_eddy_viscosity!(σ, visc, setup)
    (; grid, workgroupsize) = setup
    (; Np, Ip, Δ, Δu) = grid
    @kernel function apply!(σ, visc, I0::CartesianIndex{2})
        I = @index(Global, Cartesian)
        I = I + I0
        ex, ey = unit_cartesian_indices(2)
        Δpx, Δpy = Δ[I[1]], Δ[I[2]]
        Δux, Δuy = Δu[1][I[1]], Δu[2][I[2]]
        # TODO: Add interpolation weigths here
        visc_xy = (visc[I] + visc[I+ex] + visc[I+ey] + visc[I+ex+ey]) / 4
        σ.xx[I] = 2 * visc[I] * s.xx[I]
        σ.yy[I] = 2 * visc[I] * s.yy[I]
        σ.xy[I] = 2 * visc_xy * s.xy[I]
    end
    @kernel function apply!(σ, visc, I0::CartesianIndex{3})
        I = @index(Global, Cartesian)
        I = I + I0
        ex, ey, ez = unit_cartesian_indices(3)
        Δpx, Δpy, Δpz = Δ[I[1]], Δ[I[2]], Δ[I[3]]
        Δux, Δuy, Δuz = Δu[1][I[1]], Δu[2][I[2]], Δu[3][I[3]]
        # TODO: Add interpolation weigths here
        visc_xy = (visc[I] + visc[I+ex] + visc[I+ey] + visc[I+ex+ey]) / 4
        visc_xz = (visc[I] + visc[I+ex] + visc[I+ez] + visc[I+ex+ez]) / 4
        visc_yz = (visc[I] + visc[I+ey] + visc[I+ez] + visc[I+ey+ez]) / 4
        σ.xx[I] = 2 * visc[I] * s.xx[I]
        σ.yy[I] = 2 * visc[I] * s.yy[I]
        σ.zz[I] = 2 * visc[I] * s.zz[I]
        σ.xy[I] = 2 * visc_xy * s.xy[I]
        σ.xz[I] = 2 * visc_xz * s.xz[I]
        σ.yz[I] = 2 * visc_yz * s.yz[I]
    end
    I0 = first(Ip)
    I0 -= oneunit(I0)
    apply!(get_backend(u[1]), workgroupsize)(σ, visc, I0; ndrange = Np)
    σ
end

function divoftensor_natural!(c, σ, setup)
    (; grid, workgroupsize) = setup
    (; Np, Ip, Δ, Δu) = grid
    @kernel function apply!(c, σ, I0::CartesianIndex{2})
        I = @index(Global, Cartesian)
        I = I + I0
        ex, ey = unit_cartesian_indices(2)
        Δpx, Δpy = Δ[I[1]], Δ[I[2]]
        Δux, Δuy = Δu[1][I[1]], Δu[2][I[2]]
        ∂σxx∂x = (σ.xx[I+ex] - σ.xx[I]) / Δux
        ∂σxy∂y = (σ.xy[I] - σ.xy[I-ey]) / Δpy
        ∂σyx∂x = (σ.xy[I] - σ.xy[I-ex]) / Δpx
        ∂σyy∂y = (σ.yy[I+ey] - σ.yy[I]) / Δuy
        c[1][I] = ∂σxx∂x + ∂σxy∂y
        c[2][I] = ∂σxy∂x + ∂σyy∂y
    end
    @kernel function apply!(c, σ, I0::CartesianIndex{3})
        I = @index(Global, Cartesian)
        I = I + I0
        ex, ey, ez = unit_cartesian_indices(3)
        Δpx, Δpy, Δpz = Δ[I[1]], Δ[I[2]], Δ[I[3]]
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
        c[1][I] = ∂σxx∂x + ∂σxy∂y + ∂σxz∂z
        c[2][I] = ∂σyx∂x + ∂σyy∂y + ∂σyz∂z
        c[3][I] = ∂σzx∂x + ∂σzy∂y + ∂σzz∂z
    end
    I0 = first(Ip)
    I0 -= oneunit(I0)
    apply!(get_backend(u[1]), workgroupsize)(c, σ, I0; ndrange = Np)
    c
end

function smagorinsky_closure_natural(setup)
    (; dimension, x, N) = setup.grid
    D = dimension()
    T = eltype(x[1])
    σ = if D == 2
        (; xx = scalarfield(setup), yy = scalarfield(setup), xy = scalarfield(setup))
    elseif D == 3
        (;
            xx = scalarfield(setup),
            yy = scalarfield(setup),
            zz = scalarfield(setup),
            xy = scalarfield(setup),
            xz = scalarfield(setup),
            yz = scalarfield(setup),
        )
    end
    c = vectorfield(setup)
    function closure(u, θ)
        strain_natural!(σ, u, setup)
        smagorinsky_viscosity!(visc, σ, θ, setup)
        apply_eddy_viscosity!(σ, visc, setup)
        divoftensor_natural!(c, σ, setup)
        c
    end
end
