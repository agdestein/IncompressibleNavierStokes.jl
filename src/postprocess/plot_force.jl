"""
    plot_force(setup, t; kwargs...)

Plot body force.
"""
function plot_force end

plot_force(setup, t; kwargs...) = plot_force(setup.grid.dimension, setup, t; kwargs...)

# 2D version
function plot_force(::Dimension{2}, setup, t; kwargs...)
    (; grid, force) = setup
    (; xp, yp, xlims, ylims) = grid
    (; xu, yu, zu, indu, xlims, ylims) = grid
    (; F) = force

    Fp = reshape(F[indu], size(xu))
    # TODO: norm of F instead of Fu
    xp, yp = xu[:, 1], yu[1, :]

    # Levels
    μ, σ = mean(Fp), std(Fp)
    ≈(μ + σ, μ; rtol = 1e-8, atol = 1e-8) && (σ = 1e-4)
    levels = LinRange(μ - 1.5σ, μ + 1.5σ, 10)

    fig = Figure()
    ax = Axis(fig[1, 1]; aspect = DataAspect(), title = "Force", xlabel = "x", ylabel = "y")
    limits!(ax, xlims[1], xlims[2], ylims[1], ylims[2])
    cf = contourf!(ax, xp, yp, Fp; extendlow = :auto, extendhigh = :auto, levels, kwargs...)
    Colorbar(fig[1, 2], cf)
    fig
end

# 3D version
function plot_force(::Dimension{3}, setup, t; kwargs...)
    (; grid, force) = setup
    (; xu, yu, zu, indu) = grid
    (; F) = force

    # Get force at pressure points
    # up, vp, wp = get_velocity(setup, V, t)
    # qp = map((u, v, w) -> √sum(u^2 + v^2 + w^2), up, vp, wp)
    Fp = reshape(F[indu], size(xu))
    # TODO: norm of F instead of Fu
    xp, yp, zp = xu[:, 1, 1], yu[1, :, 1], zu[1, 1, :]

    # Levels
    μ, σ = mean(Fp), std(Fp)
    levels = LinRange(μ - 3σ, μ + 3σ, 10)

    # contour(xp, yp, zp, Fp; levels, kwargs...)
    contour(xp, yp, zp, Fp)
end
