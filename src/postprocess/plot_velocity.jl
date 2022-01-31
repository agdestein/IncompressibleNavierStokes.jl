"""
    plot_velocity(setup, V, t; kwargs...)

Plot velocity.
"""
function plot_velocity end

# 2D version
function plot_velocity(setup::Setup{T,2}, V, t; kwargs...) where {T}
    (; xp, yp, xlims, ylims) = setup.grid

    # Get velocity at pressure points
    up, vp = get_velocity(V, t, setup)
    qp = map((u, v) -> √(u^2 + v^2), up, vp)

    # Levels
    μ, σ = mean(qp), std(qp)
    levels = LinRange(μ - 1.5σ, μ + 1.5σ, 10)

    fig = Figure()
    ax = Axis(
        fig[1, 1];
        aspect = DataAspect(),
        title = "Velocity magnitude",
        xlabel = "x",
        ylabel = "y",
    )
    limits!(ax, xlims[1], xlims[2], ylims[1], ylims[2])
    contourf!(ax, xp, yp, qp; extendlow = :auto, extendhigh = :auto, levels, kwargs...)
    fig
end

# 3D version
function plot_velocity(setup::Setup{T,3}, V, t; kwargs...) where {T}
    (; xp, yp, zp) = setup.grid

    # Get velocity at pressure points
    up, vp, wp = get_velocity(V, t, setup)
    qp = map((u, v, w) -> √sum(u^2 + v^2 + w^2), up, vp, wp)
    
    # Levels
    μ, σ = mean(qp), std(qp)
    levels = LinRange(μ - 3σ, μ + 3σ, 10)

    contour(xp, yp, zp, qp; levels, kwargs...)
end
