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
    ≈(μ + σ, μ; rtol = 1e-8, atol = 1e-8) && (σ = 1e-4)
    levels = LinRange(μ - 1.5σ, μ + 1.5σ, 10)

    fig = Figure()
    ax = Axis(
        fig[1, 1];
        aspect = DataAspect(),
        title = "Velocity",
        xlabel = "x",
        ylabel = "y",
    )
    limits!(ax, xlims[1], xlims[2], ylims[1], ylims[2])
    cf = contourf!(ax, xp, yp, qp; extendlow = :auto, extendhigh = :auto, levels, kwargs...)
    Colorbar(fig[1,2], cf)
    # Colorbar(fig[2,1], cf; vertical = false)
    fig
end

# 3D version
function plot_velocity(setup::Setup{T,3}, V, t; kwargs...) where {T}
    (; xu, yu, zu, indu) = setup.grid

    # Get velocity at pressure points
    # up, vp, wp = get_velocity(V, t, setup)
    # qp = map((u, v, w) -> √sum(u^2 + v^2 + w^2), up, vp, wp)
    qp = reshape(V[indu], size(xu))
    xp, yp, zp = xu[:,1,1], yu[1,:,1], zu[1,1,:]
    
    # Levels
    μ, σ = mean(qp), std(qp)
    levels = LinRange(μ - 3σ, μ + 3σ, 10)

    # contour(xp, yp, zp, qp; levels, kwargs...)
    contour(xp, yp, zp, qp; kwargs...)
end
