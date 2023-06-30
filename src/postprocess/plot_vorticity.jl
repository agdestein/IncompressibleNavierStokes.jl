"""
    plot_vorticity(setup, V, t; kwargs...)

Plot vorticity field.
"""
function plot_vorticity end

plot_vorticity(setup, V, t; kwargs...) = plot_vorticity(setup.grid.dimension, setup, V, t; kwargs...)

# 2D version
function plot_vorticity(::Dimension{2}, setup, V, t; kwargs...)
    (; grid, boundary_conditions) = setup
    (; x, y, xlims, ylims) = grid

    if all(==(:periodic), (boundary_conditions.u.x[1], boundary_conditions.v.y[1]))
        xω = x
        yω = y
    else
        xω = x[2:(end-1)]
        yω = y[2:(end-1)]
    end

    # Get fields
    ω = get_vorticity(setup, V, t)

    # Levels
    μ, σ = mean(ω), std(ω)
    ≈(μ + σ, μ; rtol = 1e-8, atol = 1e-8) && (σ = 1e-4)
    levels = LinRange(μ - 1.5σ, μ + 1.5σ, 10)

    # Plot vorticity
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        aspect = DataAspect(),
        title = "Vorticity",
        xlabel = "x",
        ylabel = "y",
    )
    limits!(ax, xlims[1], xlims[2], ylims[1], ylims[2])
    cf = contourf!(ax, xω, yω, ω; extendlow = :auto, extendhigh = :auto, levels, kwargs...)
    Colorbar(fig[1, 2], cf)

    # save("output/vorticity.png", fig, pt_per_unit = 2)

    fig
end

# 3D version
function plot_vorticity(::Dimension{3}, setup, V, t; kwargs...)
    (; grid, boundary_conditions) = setup
    (; x, y, z) = grid

    if all(
        ==(:periodic),
        (
            boundary_conditions.u.x[1],
            boundary_conditions.v.y[1],
            boundary_conditions.w.z[1],
        ),
    )
        xω = x
        yω = y
        zω = z
    else
        xω = x[2:(end-1)]
        yω = y[2:(end-1)]
        zω = z[2:(end-1)]
    end

    ω = get_vorticity(setup, V, t)

    # Levels
    μ, σ = mean(ω), std(ω)
    levels = LinRange(μ - 3σ, μ + 3σ, 10)

    contour(xω, yω, zω, ω; levels, kwargs...)
end
