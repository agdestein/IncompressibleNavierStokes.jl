"""
    plot_vorticity(setup, V, t)

Plot vorticity field.
"""
function plot_vorticity end

# 2D version
function plot_vorticity(setup, V, t)
    (; bc) = setup
    (; x, y, xlims, ylims) = setup.grid

    if all(==(:periodic), (bc.u.x[1], bc.v.y[1]))
        xω = x
        yω = y
    else
        xω = x[2:end-1]
        yω = y[2:end-1]
    end

    # Get fields
    ω = get_vorticity(V, t, setup)
    # Plot vorticity
    # levels = [minimum(ω), -5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, maximum(ω)]
    levels = [-7, -5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 7]
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        aspect = DataAspect(),
        title = "Vorticity ω",
        xlabel = "x",
        ylabel = "y",
    )
    limits!(ax, xlims[1], xlims[2], ylims[1], ylims[2])
    contourf!(
        ax, xω, yω, ω;
        levels,
        extendlow = :auto,
        extendhigh = :auto,
    )

    # save("output/vorticity.png", fig, pt_per_unit = 2)

    fig
end

# 3D version
function plot_vorticity(setup::Setup{T,3}, V, t) where {T}
    (; grid, bc) = setup
    (; x, y, z) = grid

    if all(==(:periodic), (bc.u.x[1], bc.v.y[1], bc.w.z[1]))
        xω = x
        yω = y
        zω = z
    else
        xω = x[2:end-1]
        yω = y[2:end-1]
        zω = z[2:end-1]
    end

    ω = get_vorticity(V, t, setup)
    contour(
        xω,
        yω,
        zω,
        ω;
        extendlow = :auto,
        extendhigh = :auto,
    )
end
