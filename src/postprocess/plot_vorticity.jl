"""
    plot_vorticity(setup, V, t)

Plot vorticity field.
"""
function plot_vorticity(setup, V, t)
    (; bc) = setup
    (; x, y, xlims, ylims) = setup.grid

    if bc.u.x[1] == :periodic
        xω = x
    else
        xω = x[2:end-1]
    end
    if bc.v.y[1] == :periodic
        yω = y
    else
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
