"""
    plot_velocity(setup, p)

Plot velocity.
"""
function plot_velocity end

# 2D version
function plot_velocity(setup::Setup{T,2}, V, t) where {T}
    (; xp, yp, xlims, ylims) = setup.grid

    # Reshape
    up, vp = get_velocity(V, t, setup)
    qp = map((u, v) -> √(u^2 + v^2), up, vp)

    fig = Figure()
    ax = Axis(
        fig[1, 1];
        aspect = DataAspect(),
        title = "Velocity magnitude",
        xlabel = "x",
        ylabel = "y",
    )
    limits!(ax, xlims[1], xlims[2], ylims[1], ylims[2])
    contourf!(
        ax,
        xp,
        yp,
        qp;
        extendlow = :auto, extendhigh = :auto,
    )

    # save("output/pressure.png", fig, pt_per_unit = 2)

    fig
end

# 3D version
function plot_velocity(setup::Setup{T,3}, V, t) where {T}
    (; xp, yp, zp) = setup.grid

    # Reshape
    up, vp, wp = get_velocity(V, t, setup)
    qp = map((u, v, w) -> √sum(u ^ 2 + v ^ 2 + w ^ 2), up, vp, wp)

    contour(
        xp,
        yp,
        zp,
        qp;
        extendlow = :auto, extendhigh = :auto,
    )

    # save("output/pressure.png", fig, pt_per_unit = 2)
end
