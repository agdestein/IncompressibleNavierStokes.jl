"""
plot_vorticity.
"""
function plot_vorticity(setup, V, t)
    @unpack bc = setup
    @unpack Nx, Ny, Nu, Nv, x, y, x1, x2, y1, y2 = setup.grid

    # Reshape
    uₕ = @view V[1:Nu]
    vₕ = @view V[Nu+1:Nu+Nv]

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
    f = Figure()
    ax = Axis(
        f[1, 1];
        aspect = DataAspect(),
        title = "Vorticity ω",
        xlabel = "x",
        ylabel = "y",
    )
    limits!(ax, x1, x2, y1, y2)
    contourf!(ax, xω, yω, ω; levels)
    display(f)
    save("output/vorticity.png", f, pt_per_unit = 2)
end
