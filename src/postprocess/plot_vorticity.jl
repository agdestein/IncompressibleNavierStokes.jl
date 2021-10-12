"""
plot_vorticity.
"""
function plot_vorticity(solution, setup)
    @unpack Nx, Ny, Nu, Nv, x, y, x1, x2, y1, y2 = setup.grid
    @unpack V, p, t = solution

    # Reshape
    uₕ = @view V[1:Nu]
    vₕ = @view V[Nu+1:Nu+Nv]

    # Get fields
    ω_flat = get_vorticity(V, t, setup)
    ω = reshape(ω_flat, Nx - 1, Ny - 1)

    # Plot vorticity
    # levels = [minimum(ω), -5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, maximum(ω)]
    levels = [- 7, -5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 7]
    f = Figure()
    ax = Axis(
        f[1, 1];
        aspect = DataAspect(),
        title = "Vorticity ω",
        xlabel = "x",
        ylabel = "y",
    )
    limits!(ax, x1, x2, y1, y2)
    contourf!(ax, x[2:(end-1)], y[2:(end-1)], ω;
        levels
    )
    display(f)
    save("output/vorticity.png", f, pt_per_unit = 2)
end
