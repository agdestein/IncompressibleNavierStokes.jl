"""
plot_vorticity.
"""
function plot_vorticity(solution, setup)
    @unpack Nx, Ny, Nu, Nv, x, y = setup.grid
    @unpack V, p, t = solution

    # Reshape
    uₕ = @view V[1:Nu]
    vₕ = @view V[Nu+1:Nu+Nv]

    # Get fields
    ω_flat = get_vorticity(V, t, setup)
    ω = reshape(ω_flat, Nx - 1, Ny - 1)

    # Plot vorticity
    levels = [minimum(ω) - 1, -5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, maximum(ω) + 1]
    f = Figure()
    ax = Axis(
        f[1, 1];
        aspect = DataAspect(),
        title = "Vorticity ω",
        xlabel = "x",
        ylabel = "y",
    )
    contourf!(ax, x[2:(end-1)], y[2:(end-1)], ω;
        # levels
    )
    display(f)
end
