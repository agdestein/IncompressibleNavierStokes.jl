"""
    plot_streamfunction(setup, V, t)

Plot streamfunction.
"""
function plot_streamfunction end

# 2D version
function plot_streamfunction(setup::Setup{T,2}, V, t) where {T}
    (; bc) = setup
    (; x, y, xlims, ylims) = setup.grid

    if all(==(:periodic), (bc.u.x[1], bc.v.y[1]))
        xψ = x
        yψ = y
    else
        xψ = x[2:end-1]
        yψ = y[2:end-1]
    end

    # Get fields
    ψ = get_streamfunction(V, t, setup)

    # Plot stream function
    levels = [
        minimum(ψ) - 1
        -0.1175
        -0.115
        -0.11
        -0.1
        -0.09
        -0.07
        -0.05
        -0.03
        -0.01
        -0.0001
        -1.0e-5
        -1.0e-10
        0.0
        1.0e-6
        1.0e-5
        5.0e-5
        0.0001
        0.00025
        0.0005
        0.001
        0.0015
        maximum(ψ) + 1
    ]
    fig = Figure()
    ax = Axis(
        fig[1, 1],
        aspect = DataAspect(),
        title = "Stream function ψ",
        xlabel = "x",
        ylabel = "y",
    )
    limits!(ax, xlims[1], xlims[2], ylims[1], ylims[2])
    contourf!(
        ax,
        xψ,
        yψ,
        ψ;
        # levels,
        extendlow = :auto,
        extendhigh = :auto,
    )
    # save("output/streamfunction.png", fig, pt_per_unit = 2)

    fig
end

# 3D version
function plot_streamfunction(setup::Setup{T,3}, V, t) where {T}
    error("Not implemented (3D)")
end
