"""
    plotgrid(x...)

Plot nonuniform Cartesian grid.
"""
function plotgrid end

plotgrid(x, y; kwargs...) = wireframe(
    x,
    y,
    zeros(eltype(x), length(x), length(y));
    axis = (; aspect = DataAspect(), xlabel = "x", ylabel = "y"),
    kwargs...,
)

function plotgrid(x, y, z)
    nx, ny, nz = length(x), length(y), length(z)
    T = eltype(x)

    # x = repeat(x, 1, ny, nz)
    # y = repeat(reshape(y, 1, :, 1), nx, 1, nz)
    # z = repeat(reshape(z, 1, 1, :), nx, ny, 1)
    # vol = repeat(reshape(z, 1, 1, :), nx, ny, 1)
    # volume(x, y, z, vol)
    fig = Figure()

    ax = Axis3(fig[1, 1])
    wireframe!(ax, x, y, fill(z[1], length(x), length(y)))
    wireframe!(ax, x, y, fill(z[end], length(x), length(y)))
    wireframe!(ax, x, fill(y[1], length(z)), repeat(z, 1, length(x))')
    wireframe!(ax, x, fill(y[end], length(z)), repeat(z, 1, length(x))')
    wireframe!(ax, fill(x[1], length(z)), y, repeat(z, 1, length(y)))
    wireframe!(ax, fill(x[end], length(z)), y, repeat(z, 1, length(y)))
    ax.aspect = :data

    ax = Axis(fig[1, 2]; xlabel = "x", ylabel = "y")
    wireframe!(ax, x, y, zeros(T, length(x), length(y)))
    ax.aspect = DataAspect()

    ax = Axis(fig[2, 1]; xlabel = "y", ylabel = "z")
    wireframe!(ax, y, z, zeros(T, length(y), length(z)))
    ax.aspect = DataAspect()

    ax = Axis(fig[2, 2]; xlabel = "x", ylabel = "z")
    wireframe!(ax, x, z, zeros(T, length(x), length(z)))
    ax.aspect = DataAspect()

    fig
end
