# # Convergence study: Taylor-Green vortex (2D)
#
# In this example we consider the Taylor-Green vortex.
# In 2D, it has an analytical solution, given by
#
# ```math
# \begin{split}
#     u^1(x, y, t) & = - \sin(x) \cos(y) \mathrm{e}^{-2 t / Re} \\
#     u^2(x, y, t) & = + \cos(x) \sin(y) \mathrm{e}^{-2 t / Re}
# \end{split}
# ```
#
# This allows us to test the convergence of our solver.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes
using LinearAlgebra

"""
Compare numerical solution with analytical solution at final time.
"""
function compute_convergence(; D, nlist, lims, Re, tlims, Δt, uref, ArrayType = Array)
    T = typeof(lims[1])
    e = zeros(T, length(nlist))
    for (i, n) in enumerate(nlist)
        @info "Computing error for n = $n"
        x = ntuple(α -> LinRange(lims..., n + 1), D)
        setup = Setup(x...; Re, ArrayType)
        psolver = SpectralPressureSolver(setup)
        u₀ = create_initial_conditions(
            setup,
            (dim, x...) -> uref(dim, x..., tlims[1]),
            tlims[1];
            psolver,
        )
        ut = create_initial_conditions(
            setup,
            (dim, x...) -> uref(dim, x..., tlims[2]),
            tlims[2];
            psolver,
            doproject = false,
        )
        (; u, t), outputs = solve_unsteady(setup, u₀, tlims; Δt, psolver)
        (; Ip) = setup.grid
        a, b = T(0), T(0)
        for α = 1:D
            a += sum(abs2, u[α][Ip] - ut[α][Ip])
            b += sum(abs2, ut[α][Ip])
        end
        e[i] = sqrt(a) / sqrt(b)
    end
    e
end

# Analytical solution for 2D Taylor-Green vortex
solution(Re) =
    (dim, x, y, t) -> (dim() == 1 ? -sin(x) * cos(y) : cos(x) * sin(y)) * exp(-2t / Re)

# Compute error for different resolutions
Re = 2.0e3
nlist = [2, 4, 8, 16, 32, 64, 128, 256]
e = compute_convergence(;
    D = 2,
    nlist,
    lims = (0.0, 2π),
    Re,
    tlims = (0.0, 2.0),
    Δt = 0.01,
    uref = solution(Re),
)

# Plot convergence
fig = Figure()
ax = Axis(
    fig[1, 1];
    xscale = log10,
    yscale = log10,
    xticks = nlist,
    xlabel = "n",
    title = "Relative error",
)
scatterlines!(ax, nlist, e; label = "Data")
lines!(ax, collect(extrema(nlist)), n -> n^-2.0; linestyle = :dash, label = "n^-2")
axislegend(ax)
fig
