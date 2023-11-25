# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

# # Taylor-Green vortex - 2D
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

function compute_convergence(; D, nlist, lims, Re, tlims, Δt, uref, ArrayType = Array)
    T = typeof(lims[1])
    e = zeros(T, length(nlist))
    for (i, n) in enumerate(nlist)
        @info "Computing error for n = $n"
        x = ntuple(α -> LinRange(lims..., n + 1), D)
        setup = Setup(x...; Re, ArrayType)
        pressure_solver = SpectralPressureSolver(setup)
        u₀, p₀ = create_initial_conditions(
            setup,
            (dim, x...) -> uref(dim, x..., tlims[1]),
            tlims[1];
            pressure_solver,
        )
        ut, pt = create_initial_conditions(
            setup,
            (dim, x...) -> uref(dim, x..., tlims[2]),
            tlims[2];
            pressure_solver,
        )
        u, p, outputs = solve_unsteady(setup, u₀, p₀, tlims; Δt, pressure_solver)
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

solution(Re) =
    (dim, x, y, t) -> (dim() == 1 ? -sin(x) * cos(y) : cos(x) * sin(y)) * exp(-2t / Re)

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
with_theme(;
# linewidth = 5,
# markersize = 20,
# fontsize = 20,
) do
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        xscale = log10,
        yscale = log10,
        xticks = nlist,
        xlabel = "n",
        title = "Relative error",
    )
    scatterlines!(nlist, e; label = "Data")
    lines!(collect(extrema(nlist)), n -> n^-2.0; linestyle = :dash, label = "n^-2")
    axislegend()
    fig
end
