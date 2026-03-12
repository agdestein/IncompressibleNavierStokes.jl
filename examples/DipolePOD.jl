# # Two-dimensional dipole
#
# Study of a staggered fourth‐order compact scheme for unsteady incompressible viscous flows 
# Knikker, 2009, International Journal for Numerical Methods in Fluids 
# <https://onlinelibrary.wiley.com/doi/10.1002/fld.1854>

using Adapt
using CUDA
using CUDSS
using IncompressibleNavierStokes
using CairoMakie
using LinearAlgebra
using WGLMakie

getbackend() = CUDA.functional() ? CUDABackend() : IncompressibleNavierStokes.KernelAbstractions.CPU()

# Parameters
n = 1024                 # Number of grid points in each direction
stretch = 1.0            # Grid stretching factor (`nothing` for uniform grid)
lims = -1.0, 1.0         # Domain limits
viscosity = 1 / 2500     # Viscosity
tlims = 0.0, 1.0         # Simulation time
cfl = 0.9                # CFL number
nplot = 100              # How often to plot
nlog = 100               # How often to log
nsave = 20               # How often to save snapshot
dipole() = (;
    r0 = 0.1,            # Initial vortex radius
    x1 = 0.0, y1 = 0.1,  # Position of first vortex
    x2 = 0.0, y2 = -0.1, # Position of second vortex
    initialenergy = 2.0, # Initial kinetic energy
)
output = joinpath(@__DIR__, "output/dipole") |> mkpath # Output directory

# Setup
ax = isnothing(stretch) ? range(lims..., n + 1) : tanh_grid(lims..., n, stretch)
setup = Setup(;
    x = (ax, ax),
    boundary_conditions = (;
        u = (
            (DirichletBC(), DirichletBC()),
            (DirichletBC(), DirichletBC()),
        ),
    ),
    backend = getbackend(),
)

psolver = default_psolver(setup)

function U(dim, x, y)
    @inline
    (; x1, x2, y1, y2, r0) = dipole()
    r1 = (x - x1)^2 + (y - y1)^2
    r2 = (x - x2)^2 + (y - y2)^2
    d1 = ifelse(dim == 1, -(y - y1), x - x1)
    d2 = ifelse(dim == 1, y - y2, -(x - x2))
    return d1 * exp(-r1 / r0^2) + d2 * exp(-r2 / r0^2)
end

u = velocityfield(setup, U; psolver)

# Scale velocity to have the correct initial kinetic energy
let
    (; initialenergy) = dipole()
    u1 = selectdim(u, 3, 1)
    u2 = selectdim(u, 3, 2)
    Δ1x = setup.Δu[1]
    Δ1y = setup.Δ[2]'
    Δ2x = setup.Δ[1]
    Δ2y = setup.Δu[2]'
    eu = @. Δ1x * Δ1y * u1^2 / 2
    ev = @. Δ2x * Δ2y * u2^2 / 2
    kin = sum(view(eu, setup.Iu[1])) + sum(view(ev, setup.Iu[2]))
    @. u = sqrt(initialenergy / kin) * u
end

# Solve unsteady problem
state, outputs = solve_unsteady(;
    setup,
    start = (; u),
    tlims,
    params = (; viscosity),
    psolver,
    cfl,
    processors = (
        rtp = realtimeplotter(;
            setup,
            plot = fieldplot,
            docolorbar = false,
            nupdate = nplot,
        ),
        log = timelogger(; nupdate = nlog),
        saver = fieldsaver(; setup, nupdate = nsave)
    ),
);

let
    fig = Figure()
    ax = Axis(fig[1, 1];
        title = "Vorticity at final time",
        xlabel = "x",
        ylabel = "y",
        aspect = DataAspect(),
    )
    vort = vorticity(state.u, setup) |> Array
    colorrange = get_lims(vort, 3.0)
    coords = adapt(Array, setup.xp)
    heatmap!(ax, coords..., vort; colormap = :RdBu, colorrange)
    save("$output/dipole_final.png", fig)
    fig
end

# Zoom in (fig 9 of Knikker)
let
    fig = Figure()
    ax = Axis(fig[1, 1];
        title = "Vorticity at final time",
        xlabel = "x",
        ylabel = "y",
        aspect = DataAspect(),
    )
    vort = vorticity(state.u, setup) |> Array
    colorrange = get_lims(vort, 3.0)
    coords = adapt(Array, setup.xp)
    contour!(ax, coords..., vort; levels = 80, color = :grey)
    xlims!(ax, 0.4, 1.0)
    ylims!(ax, 0.0, 0.6)
    save("$output/dipole_final_knikker_fig9.png", fig)
    fig
end

outputs.saver |> length

snapshots = reshape(stack(state -> state.u, outputs.saver), :, length(outputs.saver))

e = svd(snapshots)

let
    fig = Figure()
    ax = Axis(fig[1, 1];
        title = "Singular values",
        xlabel = "Mode number",
        ylabel = "Singular value",
        yscale = log10,
    )
    scatter!(ax, e.S / e.S[1])
    save("$output/dipole_singular_values.png", fig)
    fig
end

let
    fig = Figure()
    ncol = 3
    nrow = 2
    modelist = [1, 2, 3, 5, 10, 20]
    coords = adapt(Array, setup.xp)
    for ilin in 1:nrow*ncol
        j, i = CartesianIndices((ncol, nrow))[ilin].I
        imode = modelist[ilin]
        ax = Axis(fig[i, j];
            title = "Mode $imode",
            xlabel = "x",
            ylabel = "y",
            xlabelvisible = i == nrow,
            ylabelvisible = j == 1,
            xticklabelsvisible = i == nrow,
            yticklabelsvisible = j == 1,
            aspect = DataAspect(),
        )
        mode = reshape(e.U[:, imode], size(u)) |> adapt(getbackend())
        vort = vorticity(mode, setup) |> Array
        colorrange = get_lims(vort, 5.0)
        heatmap!(ax, coords..., vort; colormap = :RdBu, colorrange)
    end
    Label(fig[0, :]; text = "Vorticity of POD modes", font = :bold)
    save("$output/dipole_modes.png", fig)
    fig
end
