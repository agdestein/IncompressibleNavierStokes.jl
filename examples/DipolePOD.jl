# # Two-dimensional dipole
#
# This example uses the setup from the paper
# > Study of a staggered fourth‐order compact scheme for unsteady incompressible viscous flows 
# > Knikker, 2009, International Journal for Numerical Methods in Fluids
# > <https://onlinelibrary.wiley.com/doi/10.1002/fld.1854>
#
# The goal is to fit a reduced-order model (ROM) to describe the evolution of the flow.

# ## Load packages
#
# We use the IncompressibleNavierStokes solver.
# CairoMakie creates static plots, WGLMakie creates interactive plots in a viewer.
# Do `WGLMakie.activate!()` to switch backends.

using Adapt
using CairoMakie
using CUDA
using CUDSS
using JLD2
using LinearAlgebra
using ProgressMeter
using WGLMakie

import IncompressibleNavierStokes as NS

# ## Define functions
#
# We first define all actions in functions, and run them later.
# This makes it convenient to comment out the slow running function calls later and
# just `include("filename.jl")` to run everything.
# It also keeps the namespace clean and avoids too many global variables.

# Define all parameters here for convenience
params() = (;
    ## Fluid
    viscosity = 1 / 2500, # Viscosity

    ## Initial conditions
    r0 = 0.1,             # Initial vortex radius
    p1 = (0.0, 0.1),      # Position of first vortex
    p2 = (0.0, -0.1),     # Position of second vortex
    initialenergy = 2.0,  # Initial kinetic energy

    ## Grid
    n = 512,              # Number of grid points in each direction
    stretch = 1.0,        # Grid stretching factor (`nothing` for uniform grid)
    lims = (-1.0, 1.0),   # Domain limits

    ## Full-order model
    tsim = 1.0,           # Simulation time
    cfl = 0.9,            # CFL number
    nsnapshot = 100,      # How many snapshots to save (in addition to the initial snapshot)

    ## Reduced-order model
    npod = 50,            # Number of POD modes to keep
)

# Inform about snapshot size
let
    (; n, nsnapshot) = params()
    numbersize = 8.0 # Bytes per Float64 number
    ncomponent = 2 # ux and uy
    nfield = 2 # u and dudt
    size = n^2 * ncomponent * nfield * (nsnapshot + 1) * numbersize
    size_gb = round(size / 1e9; sigdigits = 3)
    @info "Storing $size_gb GB of snapshot data in-memory (ensure that you have enough RAM)."
end

# Output directory (for saving plots and snapshots)
output() = mkpath(joinpath(@__DIR__, "output/DipolePOD"))

# If a CUDA-compatible GPU is available, use it. Otherwise, use CPU.
getbackend() =
    if CUDA.functional()
        CUDABackend()
    else
        NS.KernelAbstractions.CPU()
    end

# Domain setup
function getsetup()
    (; n, lims, stretch) = params()
    ax = if isnothing(stretch)
        range(lims..., n + 1)
    else
        NS.tanh_grid(lims..., n, stretch)
    end
    setup = NS.Setup(;
        x = (ax, ax),
        boundary_conditions = (;
            u = (
                (NS.DirichletBC(), NS.DirichletBC()),
                (NS.DirichletBC(), NS.DirichletBC()),
            ),
        ),
        backend = getbackend(),
    )
    return setup
end

# Initial conditions: Two vortices (dipole)
function U(dim, x, y)
    @inline
    (; p1, p2, r0) = params()
    x1, y1 = p1
    x2, y2 = p2
    r1 = (x - x1)^2 + (y - y1)^2
    r2 = (x - x2)^2 + (y - y2)^2
    d1 = ifelse(dim == 1, -(y - y1), x - x1)
    d2 = ifelse(dim == 1, y - y2, -(x - x2))
    return d1 * exp(-r1 / r0^2) + d2 * exp(-r2 / r0^2)
end

# Scale velocity to have the correct initial kinetic energy
function scalevelocity!(u, setup)
    (; initialenergy) = params()
    u1 = selectdim(u, 3, 1)
    u2 = selectdim(u, 3, 2)

    ## Grid spacings of u1 control volume
    Δ1x = setup.Δu[1]
    Δ1y = setup.Δ[2]'

    ## Grid spacings of u2 control volume
    Δ2x = setup.Δ[1]
    Δ2y = setup.Δu[2]'

    ## Kinetic energy weighed by the control volumes
    eu = @. Δ1x * Δ1y * u1^2 / 2
    ev = @. Δ2x * Δ2y * u2^2 / 2

    ## Total kinetic energy
    kin = sum(view(eu, setup.Iu[1])) + sum(view(ev, setup.Iu[2]))

    ## Scale velocity field
    @. u = sqrt(initialenergy / kin) * u

    return
end

# Run full-order model and store snapshots
function fullrun(u, setup, psolver)
    (; viscosity, tsim, cfl, nsnapshot) = params()

    ## Storage structures and allocations
    state = (; u = copy(u))
    method = NS.LMWray3()
    ode_cache = NS.get_cache(method, state, setup)
    force_cache = NS.get_cache(NS.navierstokes!, setup)
    stepper = NS.create_stepper(method; setup, psolver, state, t = 0.0)
    dudt = similar(u)
    snaps = (; u = zeros(length(u), nsnapshot + 1), dudt = zeros(length(u), nsnapshot + 1))
    cpubuf = Array(u)

    ## Step through all the save points
    prog = Progress(nsnapshot; desc = "Running full-order model")
    for isnap = 0:nsnapshot
        ## Step until next save point.
        ## For the first step, the while loop is never entered.
        tstop = isnap / nsnapshot * tsim
        isubstep = 0
        while stepper.t < prevfloat(tstop)
            ## Change timestep based on operators
            Δt =
                cfl *
                NS.propose_timestep(NS.navierstokes!, stepper.state, setup, (; viscosity))

            ## Make sure not to step past `tstop`
            Δt = min(Δt, tstop - stepper.t)

            ## Perform a single time step with the time integration method
            stepper = NS.timestep!(
                method,
                NS.navierstokes!,
                stepper,
                Δt;
                params = (; viscosity),
                ode_cache,
                force_cache,
            )
            isubstep += 1
        end

        ## Compute Navier-Stokes right-hand side (including pressure-projection)
        NS.navierstokes!(
            (; u = dudt),
            state,
            stepper.t;
            setup,
            cache = force_cache,
            viscosity,
        )
        NS.project!(dudt, setup; psolver, ode_cache.p)

        ## Store snapshot-pair (use `cpubuf` since GPU cannot copy to CPU-`view` directly)
        copyto!(cpubuf, state.u)
        copyto!(view(snaps.u, :, isnap + 1), cpubuf)
        copyto!(cpubuf, dudt)
        copyto!(view(snaps.dudt, :, isnap + 1), cpubuf)

        ## Log
        next!(prog; showvalues = [("substeps", isubstep), ("time", stepper.t)])
    end

    ## Save results
    save_object("$(output())/snapshots.jld2", snaps)

    return
end

# ## Main script
#
# Now we run everything.

# Problem setup
setup = getsetup()
psolver = NS.default_psolver(setup)
u = NS.velocityfield(setup, U; psolver)
scalevelocity!(u, setup)

# Run full order model and save snapshots
fullrun(u, setup, psolver)

# Load snapshots
snapshots = load_object("$(output())/snapshots.jld2")

# SVD decomposition of snapshots
decomp = svd(snapshots.u)

# ## Plots
#
# Now plot results.

# Plot grid
NS.plotgrid(adapt(Array, setup.x)...) |> display

# Plot final vorticity
let
    copyto!(u, snapshots.u[:, end])
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        title = "Vorticity at final time",
        xlabel = "x",
        ylabel = "y",
        aspect = DataAspect(),
    )
    vort = NS.vorticity(u, setup) |> Array
    colorrange = NS.get_lims(vort, 3.0)
    coords = adapt(Array, setup.xp)
    heatmap!(ax, coords..., vort; colormap = :RdBu, colorrange)
    save("$(output())/dipole-final.png", fig)
    fig |> display
end

# Plot final vorticity: zoom in (fig 9 of Knikker)
let
    copyto!(u, snapshots.u[:, end])
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        title = "Vorticity at final time",
        xlabel = "x",
        ylabel = "y",
        aspect = DataAspect(),
    )
    vort = NS.vorticity(u, setup) |> Array
    colorrange = NS.get_lims(vort, 3.0)
    coords = adapt(Array, setup.xp)
    contour!(ax, coords..., vort; levels = 80, color = :grey)
    xlims!(ax, 0.4, 1.0)
    ylims!(ax, 0.0, 0.6)
    save("$(output())/dipole-final-knikker-fig9.png", fig)
    fig |> display
end

# Animate flow
let
    (; nsnapshot, tsim) = params()
    copyto!(u, snapshots.u[:, end])
    vort = NS.scalarfield(setup)
    vort_cpu = vort |> Array
    vort_obs = Observable(vort_cpu)
    coords = adapt(Array, setup.xp)
    colorrange = (0.0, 1.0) |> Observable
    title = Observable("")
    fig = Figure()
    ax = Axis(fig[1, 1]; title, xlabel = "x", ylabel = "y", aspect = DataAspect())
    heatmap!(ax, coords..., vort_obs; colormap = :RdBu, colorrange)
    filename = "$(output())/dipole-evolution.mp4"
    tmovie = 5.0
    Makie.record(
        fig,
        filename,
        0:nsnapshot;
        framerate = round(Int, nsnapshot / tmovie),
    ) do i
        copyto!(u, snapshots.u[:, i+1])
        NS.vorticity!(vort, u, setup) |> Array
        tround = round(tsim * i / nsnapshot; sigdigits = 3)
        title[] = "Vorticity at time t = $(tround)"
        vort_obs[] = copyto!(vort_cpu, vort)
        colorrange[] = NS.get_lims(vort, 5.0)
    end
    fig |> display
end

# Plot singular values
let
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        title = "Singular values",
        xlabel = "Mode number",
        ylabel = "Singular value",
        yscale = log10,
    )
    scatter!(ax, decomp.S / decomp.S[1])
    save("$(output())/dipole-singular-values.pdf", fig; backend = CairoMakie)
    fig |> display
end

# Plot vorticity of selected POD modes
let
    fig = Figure()
    ncol = 3
    nrow = 2
    modelist = [1, 2, 3, 5, 10, 20]
    coords = adapt(Array, setup.xp)
    for ilin = 1:(nrow*ncol)
        j, i = CartesianIndices((ncol, nrow))[ilin].I
        imode = modelist[ilin]
        ax = Axis(
            fig[i, j];
            title = "Mode $imode",
            xlabel = "x",
            ylabel = "y",
            xlabelvisible = i == nrow,
            ylabelvisible = j == 1,
            xticklabelsvisible = i == nrow,
            yticklabelsvisible = j == 1,
            aspect = DataAspect(),
        )
        mode = reshape(decomp.U[:, imode], size(u)) |> adapt(getbackend())
        vort = NS.vorticity(mode, setup) |> Array
        colorrange = NS.get_lims(vort, 5.0)
        heatmap!(ax, coords..., vort; colormap = :RdBu, colorrange)
    end
    Label(fig[0, :]; text = "Vorticity of POD modes", font = :bold)
    save("$(output())/dipole-modes.png", fig)
    fig |> display
end

# ## Fit ROM to data
#
# We have snapshots of `u` and `dudt`.

decomp.U |> size

rom_basis = decomp.U[:, 1:params().npod]

a = rom_basis' * snapshots.u
dadt = rom_basis' * snapshots.dudt

# Now fit the operators ``L`` and ``Q`` in the ODE
#
# ```math
# \frac{\mathrm{d} a}{\mathrm{d} t} = L a + Q (a \otimes a)
# ```
#
# With least-squares regression, this can be done as
# ```math
# \min_{x} \| A x - b \|_2^2,
# ```
# which has the closed-form solution
# ```math
# x = (A^T A)^{-1} A^T b,
# ```
# where
# - ``A = [a; a \otimes a]`` are the collection of features (flattened to one long row vector per snapshot),
# - ``b = \mathrm{d} a / \mathrm{d} t`` are the targets (one row vector per snapshot),
# - ``x = [L; Q]`` are the operators (the two input dimensions of ``Q`` should be flattened to one long dimension).
#
# TODO:
# - Build ``A``, ``b``
# - Fit the operators ``x``
# - Reshape everything appropriately
# - Make code for ROM ODE, with appropriate time integrator and time step selector
# - Solve ROM ODE from initial exact POD coordinates `a[1:params().npod, 1]`
# - Compare results with exact POD coordinates `a[1:params().npod, :]`
# - Convert back to FOM-space: `u = rom_basis * a`
# - Make a pretty animation of the ROM solution vs true FOM solution.
#
# Also check the papers
#
# > Henrik Rosenberger, Benjamin Sanderse, and Giovanni Stabile.
# > Exact Operator Inference with Minimal Data.
# > Feb. 2026. doi: 10.48550/arXiv.2506.01244.
#
# for exact operator inference, and
#
# > Josep Plana-Riu et al.
# > Stable Self-Adaptive Timestepping for Reduced Order Models for Incompressible Flows.
# > Dec. 2025. doi: 10.48550/arXiv.2512.04592.
#
# for time step selection for ROMs.
