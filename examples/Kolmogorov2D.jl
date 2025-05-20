# # Kolmogorov flow (2D)
#
# The Kolmogorov flow in a periodic box ``\Omega = [0, 1]^2`` is initiated
# via the force field
#
# ```math
# f(x, y) =
# \begin{pmatrix}
#     \sin(\pi k y) \\
#     0
# \end{pmatrix}
# ```
#
# where `k` is the wavenumber where energy is injected.

# ## Packages
#
# We just need the `IncompressibleNavierStokes` and a Makie plotting package.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# ## Setup
#
# Define a uniform grid with a steady body force field.

n = 256
axis = range(0.0, 1.0, n + 1)
setup = Setup(; x = (axis, axis), visc = 5e-4);
u = random_field(setup, 0.0; A = 1e-2);

# This is the right-hand side force in the momentum equation
# By default, it is just `navierstokes!`. Here we add a
# pre-computed body force.
function force!(f, state, t, params, setup, cache)
    navierstokes!(f, state, t, nothing, setup, nothing)
    f.u .+= cache.bodyforce
end

# Tell IncompressibleNavierStokes how to prepare the cache for `force!`.
# The cache is created before time stepping begins.
function IncompressibleNavierStokes.get_cache(::typeof(force!), setup)
    f(dim, x, y) = (dim == 1) * 5 * sinpi(8 * y)
    bodyforce = velocityfield(setup, f; doproject = false)
    (; bodyforce)
end

# ## Plot body force
#
# Since the force is steady, it is just stored as a field.
let
    (; bodyforce) = IncompressibleNavierStokes.get_cache(force!, setup)
    heatmap(bodyforce[:, :, 1])
end

# ## Solve unsteady problem

state, outputs = solve_unsteady(;
    setup,
    force!,
    start = (; u),
    tlims = (0.0, 2.0),
    processors = (
        rtp = realtimeplotter(; setup, nupdate = 100),
        ehist = realtimeplotter(;
            setup,
            plot = energy_history_plot,
            nupdate = 10,
            displayfig = false,
        ),
        espec = realtimeplotter(;
            setup,
            plot = energy_spectrum_plot,
            nupdate = 10,
            displayfig = false,
        ),
        log = timelogger(; nupdate = 100),
    ),
);

# Field plot
outputs.rtp

# Energy history
outputs.ehist

# Energy spectrum
outputs.espec
