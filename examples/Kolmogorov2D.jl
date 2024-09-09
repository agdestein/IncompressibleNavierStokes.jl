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
setup = Setup(;
    x = (axis, axis),
    Re = 2e3,
    bodyforce = (dim, x, y, t) -> (dim() == 1) * 5 * sinpi(8 * y),
    issteadybodyforce = true,
);
ustart = random_field(setup, 0.0; A = 1e-2);

# ## Plot body force
#
# Since the force is steady, it is just stored as a field.

heatmap(setup.bodyforce[1])

# ## Solve unsteady problem

state, outputs = solve_unsteady(;
    setup,
    ustart,
    tlims = (0.0, 2.0),
    Δt = 1e-3,
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
