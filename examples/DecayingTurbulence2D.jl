# # Decaying Homogeneous Isotropic Turbulence - 2D
#
# In this example we consider decaying homogeneous isotropic turbulence,
# similar to the cases considered in [Kochkov2021](@cite) and
# [Kurz2022](@cite). The initial velocity field is created randomly, but with a
# specific energy spectrum. Due to viscous dissipation, the turbulent features
# eventually group to form larger visible eddies.

# ## Packages
#
# We just need IncompressibleNavierStokes and a Makie plotting backend.

if false                       #src
    include("src/Examples.jl") #src
end                            #src

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Setup
n = 256
ax = LinRange(0.0, 1.0, n + 1)
setup = Setup(; x = (ax, ax), Re = 4e3);
u = random_field(setup, 0.0);

# Solve unsteady problem
state, outputs = solve_unsteady(;
    setup,
    start = (; u),
    tlims = (0.0, 1.0),
    processors = (
        rtp = realtimeplotter(; setup, nupdate = 10),
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

#md # ```@raw html
#md # <video src="/DecayingTurbulence2D.mp4" controls="controls" autoplay="autoplay" loop="loop"></video>
#md # ```

# ## Post-process
#
# We may visualize or export the computed fields

# Energy history
outputs.ehist

# Energy spectrum
outputs.espec

# Plot field
fieldplot(state; setup)
