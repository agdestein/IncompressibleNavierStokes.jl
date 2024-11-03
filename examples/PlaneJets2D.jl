# # Plane jets - 2D
#
# Plane jets example, as presented in [MacArt2021](@cite). Note that the
# original formulation is in 3D.

# We start by loading packages.
# A [Makie](https://github.com/JuliaPlots/Makie.jl) plotting backend is needed
# for plotting. `GLMakie` creates an interactive window (useful for real-time
# plotting), but does not work when building this example on GitHub.
# `CairoMakie` makes high-quality static vector-graphics plots.

using FFTW
#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes
using LaTeXStrings

# Output directory
outdir = joinpath(@__DIR__, "output", "PlaneJets2D")

# Floating point type
T = Float64

# Backend
backend = CPU()
## using CUDA; backend = CUDABackend()

# Reynolds number
Re = T(6_000)

# Test cases (A, B, C, D; in order)
## V() = sqrt(T(467.4))
V() = T(21.619435700313733)

U_A(y) = V() / 2 * (tanh((y + T(0.5)) / T(0.1)) - tanh((y - T(0.5)) / T(0.1)))

U_B(y) =
    V() / 2 * (tanh((y + 1 + T(0.5)) / T(0.1)) - tanh((y + 1 - T(0.5)) / T(0.1))) +
    V() / 2 * (tanh((y - 1 + T(0.5)) / T(0.1)) - tanh((y - 1 - T(0.5)) / T(0.1)))

U_C(y) =
    V() / 2 * (
        tanh(((y + T(1.0)) / 1 + T(0.5)) / T(0.1)) -
        tanh(((y + T(1.0)) / 1 - T(0.5)) / T(0.1))
    ) +
    V() / 4 * (
        tanh(((y - T(1.5)) / 2 + T(0.5)) / T(0.2)) -
        tanh(((y - T(1.5)) / 2 - T(0.5)) / T(0.2))
    )

U_D(y) =
    V() / 2 * (
        tanh(((y + T(1.0)) / 1 + T(0.5)) / T(0.1)) -
        tanh(((y + T(1.0)) / 1 - T(0.5)) / T(0.1))
    ) -
    V() / 4 * (
        tanh(((y - T(1.5)) / 2 + T(0.5)) / T(0.2)) -
        tanh(((y - T(1.5)) / 2 - T(0.5)) / T(0.2))
    )

## U(y) = U_A(y)
## U(y) = U_B(y)
U(y) = U_C(y)
## U(y) = U_D(y)

# Random noise to stimulate turbulence
U(x, y) = (1 + T(0.1) * (rand(T) - T(0.5))) * U(y)

## boundary_conditions = (
##     (PeriodicBC(), PeriodicBC()),
##     (PressureBC(), PressureBC())
## )

# A 2D grid is a Cartesian product of two vectors
n = 64
## n = 128
## n = 256
x = LinRange(T(0), T(16), 4n + 1), LinRange(-T(10), T(10), 5n + 1)
plotgrid(x...)

# Build setup and assemble operators
setup = Setup(x, Re, backend);
## setup = Setup(; x, Re, boundary_conditions, backend);

# Initial conditions
ustart = velocityfield(setup, (dim, x, y) -> dim == 1 ? U(x, y) : zero(x));

# Real time plot: Streamwise average and spectrum
function meanplot(state; setup)
    (; xp, Iu, Ip, Nu, N) = setup.grid

    umean = lift(state) do (; u, p, t)
        reshape(sum(u[1][Iu[1]]; dims = 1), :) ./ Nu[1][1] ./ V()
    end

    K = Nu[1][2] ÷ 2
    k = 1:(K-1)

    ## Find energy spectrum where y = 0
    n₀ = findmin(abs, xp[2])[2]
    E₀ = lift(state) do (; u, p, t)
        u_y = u[1][:, n₀]
        abs.(fft(u_y .^ 2))[k.+1]
    end
    y₀ = xp[2][n₀]

    ## Find energy spectrum where y = 1
    n₁ = findmin(y -> abs(y - 1), xp[2])[2]
    E₁ = lift(state) do (; u, p, t)
        u_y = u[1][:, n₁]
        abs.(fft(u_y .^ 2))[k.+1]
    end
    y₁ = xp[2][n₁]

    fig = Figure()
    ax = Axis(
        fig[1, 1];
        title = "Mean streamwise flow",
        xlabel = "y",
        ylabel = L"\langle u \rangle / U_0",
    )
    lines!(ax, xp[2][2:end-1], umean)
    ax = Axis(
        fig[1, 2];
        title = "Streamwise energy spectrum",
        xscale = log10,
        yscale = log10,
        xlabel = L"k_x",
        ylabel = L"\hat{U}_{cl} / U_0",
    )
    ## ylims!(ax, (10^(0.0), 10^4.0))
    ksub = k[10:end]
    ## lines!(ax, ksub, 1000 .* ksub .^ (-5 / 3); label = L"k^{-5/3}")
    lines!(ax, ksub, 1e7 .* ksub .^ -3; label = L"k^{-3}")
    scatter!(ax, k, E₀; label = "y = $y₀")
    scatter!(ax, k, E₁; label = "y = $y₁")
    axislegend(ax; position = :lb)
    ## on(_ -> autolimits!(ax), E₁)

    fig
end

# Solve unsteady problem
state, outputs = solve_unsteady(;
    setup,
    ustart,
    tlims = (T(0), T(1)),
    method = RKMethods.RK44P2(),
    Δt = 0.001,
    processors = (
        rtp = realtimeplotter(;
            setup,
            ## plot = fieldplot,
            ## plot = energy_history_plot,
            ## plot = energy_spectrum_plot,
            plot = meanplot,
            nupdate = 1,
        ),
        ## anim = animator(; setup, path = "$outdir/vorticity.mkv", nupdate = 4),
        ## vtk = vtk_writer(; setup, nupdate = 10, dir = outdir, filename = "solution"),
        ## field = fieldsaver(; setup, nupdate = 10),
        log = timelogger(; nupdate = 100),
    ),
);

# ## Post-process
#
# We may visualize or export the computed fields `(u, p)`

outputs.rtp

# Export to VTK
save_vtk(state; setup, filename = joinpath(outdir, "solution"))

# Plot pressure
fieldplot(state; setup, fieldname = :pressure)

# Plot initial velocity
fieldplot((; u = u₀, p = p₀, t = T(0)); setup, fieldname = :velocitynorm)

# Plot final velocity
fieldplot(state; setup, fieldname = :velocitynorm)

# Plot vorticity
fieldplot(state; setup, fieldname = :vorticity)

#md # ## Copy-pasteable code
#md #
#md # Below is the full code for this example stripped of comments and output.
#md #
#md # ```julia
#md # CODE_CONTENT
#md # ```
