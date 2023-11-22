# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

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

# Case name for saving results
name = "PlaneJets2D"

# Floating point type
T = Float64

# Array type
ArrayType = Array
## using CUDA; ArrayType = CuArray
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

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
x = LinRange(T(0), T(16), 4n)
y = LinRange(-T(10), T(10), 5n)
plot_grid(x, y)

# Build setup and assemble operators
setup = Setup(x, y; Re, ArrayType);
## setup = Setup(x, y; Re, boundary_conditions, ArrayType);

# Since the grid is uniform and identical for x and y, we may use a specialized
# spectral pressure solver
pressure_solver = SpectralPressureSolver(setup)

# Initial conditions
u₀, p₀ = create_initial_conditions(
    setup,
    (dim, x, y) -> dim() == 1 ? U(x, y) : zero(x);
    pressure_solver,
);

# Real time plot: Streamwise average and spectrum
mean_plotter(setup; nupdate = 1) = processor(
    function (state)
        (; Ip) = setup.grid

        umean = @lift begin
            (; u, p, t) = $state
            up = IncompressibleNavierStokes.interpolate_u_p(u, setup)
            u1 = u[1]
            reshape(sum(u1[Ip]; dims = 1), :) ./ size(u1, 1) ./ V()
        end

        K = size(Ip, 1) ÷ 2
        k = 1:(K-1)

        # Find energy spectrum where y = 0
        n₀ = size(Ip, 2) ÷ 2
        E₀ = @lift begin
            (; u, p, t) = $state
            u_y = u[1][:, n₀]
            abs.(fft(u_y .^ 2))[k.+1]
        end

        # Find energy spectrum where y = 1
        n₁ = argmin(n -> abs(yin[n] .- 1), 1:Nuy_in)
        E₁ = @lift begin
            (; V, p, t) = $state
            u = V[indu]
            u_y = reshape(u, size(yu))[:, n₁]
            abs.(fft(u_y .^ 2))[k.+1]
        end

        fig = Figure()
        ax = Axis(
            fig[1, 1];
            title = "Mean streamwise flow",
            xlabel = "y",
            ylabel = L"\langle u \rangle / U_0",
        )
        lines!(ax, yu[1, :], umean)
        ax = Axis(
            fig[1, 2];
            title = "Streamwise energy spectrum",
            xscale = log10,
            yscale = log10,
            xlabel = L"k_x",
            ylabel = L"\hat{U}_{cl} / U_0",
        )
        # ylims!(ax, (10^(0.0), 10^4.0))
        ksub = k[10:end]
        lines!(ax, ksub, 1000 .* ksub .^ (-3 / 5); label = L"k^{-3/5}")
        lines!(ax, ksub, 1e7 .* ksub .^ -3; label = L"k^{-3}")
        scatter!(ax, k, E₀; label = "y = $(yin[n₀])")
        scatter!(ax, k, E₁; label = "y = $(yin[n₁])")
        axislegend(ax; position = :lb)

        display(fig)
        fig
    end;
    nupdate,
)

# Solve unsteady problem
toto, p, outputs = solve_unsteady(
    setup,
    u₀,
    p₀,
    (T(0), T(1));
    method = RK44P2(),
    Δt = 0.001,
    pressure_solver,
    processors = (
        # field_plotter(setup; nupdate = 1),
        ## energy_history_plotter(setup; nupdate = 1),
        ## energy_spectrum_plotter(setup; nupdate = 100),
        ## animator(setup, "vorticity.mkv"; nupdate = 4),
        ## vtk_writer(setup; nupdate = 10, dir = "output/$name", filename = "solution"),
        ## field_saver(setup; nupdate = 10),
        mean_plotter(setup),
        step_logger(; nupdate = 1),
    ),
);

# ## Post-process
#
# We may visualize or export the computed fields `(V, p)`

outputs[1]

#-

outputs[2]

# Export to VTK
save_vtk(setup, toto, p, "output/solution")

# Plot pressure
plot_pressure(setup, p)

# Plot velocity
plot_velocity(setup, u₀)
plot_velocity(setup, toto)

# Plot vorticity
plot_vorticity(setup, toto)

# Plot stream function
plot_streamfunction(setup, toto)
