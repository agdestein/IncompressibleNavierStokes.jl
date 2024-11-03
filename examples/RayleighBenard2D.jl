# # Rayleigh-Bénard convection (2D)
#
# A hot and a cold plate generate a convection cell in a box.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Output directory for saving results
outdir = joinpath(@__DIR__, "output", "RayleighBenard2D")

# Hardware
backend = CPU()

## using CUDA, CUDSS
## backend = CUDABackend()

# Define observer function to track Nusselt numbers
# on top and bottom plates.
function nusseltplot(state; setup)
    state isa Observable || (state = Observable(state))
    (; Δ, Δu) = setup.grid
    Δy1 = Δu[2][1:1] |> sum
    Δy2 = Δu[2][end-1:end-1] |> sum

    ## Observe Nusselt numbers
    Nu1 = Observable(Point2f[])
    Nu2 = Observable(Point2f[])
    on(state) do (; temp, t)
        dTdy = @. (temp[:, 2] - temp[:, 1]) / Δy1
        Nu = sum((.-dTdy.*Δ[1])[2:end-1])
        push!(Nu1[], Point2f(t, Nu))
        dTdy = @. (temp[:, end-1] - temp[:, end-2]) / Δy2
        Nu = sum((.-dTdy.*Δ[1])[2:end-1])
        push!(Nu2[], Point2f(t, Nu))
        (Nu1, Nu2) .|> notify ## Update plot
    end

    ## Plot Nu history
    fig = Figure()
    ax = Axis(fig[1, 1]; title = "Nusselt number", xlabel = "t", ylabel = "Nu")
    lines!(ax, Nu1; label = "Lower plate")
    lines!(ax, Nu2; label = "Upper plate")
    axislegend(ax)
    on(_ -> autolimits!(ax), Nu2)
    fig
end

# Define observer function to track average temperature.
function averagetemp(state; setup)
    state isa Observable || (state = Observable(state))
    (; xp, Δ, Ip) = setup.grid
    ix = Ip.indices[1]
    Ty = lift(state) do (; temp)
        Ty = sum(temp[ix, :] .* Δ[1][ix]; dims = 1) ./ sum(Δ[1][ix])
        Array(Ty)[:]
    end
    Ty0 = copy(Ty[])
    yy = Array(xp[2])
    fig = Figure()
    ax = Axis(fig[1, 1]; title = "Average temperature", xlabel = "T", ylabel = "y")
    lines!(ax, Ty0, yy; label = "t = 0")
    lines!(ax, Ty, yy; label = "t = t")
    axislegend(ax)
    on(_ -> autolimits!(ax), Ty)
    fig
end

# Instabilities should depend on the floating point precision.
# Try both `Float32` and `Float64`.
T = Float32

# Temperature equation setup.
temperature = temperature_equation(;
    Pr = T(0.71),
    Ra = T(1e7),
    Ge = T(1.0),
    dodissipation = true,
    boundary_conditions = (
        (SymmetricBC(), SymmetricBC()),
        (DirichletBC(T(1)), DirichletBC(T(0))),
    ),
    gdir = 2,
    nondim_type = 1,
)

# Grid
n = 100
x = tanh_grid(T(0), T(2), 2n, T(1.2)), tanh_grid(T(0), T(1), n, T(1.2))
plotgrid(x...)

# Setup
setup = Setup(;
    x,
    boundary_conditions = ((DirichletBC(), DirichletBC()), (DirichletBC(), DirichletBC())),
    Re = 1 / temperature.α1,
    temperature,
    backend,
);

# Initial conditions
ustart = velocityfield(setup, (dim, x, y) -> zero(x));
tempstart = temperaturefield(setup, (x, y) -> one(y) / 2 + max(sinpi(20 * x) / 100, 0));

# Processors
GLMakie.closeall() #!md
processors = (;
    rtp = realtimeplotter(;
        screen = GLMakie.Screen(), #!md
        setup,
        fieldname = :temperature,
        colorrange = (T(0), T(1)),
        size = (600, 350),
        colormap = :seaborn_icefire_gradient,
        nupdate = 20,
    ),
    nusselt = realtimeplotter(;
        screen = GLMakie.Screen(), #!md
        setup,
        plot = nusseltplot,
        nupdate = 20,
    ),
    avg = realtimeplotter(;
        screen = GLMakie.Screen(), #!md
        setup,
        plot = averagetemp,
        nupdate = 50,
    ),
    log = timelogger(; nupdate = 1000),
)

# Solve equation
state, outputs = solve_unsteady(;
    setup,
    ustart,
    tempstart,
    tlims = (T(0), T(20)),
    Δt = T(1e-2),
    processors,
);

#md # ```@raw html
#md # <video src="/RayleighBenard2D.mp4" controls="controls" autoplay="autoplay" loop="loop"></video>
#md # ```

# Nusselt numbers

outputs.nusselt

# Average temperature

outputs.avg

#md # ## Copy-pasteable code
#md #
#md # Below is the full code for this example stripped of comments and output.
#md #
#md # ```julia
#md # CODE_CONTENT
#md # ```
