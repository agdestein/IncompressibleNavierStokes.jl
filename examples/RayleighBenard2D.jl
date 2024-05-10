#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Hardware
ArrayType = Array

## using CUDA, CUDSS
## ArrayType = CuArray

# Define observer function to track Nusselt numbers
# on top and bottom plates.
function nusseltplot(state; setup)
    state isa Observable || (state = Observable(state))
    (; Δ, Δu) = setup.grid
    T = eltype(Δ[1])
    Δy1 = Δu[2][1:1] |> sum
    Δy2 = Δu[2][end-1:end-1] |> sum
    _Nu1 = Point2f[]
    Nu1 = lift(state) do (; temp, t)
        dTdy = @. (temp[:, 2] - temp[:, 1]) / Δy1
        Nu = sum((.-dTdy.*Δ[1])[2:end-1])
        push!(_Nu1, Point2f(t, Nu))
    end
    _Nu2 = Point2f[]
    Nu2 = lift(state) do (; temp, t)
        dTdy = @. (temp[:, end-1] - temp[:, end-2]) / Δy2
        Nu = sum((.-dTdy.*Δ[1])[2:end-1])
        push!(_Nu2, Point2f(t, Nu))
    end
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
    (; x, Δ, Ip, Δu) = setup.grid
    T = eltype(Δ[1])
    Ty = lift(state) do (; temp, t)
        Ty = sum(temp[Ip.indices[1], :] .* Δ[1][Ip.indices[1]]; dims = 1)
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

# Instabilities should depend on the floating point precisision.
# Try both `Float32` and `Float64`.
T = Float32

# Temperature equation setup.
temperature = temperature_equation(;
    Pr = T(0.71),
    Ra = T(1e6),
    Ge = T(0.1),
    dodissipation = true,
    boundary_conditions = (
        (SymmetricBC(), SymmetricBC()),
        (DirichletBC(T(1)), DirichletBC(T(0))),
    ),
    gdir = 2,
    nondim_type = 1,
)

# Grid
n = 50
x = tanh_grid(T(0), T(1), n, T(1.2))
y = tanh_grid(T(0), T(1), n, T(1.2))
plotgrid(x, y)

# Setup
setup = Setup(
    x,
    y;
    boundary_conditions = ((DirichletBC(), DirichletBC()), (DirichletBC(), DirichletBC())),
    Re = 1 / temperature.α1,
    temperature,
    ArrayType,
);

# Initial conditions
ustart = create_initial_conditions(setup, (dim, x, y) -> zero(x));
(; xp) = setup.grid;
## T0(x, y) = 1 - y;
T0(x, y) = 1 - y + max(sinpi(3 * x) / 1000, 0); ## Perturbation
tempstart = T0.(xp[1], xp[2]');

# Solve equation
state, outputs = solve_unsteady(;
    setup,
    ustart,
    tempstart,
    tlims = (T(0), T(40)),
    Δt = T(5e-3),
    processors = (;
        # rtp = realtimeplotter(;
        #     setup,
        #     nupdate = 50,
        #     fieldname = :temperature,
        #     colorrange = (T(0), T(1)),
        #     size = (600, 500),
        # ),
        nusselt = realtimeplotter(; setup, plot = nusseltplot, nupdate = 100),
        avg = realtimeplotter(; setup, plot = averagetemp, nupdate = 50),
        log = timelogger(; nupdate = 20),
    ),
);

# Field

fieldplot(state; setup, fieldname = :temperature)

# Nusselt numbers

outputs.nusselt

# Average temperature

outputs.avg
