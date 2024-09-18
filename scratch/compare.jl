# # DNS, filtered DNS, and LES
#
# In this example we compare DNS, filtered DNS, and LES.

# ## Packages
#
# We need NeuralClosure for the filter definitions.

if false                      #src
    include("src/PaperDC.jl") #src
end                           #src

using CairoMakie
using IncompressibleNavierStokes
using NeuralClosure

const INS = IncompressibleNavierStokes

# Setups
Φ = FaceAverage()
compression = 4
n_dns = 256
n_les = div(n_dns, compression)
dns, les = map((n_dns, n_les)) do n
    ax = LinRange(0.0, 1.0, n + 1)
    setup = Setup(; x = (ax, ax), Re = 1e3)
    psolver = default_psolver(setup)
    u = random_field(setup, 0.0)
    (; setup, psolver, u)
end;

Φ(les.u, dns.u, les.setup, compression);

fieldplot((; dns.u, temp = nothing, t = 0.0); dns.setup)
fieldplot((; les.u, temp = nothing, t = 0.0); les.setup)

# Solve unsteady problem

let
    tstop = 5.0
    cfl = 0.9
    nupdate = 10
    framerate = 24
    sleeptime = 0.01
    fieldname = :vorticity
    filename = "dns_and_les.mp4"

    x1, x2 = 0.3, 0.5
    y1, y2 = 0.5, 0.7
    box = [
        Point2f(x1, y1),
        Point2f(x2, y1),
        Point2f(x2, y2),
        Point2f(x1, y2),
        Point2f(x1, y1),
    ]

    method = RKMethods.RK44()
    stepper_dns = create_stepper(
        method;
        dns.setup,
        dns.psolver,
        u = copy.(dns.u),
        temp = nothing,
        t = 0.0,
    )
    stepper_les = create_stepper(
        method;
        les.setup,
        les.psolver,
        u = copy.(les.u),
        temp = nothing,
        t = 0.0,
    )
    cache_dns = INS.ode_method_cache(method, dns.setup)
    cache_les = INS.ode_method_cache(method, les.setup)
    cflbuf = scalarfield(dns.setup)

    state_dns = Observable(INS.get_state(stepper_dns))
    state_ref = Observable(INS.get_state(stepper_les))
    state_les = Observable(INS.get_state(stepper_les))
    f_dns = observefield(state_dns; dns.setup, fieldname)
    f_ref = observefield(state_ref; les.setup, fieldname)
    f_les = observefield(state_les; les.setup, fieldname)

    colorrange = lift(f -> get_lims(f), f_dns)
    colormap = :seaborn_icefire_gradient

    x_dns, x_les = map((dns, les)) do (; setup)
        (; Ip, xp) = setup.grid
        Array.(getindex.(xp, Ip.indices))
    end

    fig = Figure(; size = (1000, 400))
    ax = (
        Axis(fig[1, 1]; aspect = DataAspect(), title = "DNS"),
        Axis(fig[1, 2]; aspect = DataAspect(), title = "Filtered DNS"),
        Axis(fig[1, 3]; aspect = DataAspect(), title = "LES"),
    )
    heatmap!(ax[1], x_dns..., f_dns; colorrange, colormap)
    heatmap!(ax[2], x_les..., f_ref; colorrange, colormap)
    heatmap!(ax[3], x_les..., f_les; colorrange, colormap)
    lines!(ax[1], box; linewidth = 2, color = :red)
    lines!(ax[2], box; linewidth = 2, color = :red)
    lines!(ax[3], box; linewidth = 2, color = :red)

    display(fig)

    # stream = VideoStream(fig; framerate, visible = false)

    while stepper_dns.t < tstop
        Δt = cfl * INS.get_cfl_timestep!(cflbuf, stepper_dns.u, dns.setup)
        stepper_dns =
            IncompressibleNavierStokes.timestep!(method, stepper_dns, Δt; cache = cache_dns)
        stepper_les =
            IncompressibleNavierStokes.timestep!(method, stepper_les, Δt; cache = cache_les)
        if stepper_dns.n % nupdate == 0
            state_dns[] = INS.get_state(stepper_dns)
            state_les[] = INS.get_state(stepper_les)
            Φ(state_ref[].u, state_dns[].u, les.setup, compression)
            notify(state_ref)
            display(fig)
            # recordframe!(stream)
            sleep(sleeptime)
        end
    end
    # save(filename, stream)
end
