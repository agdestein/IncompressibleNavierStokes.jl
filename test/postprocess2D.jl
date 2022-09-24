@testset "Postprocess 2D" begin
    n = 10
    x = LinRange(0, 2π, n)
    y = LinRange(0, 2π, n)

    setup = Setup(x, y)

    @test plot_grid(setup.grid) isa Makie.FigureAxisPlot
    @test plot_grid(x, y) isa Makie.FigureAxisPlot

    pressure_solver = FourierPressureSolver(setup)

    t_start, t_end = tlims = (0.0, 1.0)

    # Initial conditions
    initial_velocity_u(x, y) = -sin(x)cos(y)
    initial_velocity_v(x, y) = cos(x)sin(y)
    initial_pressure(x, y) = 1 / 4 * (cos(2x) + cos(2y))
    V₀, p₀ = create_initial_conditions(
        setup,
        t_start;
        initial_velocity_u,
        initial_velocity_v,
        initial_pressure,
        pressure_solver,
    )

    # Iteration processors
    logger = Logger()
    plotter = RealTimePlotter(; nupdate = 5, fieldname = :vorticity)
    writer = VTKWriter(; nupdate = 5, dir = "output", filename = "solution2D")
    tracer = QuantityTracer(; nupdate = 1)
    observer = StateObserver(1, V₀, p₀, t_start)
    processors = [logger, plotter, writer, tracer, observer]

    # Lift observable (kinetic energy history)
    (; Ωp) = setup.grid
    _E = zeros(0)
    E = @lift begin
        V, p, t = $(observer.state)
        up, vp = get_velocity(V, t, setup)
        up = reshape(up, :)
        vp = reshape(vp, :)
        push!(_E, up' * Diagonal(Ωp) * up + vp' * Diagonal(Ωp) * vp)
    end

    # Solve unsteady problem
    problem = UnsteadyProblem(setup, V₀, p₀, tlims)
    V, p = solve(problem, RK44(); Δt = 0.01, processors, pressure_solver)

    @testset "State observer" begin
        # First @lift, initialize!, and after each of the 100 time steps
        @test length(_E) == 102
        @test all(<(0), diff(_E))
    end

    @testset "VTK files" begin
        @test isfile("output/solution2D.pvd")
        @test isfile("output/solution2D_t=0p0.vtr")
        save_vtk(V, p, t_end, setup, "output/field2D")
        @test isfile("output/field2D.vtr")
    end

    @testset "Plot fields" begin
        @test plot_tracers(tracer) isa Figure
        @test plot_pressure(setup, p) isa Figure
        @test plot_velocity(setup, V, t_end) isa Figure
        @test plot_vorticity(setup, V, t_end) isa Figure
        @test_broken plot_streamfunction(setup, V, t_end) isa Figure
        @test plot_force(setup, t_end) isa Figure
    end

    @testset "Animate" begin
        V, p = solve_animate(
            problem,
            RK44();
            Δt = 4π / 200,
            pressure_solver,
            filename = "output/vorticity2D.gif",
            nframe = 10,
            nsubframe = 4,
        )
        @test isfile("output/vorticity2D.gif")
    end
end
