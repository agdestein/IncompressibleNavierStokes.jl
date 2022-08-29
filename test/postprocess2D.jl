@testset "Postprocess 2D" begin

    # Taylor-Green vortex case (TG).

    # Viscosity model
    viscosity_model = LaminarModel(; Re = 2000.0)

    # Boundary conditions
    u_bc(x, y, t) = zero(x)
    v_bc(x, y, t) = zero(x)
    boundary_conditions = BoundaryConditions(
        u_bc,
        v_bc;
        bc_unsteady = false,
        bc_type = (;
            u = (; x = (:periodic, :periodic), y = (:periodic, :periodic)),
            v = (; x = (:periodic, :periodic), y = (:periodic, :periodic)),
        ),
    )

    # Grid
    x = stretched_grid(0, 2π, 10)
    y = stretched_grid(0, 2π, 10)
    grid = Grid(x, y; boundary_conditions)
 
    @test plot_grid(grid) isa Makie.FigureAxisPlot
    @test plot_grid(x, y) isa Makie.FigureAxisPlot


    # Forcing parameters
    bodyforce_u(x, y) = 0
    bodyforce_v(x, y) = 0
    force = SteadyBodyForce(bodyforce_u, bodyforce_v, grid)

    # Build setup and assemble operators
    setup = Setup(; viscosity_model, grid, force, boundary_conditions)

    # Pressure solver
    pressure_solver = FourierPressureSolver(setup)

    # Time interval
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
    processors = [logger, plotter, writer, tracer]

    # Solve unsteady problem
    problem = UnsteadyProblem(setup, V₀, p₀, tlims)
    V, p = solve(problem, RK44(); Δt = 0.01, processors, pressure_solver)

    @testset "VTK files" begin
        @test isfile("output/solution2D.pvd")
        @test isfile("output/solution2D_t=0p0.vtr")

        save_vtk(V, p, t_end, setup, "output/field2D")
        @test isfile("output/field2D.vtr")
    end

    @testset "Plot fields" begin
        typeof(plot_tracers(tracer))
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
            filename = "output/vorticity.gif",
            nframe = 10,
            nsubframe = 4,
        )
        @test isfile("output/vorticity.gif")
    end
end
