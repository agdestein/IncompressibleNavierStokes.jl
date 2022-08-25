@testset "Postprocess 2D" begin

    # Taylor-Green vortex case (TG).

    # Viscosity model
    viscosity_model = LaminarModel(; Re = 2000.0)

    # Boundary conditions
    u_bc(x, y, z, t) = zero(x)
    v_bc(x, y, z, t) = zero(x)
    w_bc(x, y, z, t) = zero(x)
    boundary_conditions = BoundaryConditions(
        u_bc,
        v_bc,
        w_bc;
        bc_unsteady = false,
        bc_type = (;
            u = (;
                x = (:periodic, :periodic),
                y = (:periodic, :periodic),
                z = (:periodic, :periodic),
            ),
            v = (;
                x = (:periodic, :periodic),
                y = (:periodic, :periodic),
                z = (:periodic, :periodic),
            ),
            w = (;
                x = (:periodic, :periodic),
                y = (:periodic, :periodic),
                z = (:periodic, :periodic),
            ),
        ),
    )

    # Grid
    x = stretched_grid(0, 2π, 10)
    y = stretched_grid(0, 2π, 10)
    z = stretched_grid(0, 2π, 10)
    grid = Grid(x, y, z; boundary_conditions)

    @test plot_grid(grid) isa Makie.Figure

    # Forcing parameters
    bodyforce_u(x, y, z) = 0.0
    bodyforce_v(x, y, z) = 0.0
    bodyforce_w(x, y, z) = 0.0
    force = SteadyBodyForce(bodyforce_u, bodyforce_v, bodyforce_w, grid)

    # Build setup and assemble operators
    setup = Setup(; viscosity_model, grid, force, boundary_conditions)

    # Pressure solver
    pressure_solver = FourierPressureSolver(setup)

    # Time interval
    t_start, t_end = tlims = (0.0, 1.0)

    # Initial conditions
    initial_velocity_u(x, y, z) = sin(x)cos(y)cos(z)
    initial_velocity_v(x, y, z) = -cos(x)sin(y)cos(z)
    initial_velocity_w(x, y, z) = 0.0
    initial_pressure(x, y, z) = 1 / 4 * (cos(2x) + cos(2y) + cos(2z))
    V₀, p₀ = create_initial_conditions(
        setup,
        t_start;
        initial_velocity_u,
        initial_velocity_v,
        initial_velocity_w,
        initial_pressure,
        pressure_solver,
    )

    # Iteration processors
    logger = Logger()
    plotter = RealTimePlotter(; nupdate = 5, type = contour, fieldname = :vorticity)
    writer = VTKWriter(; nupdate = 5, dir = "output", filename = "solution3D")
    tracer = QuantityTracer(; nupdate = 1)
    processors = [logger, plotter, writer, tracer]

    # Solve unsteady problem
    problem = UnsteadyProblem(setup, V₀, p₀, tlims)
    V, p = solve(problem, RK44(); Δt = 0.01, processors, pressure_solver)

    @testset "VTK files" begin
        @test isfile("output/solution3D.pvd")
        @test isfile("output/solution3D_t=0p0.vtr")
        save_vtk(V, p, t_end, setup, "output/field3D")
        @test isfile("output/field3D.vtr")
    end

    @testset "Plot fields" begin
        typeof(plot_pressure(setup, p))
        @test plot_tracers(tracer) isa Figure
        @test plot_pressure(setup, p) isa Makie.FigureAxisPlot
        @test plot_velocity(setup, V, t_end) isa Makie.FigureAxisPlot
        @test plot_vorticity(setup, V, tlims[2]) isa Makie.FigureAxisPlot
        @test_broken plot_streamfunction(setup, V, tlims[2]) isa Makie.FigureAxisPlot
        @test plot_force(setup, setup.force.F, t_end) isa Makie.FigureAxisPlot
    end

end
