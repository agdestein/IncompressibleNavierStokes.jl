@testset "Postprocess 2D" begin
    n = 10
    x = LinRange(0, 2π, n)
    y = LinRange(0, 2π, n)

    setup = Setup(x, y)

    @test plot_grid(x, y) isa Makie.FigureAxisPlot

    pressure_solver = FourierPressureSolver(setup)

    t_start, t_end = tlims = (0.0, 1.0)

    # Initial conditions
    initial_velocity_u(x, y) = -sin(x)cos(y)
    initial_velocity_v(x, y) = cos(x)sin(y)
    initial_pressure(x, y) = 1 / 4 * (cos(2x) + cos(2y))
    V₀, p₀ = create_initial_conditions(
        setup,
        initial_velocity_u,
        initial_velocity_v,
        t_start;
        initial_pressure,
        pressure_solver,
    )

    # Iteration processors
    processors = (
        field_plotter(setup; nupdate = 1),
        vtk_writer(setup; nupdate = 5, dir = "output", filename = "solution2D"),
        animator(setup, "output/vorticity2D.mkv"; nupdate = 10),
        step_logger(),
    )

    # Solve unsteady problem
    V, p = solve_unsteady(setup, V₀, p₀, tlims; Δt = 0.01, processors, pressure_solver)

    @testset "VTK files" begin
        @test isfile("output/solution2D.pvd")
        @test isfile("output/solution2D_t=0p0.vtr")
        save_vtk(setup, V, p, t_end, "output/field2D")
        @test isfile("output/field2D.vtr")
    end

    @testset "Plot fields" begin
        @test plot_pressure(setup, p) isa Figure
        @test plot_velocity(setup, V, t_end) isa Figure
        @test plot_vorticity(setup, V, t_end) isa Figure
        @test_broken plot_streamfunction(setup, V, t_end) isa Figure
        @test plot_force(setup, t_end) isa Figure
    end

    @testset "Animate" begin
        @test isfile("output/vorticity2D.mkv")
    end
end
