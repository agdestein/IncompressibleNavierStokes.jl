@testset "Postprocess 3D" begin
    n = 10
    x = LinRange(0, 2π, n)
    y = LinRange(0, 2π, n)
    z = LinRange(0, 2π, n)
    setup = Setup(x, y, z)

    @test plot_grid(x, y, z) isa Makie.Figure

    pressure_solver = FourierPressureSolver(setup)

    t_start, t_end = tlims = (0.0, 1.0)

    initial_velocity_u(x, y, z) = sin(x)cos(y)cos(z)
    initial_velocity_v(x, y, z) = -cos(x)sin(y)cos(z)
    initial_velocity_w(x, y, z) = 0.0
    initial_pressure(x, y, z) = 1 / 4 * (cos(2x) + cos(2y) + cos(2z))
    V₀, p₀ = create_initial_conditions(
        setup,
        initial_velocity_u,
        initial_velocity_v,
        initial_velocity_w,
        t_start;
        initial_pressure,
        pressure_solver,
    )

    # Iteration processors
    processors = (
        field_plotter(setup; nupdate = 5),
        vtk_writer(setup; nupdate = 5, dir = "output", filename = "solution3D"),
        animator(setup, "output/vorticity3D.mkv"; nupdate = 10),
        step_logger(),
    )

    # Solve unsteady problem
    V, p, outputs =
        solve_unsteady(setup, V₀, p₀, tlims; Δt = 0.01, processors, pressure_solver)

    @testset "VTK files" begin
        @test isfile("output/solution3D.pvd")
        @test isfile("output/solution3D_t=0p0.vti")
        save_vtk(setup, V, p, t_end, "output/field3D")
        @test isfile("output/field3D.vti")
    end

    @testset "Plot fields" begin
        @test plot_pressure(setup, p) isa Makie.FigureAxisPlot
        @test plot_velocity(setup, V, t_end) isa Makie.FigureAxisPlot
        @test plot_vorticity(setup, V, tlims[2]) isa Makie.FigureAxisPlot
        @test_broken plot_streamfunction(setup, V, tlims[2]) isa Makie.FigureAxisPlot
        @test plot_force(setup, t_end) isa Makie.FigureAxisPlot
    end

    @testset "Animate" begin
        @test isfile("output/vorticity3D.mkv")
    end
end
