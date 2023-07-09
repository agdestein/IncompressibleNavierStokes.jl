@testset "Postprocess 3D" begin
    @info "Testing 3D processors"

    T = Float32
    lims = (T(0), T(2π))

    n = 10
    x = LinRange(lims..., n)
    y = LinRange(lims..., n)
    z = LinRange(lims..., n)
    setup = Setup(x, y, z)

    @test plot_grid(x, y, z) isa Makie.Figure

    pressure_solver = SpectralPressureSolver(setup)

    t_start, t_end = tlims = (T(0), T(1))

    initial_velocity_u(x, y, z) = sin(x)cos(y)cos(z)
    initial_velocity_v(x, y, z) = -cos(x)sin(y)cos(z)
    initial_velocity_w(x, y, z) = zero(x)
    initial_pressure(x, y, z) = 1 // 4 * (cos(2x) + cos(2y) + cos(2z))
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
        field_plotter(setup; nupdate = 5, displayfig = false),
        vtk_writer(setup; nupdate = 5, dir = "output", filename = "solution3D"),
        animator(
            setup,
            "output/vorticity3D.mp4";
            nupdate = 10,
            plotter = field_plotter(setup; displayfig = false),
        ),
        step_logger(),
    )

    # Solve unsteady problem
    # FIXME: using too many processors results in endless compilation
    V, p, outputs = solve_unsteady(
        setup,
        V₀,
        p₀,
        tlims;
        Δt = T(0.01),
        processors,
        pressure_solver,
        inplace = true,
    )

    @testset "VTK files" begin
        @info "Testing 3D processors: VTK files"
        @test isfile("output/solution3D.pvd")
        @test isfile("output/solution3D_t=0p0.vti")
        save_vtk(setup, V, p, t_end, "output/field3D")
        @test isfile("output/field3D.vti")
    end

    @testset "Plot fields" begin
        @info "Testing 3D processors: Plots"
        @test plot_pressure(setup, p) isa Makie.FigureAxisPlot
        @test plot_velocity(setup, V, t_end) isa Makie.FigureAxisPlot
        @test plot_vorticity(setup, V, tlims[2]) isa Makie.FigureAxisPlot
        @test_broken plot_streamfunction(setup, V, tlims[2]) isa Makie.FigureAxisPlot
        @test plot_force(setup, t_end) isa Makie.FigureAxisPlot
    end

    @testset "Animate" begin
        @info "Testing 3D processors: Animation"
        @test isfile("output/vorticity3D.mp4")
    end
end
