@testset "Processors" begin
    D = 2
    @info "Testing processors in $(D)D"

    # Temporary directory
    dir = joinpath(tempdir(), "INSTest")

    n = 64
    x = LinRange(0.0, 1.0, n + 1), LinRange(0.0, 1.0, n + 1)
    setup = Setup(x...; Re = 100.0)
    ustart = random_field(setup, 0.0)

    nprocess = 20
    nupdate = 10
    nstep = nprocess * nupdate
    Δt = 1e-3

    processors = (;
        rtp = realtimeplotter(; setup, nupdate, displayfig = false),
        anim = animator(;
            setup,
            path = joinpath(dir, "solution.mkv"),
            visible = false,
            nupdate,
        ),
        vtk = vtk_writer(; setup, nupdate, dir, filename = "solution"),
        field = fieldsaver(; setup, nupdate),
        log = timelogger(; nupdate),
    )

    state, outputs =
        solve_unsteady(; setup, ustart, tlims = (0.0, nstep * Δt), Δt, processors)

    @testset "Field saver" begin
        @test length(outputs.field) == nprocess
        @test outputs.field[1].u isa Tuple
        @test outputs.field[1].t isa Float64
        # Test that different copies are stored
        i = 1
        ii = setup.grid.Iu[i]
        a = outputs.field[1].u[i][ii]
        b = outputs.field[end].u[i][ii]
        @test norm(a - b) / norm(b) > 0.05
    end

    @testset "VTK files" begin
        @info "Testing VTK files"
        @test isfile(joinpath(dir, "solution.pvd"))
        @test isfile(joinpath(dir, "solution_t=0p0.vtr"))
        save_vtk(state; setup, filename = joinpath(dir, "snapshot"))
        @test isfile(joinpath(dir, "snapshot.vtr"))
    end

    @testset "Plots" begin
        @info "Testing Plots"
        @test plotgrid(x...) isa Makie.FigureAxisPlot
        @test fieldplot(state; setup) isa Figure
        @test energy_spectrum_plot(state; setup) isa Figure
    end

    @testset "Animation" begin
        @info "Testing 2D processors: Animation"
        @test isfile(joinpath(dir, "solution.mkv"))
    end
end
