@testitem "Post process" begin
    using CairoMakie
    using LinearAlgebra

    D = 2

    # Temporary directory
    dir = joinpath(tempdir(), "INSTest")

    n = 64
    ax = LinRange(0.0, 1.0, n + 1)
    x = ax, ax
    setup = Setup(;
        x,
        boundary_conditions = (;
            u = ((DirichletBC(), DirichletBC()), (DirichletBC(), DirichletBC()))
        ),
    )
    uref(dim, x, y, args...) = dim == 1 ? -sin(x) * cos(y) : cos(x) * sin(y)
    u = velocityfield(setup, uref, 0.0)

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

    state, outputs = solve_unsteady(;
        setup,
        start = (; u),
        tlims = (0.0, nstep * Δt),
        params = (; viscosity = 1e-2),
        Δt,
        processors,
    )

    @testset "Field saver" begin
        @test length(outputs.field) == nprocess
        @test outputs.field[1].u isa Array
        @test outputs.field[1].t isa Float64
        # Test that different copies are stored
        i = 1
        ii = setup.Iu[i]
        a = outputs.field[1].u[ii, i]
        b = outputs.field[end].u[ii, i]
        @test norm(a - b) / norm(b) > 0.05
    end

    @testset "VTK files" begin
        @test isfile(joinpath(dir, "solution.pvd"))
        @test isfile(joinpath(dir, "solution_t=0p0.vtr"))
        save_vtk(state; setup, filename = joinpath(dir, "snapshot"))
        @test isfile(joinpath(dir, "snapshot.vtr"))
    end

    @testset "Plots" begin
        @test plotgrid(x...) isa Makie.FigureAxisPlot
        @test fieldplot(state; setup) isa Figure
        @test energy_spectrum_plot(state; setup) isa Figure
    end

    @testset "Animation" begin
        @test isfile(joinpath(dir, "solution.mkv"))
    end
end
