function testcase(backend)
    basedir = haskey(ENV, "DEEPDIP") ? ENV["DEEPDIP"] : joinpath(@__DIR__, "..")
    outdir = mkpath(joinpath(basedir, "output", "Kolmogorov2D"))
    plotdir = mkpath(joinpath(outdir, "plots"))
    seed_dns = 123
    ntrajectory = 8
    T = Float32
    params = (;
        D = 2,
        lims = (T(0), T(1)),
        Re = T(6e3),
        tburn = T(0.5),
        tsim = T(2),
        savefreq = 50,
        ndns = 1024,
        nles = [32, 64, 128],
        filters = [FaceAverage()],
        backend,
        icfunc = (setup, psolver, rng) ->
            random_field(setup, T(0); kp = 20, psolver, rng),
        method = LMWray3(; T),
        bodyforce = (dim, x, y, t) -> (dim == 1) * 5 * sinpi(8 * y),
        issteadybodyforce = true,
        processors = (; log = timelogger(; nupdate = 100)),
    )
    (; outdir, plotdir, seed_dns, ntrajectory, params)
end
