function testcase()
    # Choose where to put output
    basedir = haskey(ENV, "DEEPDIP") ? ENV["DEEPDIP"] : joinpath(@__DIR__, "..")
    outdir = mkpath(joinpath(basedir, "output", "kolmogorov"))

    seed_dns = 123
    T = Float32

    params = (;
        D = 2,
        lims = (T(0), T(1)),
        Re = T(6e3),
        tburn = T(0.5),
        tsim = T(5),
        savefreq = 50,
        ndns = 4096,
        nles = [32, 64, 128],
        filters = [FaceAverage()],
        icfunc = (setup, psolver, rng) ->
            random_field(setup, T(0); kp = 20, psolver, rng),
        method = RKMethods.LMWray3(; T),
        bodyforce = (dim, x, y, t) -> (dim == 1) * 5 * sinpi(8 * y),
        issteadybodyforce = true,
    )
end
