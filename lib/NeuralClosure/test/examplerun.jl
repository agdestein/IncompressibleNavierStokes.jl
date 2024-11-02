@testitem "Example run" begin
    using CairoMakie
    using IncompressibleNavierStokes
    using NeuralClosure
    using LinearAlgebra
    using Optimisers
    using Random

    # Parameters
    params = (;
        D = 2,
        lims = (0.0, 1.0),
        Re = 1e3,
        tburn = 5e-2,
        tsim = 0.5,
        savefreq = 2,
        ndns = 32,
        nles = [16, 32],
        filters = (FaceAverage(), VolumeAverage()),
    )

    data = stack(splitseed(123, 3)) do seed
        (; data, t, comptime) = create_les_data(; params..., rng = Xoshiro(seed))
        map(data) do (; u, c)
            (; u, c, t, comptime)
        end
    end

    # Build LES setups and assemble operators
    setups = map(
        nles -> Setup(;
            x = ntuple(α -> range(params.lims..., nles + 1), params.D),
            params.Re,
        ),
        params.nles,
    )

    # Create input/output arrays for a-priori training (ubar vs c)
    io = map(
        I -> create_io_arrays(data[I[1], I[2], :], setups[I[1]]),
        Iterators.product(eachindex(params.nles), eachindex(params.filters)),
    )

    m_cnn = let
        rng = Xoshiro(123)
        closure, θ₀ = cnn(;
            setup = setups[1],
            radii = [2, 2, 2],
            channels = [5, 5, 2],
            activations = [tanh, tanh, identity],
            use_bias = [true, true, false],
            rng,
        )
        (; closure, θ₀)
    end

    m_gcnn = let
        rng = Xoshiro(123)
        closure, θ₀ = gcnn(;
            setup = setups[1],
            radii = [2, 2, 2],
            channels = [5, 5, 1],
            activations = [tanh, tanh, identity],
            use_bias = [true, true, false],
            rng,
        )
        (; closure, θ₀)
    end

    models = m_cnn, m_gcnn

    # Give the CNNs a test run
    models[1].closure(io[1].u[:, :, :, 1:50], models[1].θ₀)
    models[2].closure(io[1].u[:, :, :, 1:50], models[2].θ₀)

    # A-priori training
    let
        rng = Xoshiro(123)
        sample = io[1]
        for (im, m) in enumerate(models)
            dataloader = create_dataloader_prior(sample; batchsize = 10)
            θ = m.θ₀
            loss = create_loss_prior(m.closure)
            opt = Adam(1.0e-3)
            optstate = Optimisers.setup(opt, θ)
            it = rand(rng, 1:size(sample.u, 4), 5)
            validset = map(v -> v[:, :, :, it], sample)
            (; callbackstate, callback) = create_callback(
                create_relerr_prior(m.closure, validset...);
                θ,
                displayref = false,
                displayfig = false,
                displayupdates = false, # Set to `true` if using CairoMakie
                nupdate = 1,
            )
            (; trainstate, callbackstate) = train(;
                dataloader,
                loss,
                trainstate = (; optstate, θ, rng),
                niter = 10,
                callbackstate,
                callback,
            )
        end
    end

    # A-posteriori training
    let
        rng = Random.Xoshiro(123)
        ig, ifil = 1, 1
        for m in models
            setup = setups[ig]
            psolver = psolver_spectral(setup)
            loss = create_loss_post(;
                setup,
                psolver,
                method = RKMethods.RK44(),
                m.closure,
                nsubstep = 1,
            )
            dataloader = create_dataloader_post(
                map(d -> (; d.u, d.t), data[ig, ifil, :]);
                nunroll = 5,
                ntrajectory = 2,
            )
            θ = m.θ₀
            opt = Adam(1.0e-3)
            optstate = Optimisers.setup(opt, θ)
            d = data[ig, ifil, 1]
            it = 1:5
            snap = (; u = d.u[it], t = d.t[it])
            (; callbackstate, callback) = create_callback(
                create_relerr_post(;
                    data = snap,
                    setup,
                    psolver,
                    method = RKMethods.RK44(),
                    closure_model = wrappedclosure(m.closure, setup),
                    nsubstep = 2,
                );
                θ,
                displayref = false,
                displayfig = false,
                displayupdates = false, # Set to `true` if using CairoMakie
                nupdate = 1,
            )
            (; trainstate, callbackstate) = train(;
                dataloader,
                loss,
                trainstate = (; optstate, θ, rng),
                niter = 3,
                callbackstate,
                callback,
            )
        end
    end

    # Compute a-priori errors
    e_prior = let
        e = zeros(length(models))
        ig = 1
        for (im, m) in enumerate(models)
            testset = io[ig]
            err = create_relerr_prior(m.closure, testset...)
            e[im] = err(m.θ₀)
        end
        e
    end

    # Compute a-posteriori errors
    e_post = let
        e = zeros(length(models))
        ig, ifil, itraj = 1, 1, 1
        setup = setups[ig]
        psolver = psolver_spectral(setup)
        traj = data[ig, ifil, itraj]
        it = 1:5
        snaps = (; u = traj.u[it], t = traj.t[it])
        for (im, m) in enumerate(models)
            err = create_relerr_post(;
                data = snaps,
                setup,
                psolver,
                method = RKMethods.RK44(),
                closure_model = wrappedclosure(m.closure, setup),
                nsubstep = 1,
            )
            e[im] = err(m.θ₀)
        end
        e
    end

    # A-priori symmetry errors
    e_symm_prior = let
        e = zeros(length(models))
        ig, ifil, itraj = 1, 1, 1
        for (im, m) in enumerate(models)
            setup = setups[ig]
            setup = (; setup..., closure_model = wrappedclosure(m.closure, setup))
            err = create_relerr_symmetry_prior(; u = data[ig, ifil, itraj].u, setup)
            e[im] = err(m.θ₀)
        end
        e
    end

    # A-posteriori symmetry errors
    e_symm_post = let
        e = zeros(length(models))
        ig, ifil, itraj = 1, 1, 1
        for (im, m) in enumerate(models)
            setup = setups[ig]
            setup = (; setup..., closure_model = wrappedclosure(m.closure, setup))
            traj = data[ig, ifil, itraj]
            err = create_relerr_symmetry_post(;
                u = traj.u[1],
                setup,
                psolver = psolver_spectral(setup),
                Δt = (traj.t[2] - traj.t[1]),
                nstep = 5,
                g = 1,
            )
            e[im] = err(m.θ₀)
        end
        e
    end
end
