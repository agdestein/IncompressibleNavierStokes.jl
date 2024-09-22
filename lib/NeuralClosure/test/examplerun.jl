@testitem "Example run" begin
    using CairoMakie
    using IncompressibleNavierStokes
    using LinearAlgebra
    using Optimisers
    using Random

    # Parameters
    rng = Xoshiro(123)
    params = (;
        D = 2,
        Re = 1e3,
        lims = (0.0, 1.0),
        nles = [16, 32],
        ndns = 32,
        filters = (FaceAverage(),),
        tburn = 5e-2,
        tsim = 0.5,
        rng,
        savefreq = 1,
    )

    data = [create_les_data(; params...) for _ = 1:3]

    # Build LES setups and assemble operators
    setups = map(
        nles ->
            Setup(; x = ntuple(α -> LinRange(0.0, 1.0, nles + 1), params.D), params.Re),
        params.nles,
    )

    # Create input/output arrays for a-priori training (ubar vs c)
    io = create_io_arrays(data, setups)

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
        ig = 1
        for (im, m) in enumerate(models)
            dataloader = create_dataloader_prior(io[ig]; batchsize = 10, rng)
            θ = m.θ₀
            loss = create_loss_prior(mean_squared_error, m.closure)
            opt = Adam(1.0e-3)
            optstate = Optimisers.setup(opt, θ)
            it = rand(rng, 1:size(io[ig].u, 4), 5)
            validset = map(v -> v[:, :, :, it], io[ig])
            (; callbackstate, callback) = create_callback(
                create_relerr_prior(m.closure, validset...);
                θ,
                displayref = false,
                displayfig = false,
                display_each_iteration = false, # Set to `true` if using CairoMakie
            )
            (; optstate, θ, callbackstate) = train(;
                dataloader,
                loss,
                optstate,
                θ,
                niter = 10,
                ncallback = 1,
                callbackstate,
                callback,
            )
        end
    end

    # A-posteriori training
    let
        rng = Random.Xoshiro(123)
        ig = 1
        for (im, m) in enumerate(models)
            setup = setups[ig]
            psolver = psolver_spectral(setup)
            loss = create_loss_post(; setup, psolver, m.closure, nupdate = 1)
            snaps = [(; u = d.data[ig].u, d.t) for d in data]
            dataloader = create_dataloader_post(snaps; nunroll = 5, rng)
            θ = copy(m.θ₀)
            opt = Adam(1.0e-3)
            optstate = Optimisers.setup(opt, θ)
            it = 1:5
            snaps = (; u = data[1].data[ig].u[it], t = data[1].t[it])
            (; callbackstate, callback) = create_callback(
                create_relerr_post(;
                    data = snaps,
                    setup,
                    psolver,
                    closure_model = wrappedclosure(m.closure, setup),
                    nupdate = 2,
                );
                θ,
                displayref = false,
                displayfig = false,
                display_each_iteration = false, # Set to `true` if using CairoMakie
            )
            (; optstate, θ, callbackstate) = train(;
                dataloader,
                loss,
                optstate,
                θ,
                niter = 3,
                ncallback = 1,
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
        ig = 1
        setup = setups[ig]
        psolver = psolver_spectral(setup)
        it = 1:5
        snaps = (; u = data[1].data[ig].u[it], t = data[1].t[it])
        nupdate = 1
        for (im, m) in enumerate(models)
            err = create_relerr_post(;
                data = snaps,
                setup,
                psolver,
                closure_model = wrappedclosure(m.closure, setup),
                nupdate,
            )
            e[im] = err(m.θ₀)
        end
        e
    end

    # A-priori symmetry errors
    e_symm_prior = let
        e = zeros(length(models))
        ig = 1
        for (im, m) in enumerate(models)
            setup = setups[ig]
            setup = (; setup..., closure_model = wrappedclosure(m.closure, setup))
            err = create_relerr_symmetry_prior(; u = data[1].data[ig].u, setup)
            e[im] = err(m.θ₀)
        end
        e
    end

    # A-posteriori symmetry errors
    e_symm_post = let
        e = zeros(length(models))
        ig = 1
        for (im, m) in enumerate(models)
            setup = setups[ig]
            setup = (; setup..., closure_model = wrappedclosure(m.closure, setup))
            # setup = (; setup..., closure_model = wrappedclosure(m.closure, setup))
            err = create_relerr_symmetry_post(;
                u = data[1].data[ig].u[1],
                setup,
                psolver = psolver_spectral(setup),
                Δt = (data[1].t[2] - data[1].t[1]),
                nstep = 5,
                g = 1,
            )
            e[im] = err(m.θ₀)
        end
        e
    end
end
