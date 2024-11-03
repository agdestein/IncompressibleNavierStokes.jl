getdatafile(outdir, nles, filter, seed) =
    joinpath(outdir, "data", splatfileparts(; seed = repr(seed), filter, nles) * ".jld2")

"Create data files."
createdata(; params, seeds, outdir, taskid) =
    for (iseed, seed) in enumerate(seeds)
        if isnothing(taskid) || iseed == taskid
            @info "Creating DNS trajectory for seed $(repr(seed))"
        else
            # Each task does one initial condition
            @info "Skipping seed $(repr(seed)) for task $taskid"
            continue
        end
        (; data, t, comptime) = create_les_data(; params..., rng = Xoshiro(seed))
        @info("Trajectory info:", comptime / 60, length(t), Base.summarysize(data) * 1e-9,)
        for (ifilter, Φ) in enumerate(params.filters),
            (igrid, nles) in enumerate(params.nles)

            (; u, c) = data[igrid, ifilter]
            f = getdatafile(outdir, nles, Φ, seed)
            datadir = dirname(f)
            ispath(datadir) || mkpath(datadir)
            @info "Saving data to $f"
            jldsave(f; u, c, t, comptime)
        end
    end

getpriorfile(outdir, nles, filter) =
    joinpath(outdir, "priortraining", splatfileparts(; filter, nles) * ".jld2")

"Load a-priori training results from correct file names."
loadprior(outdir, nles, filters) = map(
    splat((nles, Φ) -> load_object(getpriorfile(outdir, nles, Φ))),
    Iterators.product(nles, filters),
)

"Train with a-priori loss."
function trainprior(;
    params,
    priorseed,
    dns_seeds_train,
    dns_seeds_valid,
    taskid,
    outdir,
    plotdir,
    closure,
    θ_start,
    opt,
    λ = nothing,
    noiselevel = nothing,
    scheduler = nothing,
    nvalid,
    batchsize,
    displayref,
    displayupdates,
    nupdate_callback,
    loadcheckpoint,
    nepoch,
    niter,
)
    device(x) = adapt(params.backend, x)
    itotal = 0
    for Φ in params.filters, nles in params.nles
        itotal += 1
        if isnothing(taskid) || itotal == taskid
            @info "Training a-priori" Φ nles
        else
            # Each task does one training
            @info "Skipping a-priori training for iteration $(itotal) != $(taskid)"
            continue
        end
        starttime = time()
        priorfile = getpriorfile(outdir, nles, Φ)
        priordir = dirname(priorfile)
        ispath(priordir) || mkpath(priordir)
        figdir = joinpath(plotdir, "priortraining")
        ispath(figdir) || mkpath(figdir)
        figfile = joinpath(figdir, splitext(basename(priorfile))[1] * ".pdf")
        checkfile = join(splitext(priorfile), "_checkpoint")
        batchseed, validseed = splitseed(priorseed, 2) # Same seed for all training setups
        setup = getsetup(; params, nles)
        data_train =
            map(s -> namedtupleload(getdatafile(outdir, nles, Φ, s)), dns_seeds_train)
        data_valid =
            map(s -> namedtupleload(getdatafile(outdir, nles, Φ, s)), dns_seeds_valid)
        io_train = create_io_arrays(data_train, setup)
        io_valid = create_io_arrays(data_valid, setup)
        # dataloader = create_dataloader_prior(io_train; batchsize, device)
        θ = device(θ_start)
        loss = create_loss_prior(closure)
        optstate = Optimisers.setup(opt, θ)
        it = rand(Xoshiro(validseed), 1:size(io_valid.u, params.D + 2), nvalid)
        validset = device(map(v -> collect(selectdim(v, ndims(v), it)), io_valid))
        (; callbackstate, callback) = create_callback(
            create_relerr_prior(closure, validset...);
            θ,
            displayref,
            displayupdates,
            figfile,
            nupdate = nupdate_callback,
        )
        if loadcheckpoint
            # Resume from checkpoint
            (; icheck, trainstate, callbackstate) = namedtupleload(checkfile)
            @assert eltype(callbackstate.θmin) == Float32 "gpu_device() only works with Float32"
            trainstate = trainstate |> gpu_device()
            @reset callbackstate.θmin = callbackstate.θmin |> gpu_device()
        else
            icheck = 0
            trainstate = (; optstate, θ, rng = Xoshiro(batchseed))
            callbackstate = callback(callbackstate, trainstate) # Initial callback
        end
        for iepoch = icheck+1:nepoch
            # (; trainstate, callbackstate) = train(;
            #     dataloader,
            #     loss,
            #     trainstate,
            #     scheduler,
            #     callbackstate,
            #     callback,
            #     niter,
            # )
            dataloader = DataLoader(
                (io_train.u, io_train.c);
                batchsize,
                trainstate.rng,
                buffer = true,
                shuffle = true,
                parallel = false,
                partial = false,
            )
            (; trainstate, callbackstate) = trainepoch(;
                dataloader,
                loss,
                trainstate,
                callbackstate,
                callback,
                device,
                noiselevel,
                λ,
            )
            if !isnothing(scheduler)
                eta = scheduler(iepoch)
                Optimisers.adjust!(trainstate.optstate, eta)
            end
            # Save all states to resume training later
            # First move all arrays to CPU
            c = callbackstate |> cpu_device()
            t = trainstate |> cpu_device()
            jldsave(checkfile; icheck = iepoch, callbackstate = c, trainstate = t)
        end
        θ = callbackstate.θmin # Use best θ instead of last θ
        results = (; θ = Array(θ), comptime = time() - starttime, callbackstate.hist)
        save_object(priorfile, results)
    end
    @info "Finished a-priori training."
end

getpostfile(outdir, nles, filter, projectorder) =
    joinpath(outdir, "posttraining", splatfileparts(; projectorder, filter, nles) * ".jld2")

"Load a-posteriori training results from correct file names."
loadpost(outdir, nles, filters, projectorders) = map(
    splat((nles, Φ, o) -> load_object(getpostfile(outdir, nles, Φ, o))),
    Iterators.product(nles, filters, projectorders),
)

"Train with a-posteriori loss function."
function trainpost(;
    params,
    projectorders,
    outdir,
    plotdir,
    taskid,
    postseed,
    dns_seeds_train,
    dns_seeds_valid,
    nsubstep,
    nunroll,
    ntrajectory,
    closure,
    θ_start,
    opt,
    λ = nothing,
    scheduler,
    nunroll_valid,
    nupdate_callback,
    displayref,
    displayupdates,
    loadcheckpoint,
    nepoch,
    niter,
)
    device(x) = adapt(params.backend, x)
    itotal = 0
    for projectorder in projectorders,
        (ifil, Φ) in enumerate(params.filters),
        (igrid, nles) in enumerate(params.nles)

        itotal += 1
        if isnothing(taskid) || itotal == taskid
            @info "Training a-posteriori" projectorder Φ nles
        else
            # Each task does one training
            @info "Skipping a-posteriori training for iteration $(itotal) != $(taskid)"
            continue
        end
        starttime = time()
        postfile = getpostfile(outdir, nles, Φ, projectorder)
        ispath(dirname(postfile)) || mkpath(dirname(postfile))
        figdir = joinpath(plotdir, "posttraining")
        ispath(figdir) || mkpath(figdir)
        figfile = joinpath(figdir, splitext(basename(postfile))[1] * ".pdf")
        checkfile = join(splitext(postfile), "_checkpoint")
        setup = getsetup(; params, nles)
        psolver = default_psolver(setup)
        loss = create_loss_post(;
            setup,
            psolver,
            method = RKProject(params.method, projectorder),
            closure,
            nsubstep, # Time steps per loss evaluation
        )
        data_train =
            map(s -> namedtupleload(getdatafile(outdir, nles, Φ, s)), dns_seeds_train)
        data_valid =
            map(s -> namedtupleload(getdatafile(outdir, nles, Φ, s)), dns_seeds_valid)
        dataloader = create_dataloader_post(
            map(d -> (; d.u, d.t), data_train);
            device,
            nunroll,
            ntrajectory,
        )
        θ = θ_start[igrid, ifil] |> device
        optstate = Optimisers.setup(opt, θ)
        (; callbackstate, callback) = let
            d = data_valid[1]
            it = 1:nunroll_valid
            data = (; u = device.(d.u[it]), t = d.t[it])
            create_callback(
                create_relerr_post(;
                    data,
                    setup,
                    psolver,
                    method = RKProject(params.method, projectorder),
                    closure_model = wrappedclosure(closure, setup),
                    nsubstep,
                );
                θ,
                figfile,
                displayref,
                displayupdates,
                nupdate = nupdate_callback,
            )
        end
        if loadcheckpoint
            @info "Resuming from checkpoint $checkfile"
            (; icheck, trainstate, callbackstate) = namedtupleload(checkfile)
            @assert eltype(callbackstate.θmin) == Float32 "gpu_device() only works with Float32"
            trainstate = trainstate |> gpu_device()
            @reset callbackstate.θmin = callbackstate.θmin |> gpu_device()
        else
            icheck = 0
            trainstate = (; optstate, θ, rng = Xoshiro(postseed))
            callbackstate = callback(callbackstate, trainstate) # Initial callback
        end
        for iepoch = icheck+1:nepoch
            (; trainstate, callbackstate) =
                train(; dataloader, loss, trainstate, niter, callbackstate, callback, λ)
            if !isnothing(scheduler)
                eta = scheduler(iepoch)
                Optimisers.adjust!(trainstate.optstate, eta)
            end
            @info "Saving checkpoint to $(basename(checkfile))"
            c = callbackstate |> cpu_device()
            t = trainstate |> cpu_device()
            jldsave(checkfile; icheck = iepoch, callbackstate = c, trainstate = t)
        end
        θ = callbackstate.θmin # Use best θ instead of last θ
        results = (; θ = Array(θ), comptime = time() - starttime)
        save_object(postfile, results)
    end
    @info "Finished a-posteriori training."
end

getsmagfile(outdir, nles, filter, projectorder) =
    joinpath(outdir, "smagorinsky", splatfileparts(; projectorder, filter, nles) * ".jld2")

loadsmagorinsky(outdir, nles, filters, projectorders) = map(
    splat((nles, Φ, o) -> load_object(getsmagfile(outdir, nles, Φ, o))),
    Iterators.product(nles, filters, projectorders),
)

function trainsmagorinsky(;
    params,
    projectorders,
    outdir,
    dns_seeds_train,
    taskid,
    nunroll,
    nsubstep,
    ninfo,
    θrange,
)
    device(x) = adapt(params.backend, x)
    itotal = 0
    for projectorder in projectorders, Φ in params.filters, nles in params.nles
        itotal += 1
        if isnothing(taskid) || itotal == taskid
            @info "Training Smagorinsky" projectorder Φ nles
        else
            # Each task does one training
            @info "Skipping Smagorinsky training for iteration $(itotal) != $(taskid)"
            continue
        end
        starttime = time()
        T = typeof(params.Re)
        smagfile = getsmagfile(outdir, nles, Φ, projectorder)
        smagdir = dirname(smagfile)
        ispath(smagdir) || mkpath(smagdir)
        setup = getsetup(; params, nles)
        psolver = default_psolver(setup)
        d = namedtupleload(getdatafile(outdir, nles, Φ, dns_seeds_train[1]))
        it = 1:nunroll
        data = (; u = device.(d.u[it]), t = d.t[it])
        θmin = T(0)
        emin = T(Inf)
        err = create_relerr_post(;
            data,
            setup,
            psolver,
            method = RKProject(params.method, projectorder),
            closure_model = IncompressibleNavierStokes.smagorinsky_closure_natural(setup),
            nupdate = nsubstep, # Number of time steps between t[i] and t[i + 1]
        )
        for (iθ, θ) in enumerate(θrange)
            iθ % ninfo == 0 && @info "Testing θ = $θ"
            e = err(θ)
            if e < emin
                emin = e
                θmin = θ
            end
        end
        @info "Optimal θ = $θmin"
        results = (; θ = θmin, comptime = time() - starttime)
        save_object(smagfile, results)
    end
    @info "Finished Smagorinsky training."
end
