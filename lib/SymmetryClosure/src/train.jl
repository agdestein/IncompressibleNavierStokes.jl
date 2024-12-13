function trainpost(;
    params,
    outdir,
    plotdir,
    taskid,
    postseed,
    dns_seeds_train,
    dns_seeds_valid,
    nsubstep,
    nunroll,
    ntrajectory,
    closure_models,
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
    device = x -> adapt(params.backend, x)
    Φ = params.filters[1]
    for (igrid, nles) in enumerate(params.nles)
        igrid == 1 || continue
        if isnothing(taskid) || igrid == taskid
            @info "Training a-posteriori" nles
        else
            # Each task does one training
            @info "Skipping a-posteriori training for iteration $(igrid) != $(taskid)"
            continue
        end
        starttime = time()
        closure_model = closure_models[igrid]
        file = joinpath(outdir, "training", splatfileparts(; nles) * ".jld2")
        ispath(dirname(file)) || mkpath(dirname(file))
        figdir = mkpath(joinpath(outdir, "training"))
        figfile = joinpath(figdir, splitext(basename(file))[1] * ".pdf")
        checkfile = join(splitext(file), "_checkpoint")
        setup = getsetup(; params, nles)
        psolver = default_psolver(setup)
        loss = create_loss_post(;
            setup,
            psolver,
            params.method,
            closure_model,
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
        θ = θ_start |> device
        optstate = Optimisers.setup(opt, θ)
        (; callbackstate, callback) = let
            d = data_valid[1]
            it = 1:nunroll_valid
            data =
                (; u = selectdim(d.u, ndims(d.u), it) |> collect |> device, t = d.t[it])
            create_callback(
                create_relerr_post(;
                    data,
                    setup,
                    psolver,
                    params.method,
                    closure_model,
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
        save_object(file, results)
    end
    @info "Finished a-posteriori training."
end
