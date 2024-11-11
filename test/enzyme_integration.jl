
@testsnippet EnzymeStuff begin
    using IncompressibleNavierStokes
    using Enzyme
    using Zygote
    using Random
    rng = Random.default_rng();
end

@testmodule Info begin
    using IncompressibleNavierStokes
    T = Float64
    ArrayType = Array
    Re = T(1_000)
    D = 2
    n = 64
    N = n + 2
    lims = T(0), T(1);
    x = tanh_grid(lims..., n), tanh_grid(lims..., n, 1.3) ;
    boundary_conditions = ntuple(d -> (DirichletBC(), DirichletBC()), D)
    setup = Setup(;x, boundary_conditions, Re);
    psolver = default_psolver(setup)
end

@testitem "Enzyme one force pullback" setup = [Info, EnzymeStuff] begin
    for (setup, psolver, T, N) in ((Info.setup, Info.psolver, Info.T, Info.N), )
        dudt = zeros(T, (N, N, 2))  ;
        u = rand(T, (N, N, 2));
        u0 = copy(u);
        params = [setup, psolver];
        params_ref = Ref(params);
        right_hand_side!(dudt, u, params_ref, T(0))
        F_out = create_right_hand_side(setup, psolver)
        @test dudt ≈ F_out(u, nothing, T(0))
        @test u == u0
        @test sum(dudt) != 0

        niter = 5000
        list_u = [rand(T, (N, N, 2)) for i in 1:niter];
        list_z = []
        _, tz, mz = @timed begin
            for i in 1:niter
                dudt = F_out(list_u[i], nothing, T(0))
                push!(list_z, dudt)
            end
        end
        list_e = [zeros(T, (N, N, 2)) for i in 1:niter];
        _, te, me = @timed begin
            for i in 1:niter
                right_hand_side!(list_e[i], list_u[i], params_ref, T(0))
            end
        end
        @test all([list_z[i] ≈ list_e[i] for i in 1:niter])
        if te < tz
            @info "One F in-place is faster by a factor of $(tz/te)"
        else
            @info "One F out-of-place is faster by a factor of $(te/tz)"
        end
        if me < mz
            @info "One F in-place is more memory efficient by a factor of $(mz/me)"
        else
            @info "One F out-of-place is more memory efficient by a factor of $(me/mz)"
        end

    end
end

@testitem "Enzyme RHS pullback" setup = [Info, EnzymeStuff] begin
    for (setup, psolver, T, N) in ((Info.setup, Info.psolver, Info.T, Info.N), )
        F_out = create_right_hand_side(setup, psolver)
        dudt = zeros(T, (N, N, 2))  ;
        u = rand(T, (N, N, 2));
        u0 = copy(u)
        du = Enzyme.make_zero(u);
        dd = Enzyme.make_zero(dudt) .+ 1;
        params = [setup, psolver];
        params_ref = Ref(params);
        Enzyme.autodiff(Enzyme.Reverse, right_hand_side!, Duplicated(dudt,dd), Duplicated(u,du), Const(params_ref), Const(T(0)))
        @test u0 == u
        @test dudt ≈ F_out(u, nothing, T(0))
        zpull = Zygote.pullback(F_out, u, nothing, T(0));
        @test zpull[1] ≈ dudt
        @test zpull[2](dudt)[1] ==du


        # Now I run each option multiple times from different random initial conditions
        niter = 3000
        list_u = [rand(T, (N, N, 2)) for i in 1:niter];
        list_z = []
        _, tz, mz = @timed begin
            for i in 1:niter
                du = Enzyme.make_zero(u);
                dd = Enzyme.make_zero(dudt) .+ 1;
                zpull = Zygote.pullback(F_out, list_u[i], nothing, T(0));
                push!(list_z, zpull[2](zpull[1])[1])
            end
        end
        list_e = []
        _, te, me = @timed begin
            for i in 1:niter
                du = Enzyme.make_zero(u);
                dd = Enzyme.make_zero(dudt) .+ 1;
                Enzyme.autodiff(Enzyme.Reverse, right_hand_side!, Duplicated(dudt,dd), Duplicated(list_u[i],du), Const(params_ref), Const(T(0)))
                push!(list_e, du)
            end
        end
        @test all([list_z[i] ≈ list_e[i] for i in 1:niter])
        if te < tz
            @info "Reverse AD using Enzyme is faster by a factor of $(tz/te)"
        else
            @info "Reverse AD using Zygote is faster by a factor of $(te/tz)"
        end
        if me < mz
            @info "Reverse AD using Enzyme is more memory efficient by a factor of $(mz/me)"
        else
            @info "Reverse AD using Zygote is more memory efficient by a factor of $(me/mz)"
        end
    end
end