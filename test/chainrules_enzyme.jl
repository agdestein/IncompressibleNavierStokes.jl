@testsnippet EnzymeSnip begin
    using Enzyme
    using EnzymeCore
    using Zygote
    using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
    ENABLE_LOGGING = false
end

@testmodule EnzymeCase begin
    # module EnzymeCase
    using IncompressibleNavierStokes

    D2, D3 = map((2, 3)) do D
        T = Float64
        params = (
            viscosity = T(1e-3),
            conductivity = T(1e-3),
            gdir = 2,
            gravity = T(1.0),
            dodissipation = true,
        )
        n = if D == 2
            8
        elseif D == 3
            # 4^3 = 64 grid points
            # 3*64 = 192 velocity components
            # 192^2 = 36864 fininite difference pairs in convection/diffusion
            # TODO: Check if `test_rrule` computes all combinations or only a subset
            4
        end
        lims = T(0), T(1)
        x = if D == 2
            tanh_grid(lims..., n), tanh_grid(lims..., n, 1.3)
        elseif D == 3
            tanh_grid(lims..., n, 1.2), tanh_grid(lims..., n, 1.1), cosine_grid(lims..., n)
        end
        bc = ntuple(d -> (DirichletBC(), DirichletBC()), D)
        setup = Setup(; x, boundary_conditions = (; u = bc, temp = bc))
        psolver = default_psolver(setup)
        u = randn(T, setup.N..., D)
        p = randn(T, setup.N)
        temp = randn(T, setup.N)
        (; setup, psolver, u, p, temp, params)
    end
end

@testitem "Chain rules (boundary conditions)" setup = [EnzymeSnip] begin
    using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
    T = Float64
    params = (
        viscosity = T(1e-3),
        conductivity = T(1e-3),
        gdir = 2,
        gravity = T(1.0),
        dodissipation = true,
    )
    n = 7
    lims = T(0), T(1)
    x = range(lims..., n + 1), range(lims..., n + 1)
    for bc in (PeriodicBC(), DirichletBC(), SymmetricBC(), PressureBC())
        BC = (bc, bc), (bc, bc)
        setup = Setup(; x, boundary_conditions = (; u = BC, temp = BC))
        u = randn(T, setup.N..., 2)
        p = randn(T, setup.N)
        temp = randn(T, setup.N)
        u0 = copy(u)
        p0 = copy(p)
        temp0 = copy(temp)

        # --- bc_u
        Zygote.pullback(apply_bc_u, u, nothing, setup)[2](u0)[1]
        zpull, z_time = @timed Zygote.pullback(apply_bc_u, u, nothing, setup)[2](u0)[1]
        du = Enzyme.make_zero(u)
        y = Enzyme.make_zero(u)
        dy = Enzyme.make_zero(u) .+ 1
        f = INS.enzyme_wrap(INS.apply_bc_u!)
        @test f isa Function
        f(y, u, nothing, setup)
        @test y != u
        @test any(!isnan, y)
        Enzyme.autodiff(
            Enzyme.Reverse,
            f,
            Duplicated(y, dy),
            Duplicated(u, du),
            Const(nothing),
            Const(setup),
        )
        du = Enzyme.make_zero(u)
        y = Enzyme.make_zero(u)
        dy = Enzyme.make_zero(u) .+ 1
        eg, e_time = @timed Enzyme.autodiff(
            Enzyme.Reverse,
            f,
            Duplicated(y, dy),
            Duplicated(u0, du),
            Const(nothing),
            Const(setup),
        )
        if ENABLE_LOGGING
            if e_time < z_time
                @info "Enzyme is faster (bc_u): ", e_time, " vs ", z_time
            else
                @info "Zygote is faster (bc_u): ", z_time, " vs ", e_time
            end
        end
        @test du == zpull

        # --- bc_p
        Zygote.pullback(apply_bc_p, p, nothing, setup)[2](p0)[1]
        zpull, z_time = @timed Zygote.pullback(apply_bc_p, p, nothing, setup)[2](p0)[1]
        dp = Enzyme.make_zero(p)
        y = Enzyme.make_zero(p)
        dy = Enzyme.make_zero(p) .+ 1
        f = INS.enzyme_wrap(INS.apply_bc_p!)
        @test f isa Function
        f(y, p, nothing, setup)
        # @test y != p
        # TODO: check this test above. With DirichletBC, we now get y == p, but that is intentional
        @test all(!isnan, y)
        Enzyme.autodiff(
            Enzyme.Reverse,
            f,
            Duplicated(y, dy),
            Duplicated(p0, dp),
            Const(nothing),
            Const(setup),
        )
        dp = Enzyme.make_zero(p)
        y = Enzyme.make_zero(p)
        dy = Enzyme.make_zero(p) .+ 1
        eg, e_time = @timed Enzyme.autodiff(
            Enzyme.Reverse,
            f,
            Duplicated(y, dy),
            Duplicated(p0, dp),
            Const(nothing),
            Const(setup),
        )
        if ENABLE_LOGGING
            if e_time < z_time
                @info "Enzyme is faster (bc_p): ", e_time, " vs ", z_time
            else
                @info "Zygote is faster (bc_p): ", z_time, " vs ", e_time
            end
        end
        @test dp == zpull

        # --- bc_temp
        Zygote.pullback(apply_bc_temp, temp, nothing, setup)[2](temp0)[1]
        zpull, z_time =
            @timed Zygote.pullback(apply_bc_temp, temp, nothing, setup)[2](temp0)[1]
        dtemp = Enzyme.make_zero(temp)
        y = Enzyme.make_zero(temp)
        dy = Enzyme.make_zero(temp) .+ 1
        f = INS.enzyme_wrap(INS.apply_bc_temp!)
        @test f isa Function
        f(y, temp, nothing, setup)
        @test y != temp
        @test any(!isnan, y)
        Enzyme.autodiff(
            Enzyme.Reverse,
            f,
            Duplicated(y, dy),
            Duplicated(temp0, dtemp),
            Const(nothing),
            Const(setup),
        )
        dtemp = Enzyme.make_zero(temp)
        y = Enzyme.make_zero(temp)
        dy = Enzyme.make_zero(temp) .+ 1
        f = INS.enzyme_wrap(INS.apply_bc_temp!)
        eg, e_time = @timed Enzyme.autodiff(
            Enzyme.Reverse,
            f,
            Duplicated(y, dy),
            Duplicated(temp0, dtemp),
            Const(nothing),
            Const(setup),
        )
        if ENABLE_LOGGING
            if e_time < z_time
                @info "Enzyme is faster (bc_temp): ", e_time, " vs ", z_time
            else
                @info "Zygote is faster (bc_temp): ", z_time, " vs ", e_time
            end
        end
        @test dtemp == zpull
    end
end

@testitem "Divergence" setup = [EnzymeCase, EnzymeSnip] begin
    using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
    # case = EnzymeCase.D2
    for case in (EnzymeCase.D2, EnzymeCase.D3)
        (; u, setup) = case
        d = divergence(u, setup)
        d0 = copy(d)
        u0 = copy(u)
        Zygote.pullback(INS.divergence, u, setup)[2](d0)[1]
        zpull, z_time = @timed Zygote.pullback(INS.divergence, u, setup)[2](d0)[1]
        dd = Enzyme.make_zero(d) .+ 1
        du = Enzyme.make_zero(u)
        f = INS.enzyme_wrap(INS.divergence!)
        @test f isa Function
        f(d, u, setup)
        @test d == d0
        Enzyme.autodiff(
            Enzyme.Reverse,
            f,
            Duplicated(d, dd),
            Duplicated(u, du),
            Const(setup),
        )
        dd = Enzyme.make_zero(d) .+ 1
        du = Enzyme.make_zero(u)
        eg, e_time = @timed Enzyme.autodiff(
            Enzyme.Reverse,
            f,
            Duplicated(d0, dd),
            Duplicated(u0, du),
            Const(setup),
        )
        if ENABLE_LOGGING
            if e_time < z_time
                @info "Enzyme is faster (divergence): ", e_time, " vs ", z_time
            else
                @info "Zygote is faster (divergence): ", z_time, " vs ", e_time
            end
        end
        @test du == zpull
    end
end

@testitem "Pressuregradient" setup = [EnzymeCase, EnzymeSnip] begin
    using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
    for case in (EnzymeCase.D2, EnzymeCase.D3)
        (; p, setup) = case
        p0 = copy(p)
        pg = INS.pressuregradient(p, setup)
        pg0 = copy(pg)
        Zygote.pullback(INS.pressuregradient, p, setup)[2](pg0)[1]
        zpull, z_time = @timed Zygote.pullback(INS.pressuregradient, p, setup)[2](pg0)[1]
        dpg = Enzyme.make_zero(pg) .+ 1
        dp = Enzyme.make_zero(p)
        f = INS.enzyme_wrap(INS.pressuregradient!)
        @test f isa Function
        f(pg, p, setup)
        @test pg == pg0
        Enzyme.autodiff(
            Enzyme.Reverse,
            f,
            Duplicated(pg, dpg),
            Duplicated(p, dp),
            Const(setup),
        )
        dpg = Enzyme.make_zero(pg) .+ 1
        dp = Enzyme.make_zero(p)
        eg, e_time = @timed Enzyme.autodiff(
            Enzyme.Reverse,
            f,
            Duplicated(pg0, dpg),
            Duplicated(p0, dp),
            Const(setup),
        )
        if ENABLE_LOGGING
            if e_time < z_time
                @info "Enzyme is faster (pressuregradient): ", e_time, " vs ", z_time
            else
                @info "Zygote is faster (pressuregradient): ", z_time, " vs ", e_time
            end
        end
        @test dp == zpull
    end
end

@testitem "Poisson" setup = [EnzymeCase, EnzymeSnip] begin
    using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
    for case in (EnzymeCase.D2, EnzymeCase.D3)
        (; psolver, setup, u) = case
        d = divergence(u, setup)
        p0 = INS.poisson(psolver, d)
        Zygote.pullback(INS.poisson, psolver, d)[2](p0)[1]
        zpull, z_time = @timed Zygote.pullback(INS.poisson, psolver, d)[2](p0)[2]

        dd = Enzyme.make_zero(d)
        p = Enzyme.make_zero(p0)
        dp = Enzyme.make_zero(p) .+ 1
        f = INS.enzyme_wrap(INS.poisson!)
        @test f isa Function
        f(p, psolver, d)
        @test p == p0
        dp = Enzyme.make_zero(p) .+ 1

        Enzyme.autodiff(
            Enzyme.Reverse,
            f,
            Duplicated(p, dp),
            Const(psolver),
            Duplicated(d, dd),
        )
        ep, e_time = @timed Enzyme.autodiff(
            Enzyme.Reverse,
            f,
            Duplicated(p0, dp),
            Const(psolver),
            Duplicated(d, dd),
        )
        if ENABLE_LOGGING
            if e_time < z_time
                @info "Enzyme is faster (poisson): ", e_time, " vs ", z_time
            else
                @info "Zygote is faster (poisson): ", z_time, " vs ", e_time
            end
        end
        @test dd == zpull
    end
end

@testitem "Convection" setup = [EnzymeCase, EnzymeSnip] begin
    using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
    for case in (EnzymeCase.D2, EnzymeCase.D3)
        (; u, setup) = case
        c = INS.convection(u, setup)
        c0 = copy(c)
        Zygote.pullback(INS.convection, u, setup)[2](u)[1]
        zpull, z_time = @timed Zygote.pullback(INS.convection, u, setup)[2](c)[1]

        # [!] convection! wants to start from 0 initialized field
        EnzymeCore.make_zero!(c)
        dc = Enzyme.make_zero(c) .+ 1
        du = Enzyme.make_zero(u)
        f = INS.enzyme_wrap(INS.convection!)
        @test f isa Function
        f(c, u, setup)
        @test c == c0
        Enzyme.autodiff(
            Enzyme.Reverse,
            f,
            Duplicated(c, dc),
            Duplicated(u, du),
            Const(setup),
        )
        EnzymeCore.make_zero!(c)
        dc = Enzyme.make_zero(c) .+ 1
        du = Enzyme.make_zero(u)
        ec, e_time = @timed Enzyme.autodiff(
            Enzyme.Reverse,
            f,
            Duplicated(c, dc),
            Duplicated(u, du),
            Const(setup),
        )
        if ENABLE_LOGGING
            if e_time < z_time
                @info "Enzyme is faster (convection): ", e_time, " vs ", z_time
            else
                @info "Zygote is faster (convection): ", z_time, " vs ", e_time
            end
        end
        @test du == zpull
    end
end

@testitem "Diffusion" setup = [EnzymeCase, EnzymeSnip] begin
    using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
    # case = EnzymeCase.D2
    for case in (EnzymeCase.D2, EnzymeCase.D3)
        (; u, setup, params) = case
        (; viscosity) = params
        d = INS.diffusion(u, setup, viscosity)
        d0 = copy(d)
        Zygote.pullback(INS.diffusion, u, setup, viscosity)[2](d)[1]
        zpull, z_time = @timed Zygote.pullback(INS.diffusion, u, setup, viscosity)[2](d)[1]

        # [!] diffusion! wants to start from 0 initialized field
        EnzymeCore.make_zero!(d)
        dd = Enzyme.make_zero(d) .+ 1
        du = Enzyme.make_zero(u)
        f = INS.enzyme_wrap(INS.diffusion!)
        @test f isa Function
        f(d, u, setup, viscosity)
        @test d == d0
        Enzyme.autodiff(
            Enzyme.Reverse,
            f,
            Duplicated(d, dd),
            Duplicated(u, du),
            Const(setup),
            Const(viscosity),
        )
        EnzymeCore.make_zero!(d)
        dd = Enzyme.make_zero(d) .+ 1
        du = Enzyme.make_zero(u)
        ec, e_time = @timed Enzyme.autodiff(
            Enzyme.Reverse,
            f,
            Duplicated(d, dd),
            Duplicated(u, du),
            Const(setup),
            Const(viscosity),
        )
        if ENABLE_LOGGING
            if e_time < z_time
                @info "Enzyme is faster (diffusion): ", e_time, " vs ", z_time
            else
                @info "Zygote is faster (diffusion): ", z_time, " vs ", e_time
            end
        end
        @test du == zpull
    end
end

@testitem "Gravity" setup = [EnzymeCase, EnzymeSnip] begin
    using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
    for case in (EnzymeCase.D2, EnzymeCase.D3)
        (; setup, params) = case
        (; gdir, gravity) = params
        t = case.temp
        g = INS.applygravity(t, setup, gdir, gravity)
        Zygote.pullback(INS.applygravity, t, setup, gdir, gravity)[2](g)
        zpull, z_time =
            @timed Zygote.pullback(INS.applygravity, t, setup, gdir, gravity)[2](g)[1]

        g = vectorfield(setup)
        dg = Enzyme.make_zero(g) .+ 1
        dt = Enzyme.make_zero(t)
        f = INS.enzyme_wrap(INS.applygravity!)
        @test f isa Function
        f(g, t, setup, gdir, gravity)
        @test g != 0
        Enzyme.autodiff(
            Enzyme.Reverse,
            f,
            Duplicated(g, dg),
            Duplicated(t, dt),
            Const(setup),
            Const(gdir),
            Const(gravity),
        )
        g = vectorfield(setup)
        dg = Enzyme.make_zero(g) .+ 1
        dt = Enzyme.make_zero(t)
        gb, e_time = @timed Enzyme.autodiff(
            Enzyme.Reverse,
            f,
            Duplicated(g, dg),
            Duplicated(t, dt),
            Const(setup),
            Const(gdir),
            Const(gravity),
        )
        if ENABLE_LOGGING
            if e_time < z_time
                @info "Enzyme is faster (gravity): ", e_time, " vs ", z_time
            else
                @info "Zygote is faster (gravity): ", z_time, " vs ", e_time
            end
        end
        @test dt == zpull
    end
end

@testitem "Dissipation" setup = [EnzymeCase, EnzymeSnip] begin
    @test_broken 1 == 2 # Just to mark undefined adjoint
    # using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
    # for case in (EnzymeCase.D2, EnzymeCase.D3)
    #     (; u, setup, params) = case
    #     (; viscosity) = params
    #     diss = INS.dissipation(u, setup, viscosity)
    #     diss0 = copy(diss)
    #     Zygote.pullback(INS.dissipation, u, setup, viscosity)[2](diss)
    #     zpull, z_time = @timed Zygote.pullback(INS.dissipation, u, setup, viscosity)[2](diss)[1]
    #     diss = scalarfield(setup)
    #     ddiss = Enzyme.make_zero(diss) .+ 1
    #     du = Enzyme.make_zero(u)
    #     f = INS.enzyme_wrap(INS.dissipation!)
    #     @test f isa Function
    #     f(diss, u, setup, dissipation)
    #     @test diss == diss0
    #     Enzyme.autodiff(
    #         Enzyme.Reverse,
    #         f,
    #         Duplicated(diss, ddiss),
    #         Duplicated(u, du),
    #         Const(setup),
    #         Const(viscosity),
    #     )
    #     diss = scalarfield(setup)
    #     diss = Enzyme.make_zero(diss) .+ 1
    #     du = Enzyme.make_zero(u)
    #     ed, e_time = @timed Enzyme.autodiff(
    #         Enzyme.Reverse,
    #         f,
    #         Duplicated(diss, ddiss),
    #         Duplicated(u, du),
    #         Const(setup),
    #         Const(viscosity),
    #     )
    #     if ENABLE_LOGGING
    #         if e_time < z_time
    #             @info "Enzyme is faster (dissipation): ", e_time, " vs ", z_time
    #         else
    #             @info "Zygote is faster (dissipation): ", z_time, " vs ", e_time
    #         end
    #     end
    #     @test du == zpull
    # end
end

@testitem "Convection_diffusion_temp" setup = [EnzymeCase, EnzymeSnip] begin
    @test_broken 1 == 2
end

@testitem "Convectiondiffusion" setup = [EnzymeCase, EnzymeSnip] begin
    # the pullback rule is missing for this one
    @test_broken 1 == 2
end
