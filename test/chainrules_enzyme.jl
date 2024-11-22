@testsnippet EnzymeSnip begin
    using Enzyme
    using EnzymeCore
    using Zygote
    using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
    ENABLE_LOGGING = false
end

@testmodule Case begin
    using IncompressibleNavierStokes

    D2, D3 = map((2, 3)) do D
        T = Float64
        Re = T(1_000)
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
        boundary_conditions = ntuple(d -> (DirichletBC(), DirichletBC()), D)
        temperature = temperature_equation(;
            Pr = T(0.71),
            Ra = T(1e6),
            Ge = T(1.0),
            boundary_conditions,
        )
        if D == 2
            bodyforce = (dim, x, y, t) -> (dim == 1) * 5 * sinpi(8 * y)
            dbodyforce = (dim, x, y, t) -> (dim == 1) * 5 * pi * 8 * cos(pi * 8 * y)
        elseif D == 3
            bodyforce = (dim, x, y, z, t) -> (dim == 1) * 5 * sinpi(8 * y)
            dbodyforce = (dim, x, y, z, t) -> (dim == 1) * 5 * pi * 8 * cos(pi * 8 * y)
        end
        setup = Setup(;
            x,
            boundary_conditions,
            Re,
            temperature,
            bodyforce,
            dbodyforce,
            issteadybodyforce = true,
        )
        psolver = default_psolver(setup)
        u = randn(T, setup.grid.N..., D)
        p = randn(T, setup.grid.N)
        temp = randn(T, setup.grid.N)
        div = divergence(u, setup)
        (; setup, psolver, u, p, temp, div)
    end
end

@testitem "Chain rules (boundary conditions)" setup = [EnzymeSnip] begin
    using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
    T = Float64
    Re = T(1_000)
    Pr = T(0.71)
    Ra = T(1e6)
    Ge = T(1.0)
    n = 7
    lims = T(0), T(1)
    x = range(lims..., n + 1), range(lims..., n + 1)

    for bc in (PeriodicBC(), DirichletBC(), SymmetricBC(), PressureBC())
        boundary_conditions = (bc, bc), (bc, bc)
        setup = Setup(;
            x,
            Re,
            boundary_conditions,
            temperature = temperature_equation(; Pr, Ra, Ge, boundary_conditions),
        )
        u = randn(T, setup.grid.N..., 2)
        p = randn(T, setup.grid.N)
        temp = randn(T, setup.grid.N)
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
        @test y != p
        @test any(!isnan, y)
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

@testitem "Divergence" setup = [Case, EnzymeSnip] begin
    using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
    for (u, setup, d) in
        ((Case.D2.u, Case.D2.setup, Case.D2.div), (Case.D3.u, Case.D3.setup, Case.D3.div))
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

@testitem "Pressuregradient" setup = [Case, EnzymeSnip] begin
    using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
    for (p, setup) in ((Case.D2.p, Case.D2.setup), (Case.D3.p, Case.D3.setup))
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

@testitem "Poisson" setup = [Case, EnzymeSnip] begin
    using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
    for (psolver, d, setup) in (
        (Case.D2.psolver, Case.D2.div, Case.D2.setup),
        (Case.D3.psolver, Case.D3.div, Case.D3.setup),
    )
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

@testitem "Convection" setup = [Case, EnzymeSnip] begin
    using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
    for (u, setup) in ((Case.D2.u, Case.D2.setup), (Case.D3.u, Case.D3.setup))
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

@testitem "Diffusion" setup = [Case, EnzymeSnip] begin
    using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
    for (u, setup) in ((Case.D2.u, Case.D2.setup), (Case.D3.u, Case.D3.setup))
        d = INS.diffusion(u, setup)
        d0 = copy(d)
        Zygote.pullback(INS.diffusion, u, setup)[2](d)[1]
        zpull, z_time = @timed Zygote.pullback(INS.diffusion, u, setup)[2](d)[1]

        # [!] diffusion! wants to start from 0 initialized field
        EnzymeCore.make_zero!(d)
        dd = Enzyme.make_zero(d) .+ 1
        du = Enzyme.make_zero(u)
        f = INS.enzyme_wrap(INS.diffusion!)
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
        EnzymeCore.make_zero!(d)
        dd = Enzyme.make_zero(d) .+ 1
        du = Enzyme.make_zero(u)
        ec, e_time = @timed Enzyme.autodiff(
            Enzyme.Reverse,
            f,
            Duplicated(d, dd),
            Duplicated(u, du),
            Const(setup),
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

@testitem "Bodyforce" setup = [Case, EnzymeSnip] begin
    using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
    @warn "bodyforce is tested only in the static case"
    for (u, setup) in ((Case.D2.u, Case.D2.setup), (Case.D3.u, Case.D3.setup))
        t = 0.5
        bf = INS.applybodyforce(u, t, setup)
        bf0 = copy(bf)
        setup0 = deepcopy(setup)
        Zygote.pullback(INS.applybodyforce, u, t, setup)[2](bf0)
        zpull, z_time =
            @timed Zygote.pullback(INS.applybodyforce, u, t, setup)[2](bf0)[3].bodyforce

        # We can also test Zygote autodiff
        @test zpull == setup.bodyforce

        bf = bf .* 0
        dbf = Enzyme.make_zero(bf) .+ 1
        du = Enzyme.make_zero(u)
        f = INS.enzyme_wrap(INS.applybodyforce!)
        @test f isa Function
        f(bf, u, t, setup)
        @test bf == bf0
        Enzyme.autodiff(
            Enzyme.Reverse,
            f,
            Duplicated(bf, dbf),
            Duplicated(u, du),
            Const(t),
            Const(setup),
        )
        bf = bf .* 0
        dbf = Enzyme.make_zero(bf) .+ 1
        du = Enzyme.make_zero(u)
        eb, e_time = @timed Enzyme.autodiff(
            Enzyme.Reverse,
            f,
            Duplicated(bf, dbf),
            Duplicated(u, du),
            Const(t),
            Const(setup),
        )
        if ENABLE_LOGGING
            if e_time < z_time
                @info "Enzyme is faster (bodyforce): ", e_time, " vs ", z_time
            else
                @info "Zygote is faster (bodyforce): ", z_time, " vs ", e_time
            end
        end
        @test du == zpull
    end
end

@testitem "Gravity" setup = [Case, EnzymeSnip] begin
    using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
    for (t, setup) in ((Case.D2.temp, Case.D2.setup), (Case.D3.temp, Case.D3.setup))
        g = INS.gravity(t, setup)
        Zygote.pullback(INS.gravity, t, setup)[2](g)
        zpull, z_time = @timed Zygote.pullback(INS.gravity, t, setup)[2](g)[1]

        g = vectorfield(setup)
        dg = Enzyme.make_zero(g) .+ 1
        dt = Enzyme.make_zero(t)
        f = INS.enzyme_wrap(INS.gravity!)
        @test f isa Function
        f(g, t, setup)
        @test g != 0
        Enzyme.autodiff(
            Enzyme.Reverse,
            f,
            Duplicated(g, dg),
            Duplicated(t, dt),
            Const(setup),
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

@testitem "Dissipation" setup = [Case, EnzymeSnip] begin
    using IncompressibleNavierStokes: IncompressibleNavierStokes as INS
    for (u, setup) in ((Case.D2.u, Case.D2.setup), (Case.D3.u, Case.D3.setup))
        diss = INS.dissipation(u, setup)
        diss0 = copy(diss)
        Zygote.pullback(INS.dissipation, u, setup)[2](diss)
        zpull, z_time = @timed Zygote.pullback(INS.dissipation, u, setup)[2](diss)[1]

        diss = scalarfield(setup)
        diff = vectorfield(setup)
        ddiss = Enzyme.make_zero(diss) .+ 1
        ddiff = Enzyme.make_zero(diff)
        du = Enzyme.make_zero(u)
        f = INS.enzyme_wrap(INS.dissipation!)
        @test f isa Function
        f(diss, diff, u, setup)
        @test diss == diss0
        Enzyme.autodiff(
            Enzyme.Reverse,
            f,
            Duplicated(diss, ddiss),
            Duplicated(diff, ddiff),
            Duplicated(u, du),
            Const(setup),
        )
        diss = scalarfield(setup)
        diff = vectorfield(setup)
        diss = Enzyme.make_zero(diss) .+ 1
        diff = Enzyme.make_zero(diff)
        du = Enzyme.make_zero(u)
        ed, e_time = @timed Enzyme.autodiff(
            Enzyme.Reverse,
            f,
            Duplicated(diss, ddiss),
            Duplicated(diff, ddiff),
            Duplicated(u, du),
            Const(setup),
        )
        if ENABLE_LOGGING
            if e_time < z_time
                @info "Enzyme is faster (dissipation): ", e_time, " vs ", z_time
            else
                @info "Zygote is faster (dissipation): ", z_time, " vs ", e_time
            end
        end
        @test du == zpull
    end
end
@testitem "Convection_diffusion_temp" setup = [Case, EnzymeSnip] begin
    @test_broken 1 == 2
end

@testitem "Convectiondiffusion" setup = [Case, EnzymeSnip] begin
    # the pullback rule is missing for this one
    @test_broken 1 == 2
end
