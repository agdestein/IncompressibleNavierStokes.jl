function observe_v(dnsobs, Φ, les, compression, psolver)
    (; grid) = les
    (; dimension, N, Iu, Ip) = grid
    D = dimension()
    Mα = N[1] - 2
    v = zero.(Φ(dnsobs[].u, les, compression))
    Pv = zero.(v)
    p = zero(v[1])
    div = zero(p)
    ΦPF = zero.(v)
    PFΦ = zero.(v)
    c = zero.(v)
    T = eltype(v[1])
    results = (;
        Φ,
        Mα,
        t = zeros(T, 0),
        Dv = zeros(T, 0),
        Pv = zeros(T, 0),
        Pc = zeros(T, 0),
        c = zeros(T, 0),
        E = zeros(T, 0),
    )
    on(dnsobs) do (; u, PF, t, E)
        push!(results.t, t)

        Φ(v, u, les, compression)
        apply_bc_u!(v, t, les)
        Φ(ΦPF, PF, les, compression)
        momentum!(PFΦ, v, nothing, t, les)
        apply_bc_u!(PFΦ, t, les; dudt = true)
        project!(PFΦ, les; psolver, p)
        foreach(α -> c[α] .= ΦPF[α] .- PFΦ[α], 1:D)
        apply_bc_u!(c, t, les)
        divergence!(div, v, les)
        norm_Du = norm(div[Ip])
        norm_v = sqrt(sum(α -> sum(abs2, v[α][Iu[α]]), 1:D))
        push!(results.Dv, norm_Du / norm_v)

        copyto!.(Pv, v)
        project!(Pv, les; psolver, p)
        foreach(α -> Pv[α] .= Pv[α] .- v[α], 1:D)
        norm_vmPv = sqrt(sum(α -> sum(abs2, Pv[α][Iu[α]]), 1:D))
        push!(results.Pv, norm_vmPv / norm_v)

        Pc = Pv
        copyto!.(Pc, c)
        project!(Pc, les; psolver, p)
        foreach(α -> Pc[α] .= Pc[α] .- c[α], 1:D)
        norm_cmPc = sqrt(sum(α -> sum(abs2, Pc[α][Iu[α]]), 1:D))
        norm_c = sqrt(sum(α -> sum(abs2, c[α][Iu[α]]), 1:D))
        push!(results.Pc, norm_cmPc / norm_c)

        norm_ΦPF = sqrt(sum(α -> sum(abs2, ΦPF[α][Iu[α]]), 1:D))
        push!(results.c, norm_c / norm_ΦPF)

        kinetic_energy!(p, v, les)
        scalewithvolume!(p, les)
        Ev = sum(view(p, Ip))

        push!(results.E, Ev / E)
    end
    results
end

observe_u(dns, psolver_dns, filters; nupdate = 1) =
    processor() do state
        PF = zero.(state[].u)
        p = zero(state[].u[1])
        dnsobs = Observable((; state[].u, PF, state[].t, E = zero(eltype(p))))
        results =
            map(f -> observe_v(dnsobs, f.Φ, f.setup, f.compression, f.psolver), filters)
        on(state) do (; u, t, n)
            n % nupdate == 0 || return
            apply_bc_u!(u, t, dns)
            momentum!(PF, u, nothing, t, dns)
            apply_bc_u!(PF, t, dns; dudt = true)
            project!(PF, dns; psolver = psolver_dns, p)

            kinetic_energy!(p, u, dns)
            scalewithvolume!(p, dns)
            E = sum(view(p, dns.grid.Ip))

            dnsobs[] = (; u, PF, t, E)
        end
        # state[] = state[] # Save initial conditions
        results
    end
