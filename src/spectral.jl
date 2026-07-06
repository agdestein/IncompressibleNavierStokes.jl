# Spectral quantities (energy spectra, turbulence statistics) for periodic
# uniform grids. All transforms use RFFT, which only stores the non-negative
# wavenumbers in the first direction. Sums over all wavenumbers must therefore
# count the modes with `0 < k_1 < k_max` twice ("missing" conjugate modes).

"Integer wavenumber corresponding to index `i` in an FFT array of size `n`."
@inline fftfreq_int(n, i) = i - 1 - ifelse(i ≤ (n + 1) >> 1, 0, n)

"Size of the RFFT of an array of interior size `Np`."
rfft_size(Np) = ntuple(α -> α == 1 ? Np[1] ÷ 2 + 1 : Np[α], length(Np))

"""
Squared integer wavenumber magnitudes `|k|²` for the RFFT array of interior
size `Np`, flattened to a vector (one entry per linear index).
"""
function squared_wavenumbers(setup)
    (; dimension, Np) = setup
    D = dimension()
    k2 = map(CartesianIndices(rfft_size(Np))) do I
        sum(α -> α == 1 ? (I[1] - 1)^2 : fftfreq_int(Np[α], I[α])^2, 1:D)
    end
    reshape(k2, :)
end

"""
Precompute wavenumber shell indices for computing energy spectra.
Shell `j` contains the wavenumbers `κ[j] ≤ |k| < κ[j] + 1`, where
`κ = 0:kmax` (all integer wavenumber bins).

Return `(; κ, inds, energyinds, Nhat)`, where

- `inds[j]`: linear indices of the stored RFFT modes in shell `j`,
- `energyinds[j]`: same, but with the indices of modes whose conjugate
  counterpart is not stored in the RFFT array repeated, such that
  `sum(abs2, view(uhat, energyinds[j]))` is the total squared modulus of the
  full FFT shell,
- `Nhat`: size of the RFFT array.
"""
function spectral_stuff(setup; kmax = minimum(setup.Np) ÷ 2)
    (; Np, x) = setup
    Nhat = rfft_size(Np)
    Kx = Nhat[1]

    k2 = squared_wavenumbers(setup)
    isort = sortperm(k2)
    k2sort = k2[isort]

    IntArray = typeof(similar(x[1], Int, 0))
    inds = IntArray[]
    energyinds = IntArray[]
    for κ = 0:kmax
        jstart = searchsortedfirst(k2sort, κ^2)
        jstop = searchsortedfirst(k2sort, (κ + 1)^2) - 1
        shell = isort[jstart:jstop]

        # Modes with x-index 1 (`k_1 = 0`) or `Kx` (`k_1` at Nyquist) contain
        # their own conjugate counterpart; all others are counted twice.
        # For linear index `i`, the x-index is 1 iff `i % Kx == 1`,
        # and `Kx` iff `i % Kx == 0`.
        conjshell = filter(i -> i % Kx > 1, shell)

        push!(inds, adapt(IntArray, shell))
        push!(energyinds, adapt(IntArray, vcat(shell, conjshell)))
    end

    (; κ = collect(0:kmax), inds, energyinds, Nhat)
end

"""
Accumulate the spectral kinetic energy `sum_α |û_α|² / 2` into the real
RFFT-sized array `ehat_field`, where `û_α = rfft(u_α) / prod(Np)` (such that
the missing-mode-accounted sum over all wavenumbers is the spatial mean
kinetic energy density `⟨u_i u_i⟩ / 2`).
"""
function spectral_energy_field!(ehat_field, u, setup; plan, uin, uhat)
    (; dimension, Ip, Np) = setup
    D = dimension()
    T = eltype(uin)
    fill!(ehat_field, 0)
    for α = 1:D
        copyto!(uin, view(u, Ip, α))
        mul!(uhat, plan, uin)
        @. ehat_field += abs2(uhat)
    end
    ehat_field ./= 2 * T(prod(Np))^2
    ehat_field
end

"Sum over an RFFT-sized array, counting the missing conjugate modes."
spectralsum(f) = sum(f) + sum(selectdim(f, 1, 2:(size(f, 1)-1)))

"""
Compute the energy spectrum of the velocity field `u`. The energy at the
scalar wavenumber `κ ∈ ℕ` is defined as the sum over the wavenumber shell
`κ ≤ |k| < κ + 1`:

```math
\\hat{e}(\\kappa) = \\sum_{\\kappa \\leq \\| k \\| < \\kappa + 1}
\\frac{| \\hat{u}(k) |^2}{2},
```

such that `sum(ehat)` is the mean kinetic energy density `⟨u_i u_i⟩ / 2` (if
all wavenumbers are contained in the shells, see [`spectral_stuff`](@ref)).

Return `(; κ, ehat)`. The precomputed `stuff` can be reused between calls.
"""
function energyspectrum(u, setup; stuff = spectral_stuff(setup))
    (; Np, x) = setup
    T = eltype(x[1])
    uin = similar(x[1], Np)
    uhat = similar(x[1], Complex{T}, stuff.Nhat)
    ehat_field = similar(x[1], stuff.Nhat)
    plan = plan_rfft(uin)
    spectral_energy_field!(ehat_field, u, setup; plan, uin, uhat)
    ehat = map(inds -> sum(view(ehat_field, inds)), stuff.energyinds)
    (; stuff.κ, ehat)
end

"""
Squared dimensional wavenumber magnitudes `|2π k / L|²` in an RFFT-sized
array on the device.
"""
function squared_wavenumbers_dimensional(setup)
    (; dimension, Np, xlims, x) = setup
    D = dimension()
    T = eltype(x[1])
    k2 = map(CartesianIndices(rfft_size(Np))) do I
        sum(1:D) do α
            L = xlims[α][2] - xlims[α][1]
            kint = α == 1 ? I[1] - 1 : fftfreq_int(Np[α], I[α])
            (T(2π) * kint / L)^2
        end
    end
    k2dev = similar(x[1], size(k2))
    copyto!(k2dev, k2)
end

"""
Compute turbulence statistics of the velocity field `u`. The statistics are
computed spectrally (RFFT of the interior velocity components), and require a
uniform periodic grid. The returned quantities differ between 2D and 3D, since
two-dimensional turbulence has different cascades (see the 2D and 3D methods).
"""
function turbulence_statistics(u, setup, viscosity)
    assert_uniform_periodic(setup, "Turbulence statistics")
    turbulence_statistics(setup.dimension, u, setup, viscosity)
end

"""
Three-dimensional homogeneous isotropic turbulence statistics
(Kolmogorov scaling) [Pope2000](@cite):

- `e`: mean kinetic energy density `⟨u_i u_i⟩ / 2`
- `uavg`: per-component RMS velocity `u' = sqrt(2 e / 3)`
- `diss`: dissipation rate `ϵ = ν ⟨∇u:∇u⟩ = ν Σ_k |k|² |û|²`
- `l_int`: integral scale estimate `L = u'³ / ϵ`
- `l_tay`: Taylor microscale `λ = sqrt(15 ν / ϵ) u'`
- `l_kol`: Kolmogorov length scale `η = (ν³ / ϵ)^(1/4)`
- `t_int`: large-eddy turnover time `L / u'`
- `t_tay`: Taylor time scale `λ / u'`
- `t_kol`: Kolmogorov time scale `τ_η = sqrt(ν / ϵ)`
- `Re_int`: integral-scale Reynolds number `L u' / ν`
- `Re_tay`: Taylor-scale Reynolds number `λ u' / ν`
- `kmax_eta`: resolution indicator `k_max η` (well-resolved DNS: `≳ 1.5`)
"""
function turbulence_statistics(::Dimension{3}, u, setup, viscosity)
    (; Np, xlims, x) = setup
    T = eltype(x[1])

    Nhat = rfft_size(Np)
    uin = similar(x[1], Np)
    uhat = similar(x[1], Complex{T}, Nhat)
    ehat_field = similar(x[1], Nhat)
    plan = plan_rfft(uin)
    spectral_energy_field!(ehat_field, u, setup; plan, uin, uhat)
    k2 = squared_wavenumbers_dimensional(setup)

    e = spectralsum(ehat_field)
    diss = 2 * viscosity * spectralsum(k2 .* ehat_field)
    uavg = sqrt(2 * e / 3)
    l_kol = (viscosity^3 / diss)^T(1 / 4)
    l_tay = sqrt(15 * viscosity / diss) * uavg
    l_int = uavg^3 / diss
    t_int = l_int / uavg
    t_tay = l_tay / uavg
    t_kol = sqrt(viscosity / diss)
    Re_int = l_int * uavg / viscosity
    Re_tay = l_tay * uavg / viscosity
    kmax = T(π) * minimum(α -> Np[α] / (xlims[α][2] - xlims[α][1]), 1:3)
    kmax_eta = kmax * l_kol

    (; e, uavg, diss, l_int, l_tay, l_kol, t_int, t_tay, t_kol, Re_int, Re_tay, kmax_eta)
end

"""
Two-dimensional turbulence statistics (Kraichnan-Batchelor-Leith dual
cascade). The small scales are set by the forward *enstrophy* cascade, so the
relevant dissipation is the enstrophy-dissipation rate and the relevant length
is the Kraichnan scale (not the Kolmogorov scale):

- `e`: mean kinetic energy density `⟨u_i u_i⟩ / 2`
- `uavg`: per-component RMS velocity `u' = sqrt(e)`
- `diss`: energy dissipation rate `ϵ = 2 ν Ω = ν Σ_k |k|² |û|²`
- `enstrophy`: enstrophy `Ω = ⟨ω²⟩ / 2`
- `palinstrophy`: palinstrophy `P = ⟨|∇ω|²⟩ / 2`
- `enstrophy_diss`: enstrophy-dissipation rate `η_Ω = 2 ν P = ν Σ_k |k|⁴ |û|²`
- `omega_rms`: RMS vorticity `ω' = sqrt(2 Ω)`
- `l_kra`: Kraichnan length scale `η_K = (ν³ / η_Ω)^(1/6)`
- `l_omega`: Taylor-type length scale `u' / ω'`
- `t_eddy`: large-scale strain time `1 / ω'`
- `t_ens`: enstrophy-cascade time scale `η_Ω^(-1/3)`
- `Re`: Reynolds number `u'² / (ν ω')`
- `kmax_etaK`: resolution indicator `k_max η_K` (well-resolved DNS: `≳ 1`)
"""
function turbulence_statistics(::Dimension{2}, u, setup, viscosity)
    (; Np, xlims, x) = setup
    T = eltype(x[1])

    Nhat = rfft_size(Np)
    uin = similar(x[1], Np)
    uhat = similar(x[1], Complex{T}, Nhat)
    ehat_field = similar(x[1], Nhat)
    plan = plan_rfft(uin)
    spectral_energy_field!(ehat_field, u, setup; plan, uin, uhat)
    k2 = squared_wavenumbers_dimensional(setup)

    e = spectralsum(ehat_field)
    uavg = sqrt(e)
    diss = 2 * viscosity * spectralsum(k2 .* ehat_field)
    enstrophy = diss / (2 * viscosity)
    omega_rms = sqrt(2 * enstrophy)
    enstrophy_diss = 2 * viscosity * spectralsum(k2 .^ 2 .* ehat_field)
    palinstrophy = enstrophy_diss / (2 * viscosity)
    l_kra = (viscosity^3 / enstrophy_diss)^T(1 / 6)
    l_omega = uavg / omega_rms
    t_eddy = 1 / omega_rms
    t_ens = enstrophy_diss^(-T(1 / 3))
    Re = uavg * l_omega / viscosity
    kmax = T(π) * minimum(α -> Np[α] / (xlims[α][2] - xlims[α][1]), 1:2)
    kmax_etaK = kmax * l_kra

    (;
        e,
        uavg,
        diss,
        enstrophy,
        palinstrophy,
        enstrophy_diss,
        omega_rms,
        l_kra,
        l_omega,
        t_eddy,
        t_ens,
        Re,
        kmax_etaK,
    )
end
