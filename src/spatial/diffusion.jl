"""
    d2u, d2v, Jacu, Jacv = diffusion(V, t, setup, getJacobian)

Evaluate diffusive terms and optionally Jacobian
"""
function diffusion(V, t, setup, getJacobian)
    @unpack visc = setup.case
    @unpack Nu, Nv, indu, indv = setup.grid
    @unpack N1, N2, N3, N4 = setup.grid
    @unpack Dux, Duy, Dvx, Dvy = setup.discretization
    @unpack Diffu, Diffv, yDiffu, yDiffv = setup.discretization
    @unpack Su_ux, Su_uy, Su_vx, Sv_vx, Sv_vy, Sv_uy = setup.discretization
    @unpack Aν_ux, Aν_uy, Aν_vx, Aν_vy = setup.discretization

    uₕ = @view V[indu]
    vₕ = @view V[indv]

    Jacu = spzeros(Nu, Nu + Nv)
    Jacv = spzeros(Nv, Nu + Nv)

    if visc == "laminar"

        d2u = Diffu * uₕ + yDiffu
        d2v = Diffv * vₕ + yDiffv

        if getJacobian
            Jacu = [Diffu spzeros(Nu, Nv)]
            Jacv = [spzeros(Nv, Nu) Diffv]
        end
    elseif visc ∈ ["qr", "LES", "ML"]
        # get components of strain tensor and its magnitude;
        # the magnitude S_abs is evaluated at pressure points
        S11, S12, S21, S22, S_abs, S_abs_u, S_abs_v =
            strain_tensor(V, t, setup, getJacobian)

        ν_t = turbulent_viscosity(S_abs, setup)
        # Np = setup.grid.Np;
        # ν_t = zeros(Np);

        # we now have the turbulent viscosity at all pressure points

        # to compute the diffusion, we need ν_t at ux, uy, vx and vy
        # locations
        # this means we have to reverse the process of strain_tensor.m: go
        # from pressure points back to the ux, uy, vx, vy locations
        ν_t_ux, ν_t_uy, ν_t_vx, ν_t_vy = interpolate_ν(ν_t, setup)

        # now the total diffusive terms (laminar + turbulent) is as follows
        # note that the factor 2 is because
        # tau = 2*(ν+ν_t)*S(u), with S(u) = 0.5*(grad u + (grad u)^T)

        ν = 1 / setup.fluid.Re # molecular viscosity

        d2u = Dux * (2 * (ν + ν_t_ux) .* S11[:]) + Duy * (2 * (ν + ν_t_uy) .* S12[:])
        d2v = Dvx * (2 * (ν + ν_t_vx) .* S21[:]) + Dvy * (2 * (ν + ν_t_vy) .* S22[:])

        if getJacobian
            # freeze ν_t, i.e. we skip the derivative of ν_t wrt V in
            # the Jacobian
            Jacu1 =
                Dux * 2 * spdiagm(N2, N2, ν + ν_t_ux) * Su_ux +
                Duy * 2 * spdiagm(N1, N1, ν + ν_t_uy) * 1 / 2 * Su_uy
            Jacu2 = Duy * 2 * spdiagm(N2, N2, ν + ν_t_uy) * 1 / 2 * Sv_uy
            Jacv1 = Dvx * 2 * spdiagm(N3, N3, ν + ν_t_vx) * 1 / 2 * Su_vx
            Jacv2 =
                Dvx * 2 * spdiagm(N3, N3, ν + ν_t_vx) * 1 / 2 * Sv_vx +
                Dvy * 2 * spdiagm(N4, N4, ν + ν_t_vy) * Sv_vy
            Jacu = [Jacu1 Jacu2]
            Jacv = [Jacv1 Jacv2]

            if visc == "LES"
                # Smagorinsky
                C_S = setup.visc.Cs
                filter_length = deltax
                K = C_S^2 * filter_length^2
            elseif visc == "qr"
                C_d = deltax^2 / 8
                K = C_d * 0.5 * (1 - α / C_d)^2
            elseif visc == "ML" # mixing-length
                lm = setup.visc.lm # mixing length
                K = (lm^2)
            else
                error("wrong value for visc parameter")
            end
            tmpu1 =
                2 * Dux * spdiagm(N1, N1, S11) * Aν_ux * S_abs_u +
                2 * Duy * spdiagm(N2, N2, S12) * Aν_uy * S_abs_u
            tmpu2 = 2 * Duy * spdiagm(N2, N2, S12) * Aν_uy * S_abs_v
            tmpv1 = 2 * Dvx * spdiagm(N3, N3, S21) * Aν_vx * S_abs_u
            tmpv2 =
                2 * Dvx * spdiagm(N3, N3, S21) * Aν_vx * S_abs_v +
                2 * Dvy * spdiagm(N4, N4, S22) * Aν_vy * S_abs_v
            Jacu += K * [tmpu1 tmpu2]
            Jacv += K * [tmpv1 tmpv2]
        end
    elseif visc == "keps"
        error("k-e implementation in diffusion.m not finished")
    else
        error("wrong specification of viscosity model")
    end

    d2u, d2v, Jacu, Jacv
end
