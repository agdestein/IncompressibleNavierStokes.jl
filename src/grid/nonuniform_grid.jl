"""
    nonuniform_grid(Δz, z_low, z_up, sz, ϵ = 1e-12)

Cenerate a non-uniform grid, from `z_low` to `z_up`, starting Δz and having stretch factor close to `sz`
Includes check for uniform grid and, in that case, adapts Δz if necessary.
"""
function nonuniform_grid(Δz, z_low, z_up, sz, ϵ = 1e-12)
    if sz == 1
        n = (z_up - z_low) / Δz
        if abs(n - round(n)) > ϵ
            Δz = (z_up - z_low) / floor(n)
        end
        z = collect(z_low:Δz:z_up)
        dz = diff(z)
    else
        i = 1
        z[i] = z_low

        # First check the number of grid points by using the specified sz
        while z[i] + Δz * sz^i ≤ z_up
            i = i + 1
            z[i] = z[i-1] + Δz * (sz^(i - 2))
        end

        # Adapt sz such that the upper boundary is exactly satisfied
        # Sum of powerseries should be z_up-z_low
        S = z_up - z[1]           # Sum wanted
        n = length(z) - 1         # Number of intervals

        # Secant method for nonlinear iteration
        s[1] = sz
        s[2] = sz^2
        i = 1
        while abs(s[i+1] - s[i]) > ϵ
            s[i+2] =
                s[i+1] -
                (s[i+1] - s[i]) * (S * (s[i+1] - 1) - Δz * (s[i+1]^n - 1)) / (
                    (S * (s[i+1] - 1) - Δz * (s[i+1]^n - 1)) -
                    (S * (s[i] - 1) - Δz * (s[i]^n - 1))
                )

            i += 1
        end
        sz = s[end]

        # Update z now based on the new sz
        z = zeros(n)
        i = 1
        z[i] = z_low
        while i < n + 1
            z[i+1] = z[i] + Δz * sz^(i - 1)
            i = i + 1
        end

        dz = diff(z)

        z = z[:]
        dz = dz[:]
    end

    z, dz
end
