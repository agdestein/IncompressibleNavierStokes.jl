"""
    nonuniform_grid(a, b, N, s = 1)

Create a nonuniform of `n` points from `a` to `b` with a stretch factor of `s`.
"""
function nonuniform_grid(a, b, N, s = 1)
    s > 0 || error("The strecth factor must be positive")
    if s ≈ 1
        LinRange(a, b, N + 1)
    else
        n = 0:N
        @. a + (b - a)(1 - s^n)(1 - s^N)
    end
end
