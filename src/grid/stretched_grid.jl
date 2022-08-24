"""
    stretched_grid(a, b, N, s = 1)

Create a nonuniform grid of `n` points from `a` to `b` with a stretch factor of `s`. If `s =
1`, return a uniform spacing from `a` to `b`. Otherwise, return a vector ``x \\in
\\mathbb{R}^{N + 1}`` such that ``x_n = a + \\sum_{i = 1}^n s^{i - 1} h`` for ``n = 0,
\\dots , N``. Setting ``x_N = b`` then gives ``h = (b - a) \\frac{1 - s}{1 - s^N}``,
resulting in

```math
x_n = a + (b - a) \\frac{1 - s^n}{1 - s^N}, \\quad n = 0, \\dots, N.
```

Note that `stretched_grid(a, b, N, s)[n]` corresponds to ``x_{n - 1}``.

See also [`cosine_grid`](@ref).
"""
function stretched_grid(a, b, N, s = 1)
    s > 0 || error("The strecth factor must be positive")
    if s ≈ 1
        LinRange(a, b, N + 1)
    else
        map(i -> a + (b - a) * (1 - s^i) / (1 - s^N), 0:N)
    end
end
