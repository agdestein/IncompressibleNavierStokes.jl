"""
    cosine_grid(a, b, N)

Create a nonuniform grid of `N + 1` points from `a` to `b` using a cosine
profile, i.e.

```math
x_i = a + \\frac{1}{2} \\left( 1 - \\cos \\left( \\pi \\frac{i}{n} \\right) \\right)
(b - a), \\quad i = 0, \\dots, N
```

See also [`stretched_grid`](@ref).
"""
function cosine_grid(a, b, N) 
    T = typeof(a)
    i = T.(0:N)
    @. a + (b - a) * (1 - cospi(i / N)) / 2
end
