"""
    cosine_grid(a, b, N)

Create a nonuniform of `N + 1` points from `a` to `b` using a cosine profile, i.e.

```math
x_i = a + \\frac{1}{2} (1 - cos(\\pi * \\frac{i}{n})) (b - a), \\quad i = 0, \\dots, N
```
"""
cosine_grid(a, b, N) = map(i -> a + (b - a) * (1 - cospi(i / N)) / 2, 0:N)
