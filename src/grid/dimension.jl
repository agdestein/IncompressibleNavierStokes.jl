"""
    Dimension(N)

Represent an `N`-dimensional space.
Returns `N` when called.

```example
julia> d = Dimension(3)
Dimension{3}()

julia> d()
3
```
"""
struct Dimension{N} end

Dimension(N) = Dimension{N}()

(::Dimension{N})() where {N} = N
