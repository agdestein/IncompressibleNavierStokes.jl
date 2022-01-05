"""
    max_size(grid)

Get size of the largest grid element.
"""
function max_size(grid)
    (; hx, hy) = setup.grid
    Δx, Δy = maximum(hx), maximum(hy)
    √(Δx^2 + Δy^2)
end
