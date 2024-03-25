"""
    max_size(grid)

Get size of the largest grid element.
"""
function max_size(grid)
    (; Δ) = grid
    m = maximum.(Δ)
    sqrt(sum(m .^ 2))
end
