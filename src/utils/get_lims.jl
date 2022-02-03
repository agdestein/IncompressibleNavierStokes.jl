"""
    get_lims(x, n = 1.5)

Get approximate lower and upper limits of a field `x` based on the mean and standard
deviation (``\\mu \\pm n \\sigma``). If `x` is constant, a margin of `1e-4` is enforced. This is required for contour
plotting functions that require a certain range.
"""
function get_lims(x, n = 1.5)
    μ = mean(x)
    σ = std(x)
    ≈(μ + σ, μ; rtol = 1e-8, atol = 1e-8) && (σ = 1e-4)
    (μ - n * σ, μ + n * σ)
end
