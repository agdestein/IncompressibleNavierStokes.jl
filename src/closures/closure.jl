function create_neural_closure(m, θ, setup)
    (; dimension, Iu) = setup.grid
    D = dimension()
    function neural_closure(u)
        u = stack(ntuple(α -> u[α][Iu[α]], D))
        u = reshape(u, size(u)..., 1) # One sample
        mu = m(u, θ)
        mu = pad_circular(u)
        sz..., _ = size(mu)
        i = ntuple(α -> Returns(:), D)
        mu = ntuple(α -> mu[i..., α, 1], D)
    end
end
