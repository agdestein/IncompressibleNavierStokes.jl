relative_error(x, y) = sum(norm(x - y) / norm(y) for (x, y) ∈ zip(eachcol(x), eachcol(y))) / size(x, 2)
