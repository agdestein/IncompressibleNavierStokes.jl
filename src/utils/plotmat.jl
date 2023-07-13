"""
    plotmat(A)

Plot matrix.
"""
function plotmat end

plotmat(A) = heatmap(reverse(A'; dims = 2))
plotmat(A::AbstractSparseMatrix) = plotmat(Matrix(A))
