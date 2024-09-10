@testitem "Aqua" begin
    using Aqua
    Aqua.test_all(NeuralClosure; ambiguities = false)
end
