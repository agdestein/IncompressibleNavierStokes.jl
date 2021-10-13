abstract type PressureSolver end

struct DirectPressureSolver <: PressureSolver end
struct CGPressureSolver <: PressureSolver end
