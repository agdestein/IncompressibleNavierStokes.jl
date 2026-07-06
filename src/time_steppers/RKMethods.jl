"""
Set up Butcher arrays `A`, `b`, and `c`, as well as and SSP coefficient `r`.
For families of methods, optional input `s` is the number of stages.

Original (MATLAB) by David Ketcheson, extended by Benjamin Sanderse.
"""
module RKMethods

using IncompressibleNavierStokes: IncompressibleNavierStokes, runge_kutta_method
using DocStringExtensions

# Inherit docstring templates
@template (MODULES, FUNCTIONS, METHODS, TYPES) = IncompressibleNavierStokes

# Explicit Methods
export FE11, SSP22, SSP42, SSP33, SSP43, SSP104, rSSPs2, rSSPs3, Wray3, RK56, DOPRI6

# Half explicit methods
export HEM3, HEM3BS, HEM5

# Miscellaneous Methods
export Mid22, MTE22, Heun33, RK33C2, RK33P2, RK44, RK44C2, RK44C23, RK44P2

# "Non-SSP" Methods of Wong & Spiteri
export NSSP21, NSSP32, NSSP33, NSSP53

# Default SSP coefficient
# r = 0

## ================Explicit Methods=========================

"FE11 (Forward Euler)."
function FE11(; kwargs...)
    A = fill(0, 1, 1)
    b = [1]
    c = [0]
    r = 1
    runge_kutta_method(A, b, c, r; kwargs...)
end

"SSP22."
function SSP22(; kwargs...)
    A = [0 0; 1 0]
    b = [1 // 2, 1 // 2]
    c = sum(eachcol(A))
    r = 1
    runge_kutta_method(A, b, c, r; kwargs...)
end

"SSP42."
function SSP42(; kwargs...)
    s = 4
    A = [0 0 0 0; 1//3 0 0 0; 1//3 1//3 0 0; 1//3 1//3 1//3 0]
    b = fill(1 // 4, s)
    c = sum(eachcol(A))
    r = 3
    runge_kutta_method(A, b, c, r; kwargs...)
end

"SSP33."
function SSP33(; kwargs...)
    A = [0 0 0; 1 0 0; 1//4 1//4 0]
    b = [1 // 6, 1 // 6, 2 // 3]
    c = sum(eachcol(A))
    r = 1
    runge_kutta_method(A, b, c, r; kwargs...)
end

"SSP43."
function SSP43(; kwargs...)
    A = [0 0 0 0; 1//2 0 0 0; 1//2 1//2 0 0; 1//6 1//6 1//6 0]
    b = [1 // 6, 1 // 6, 1 // 6, 1 // 2]
    c = sum(eachcol(A))
    r = 2
    runge_kutta_method(A, b, c, r; kwargs...)
end

"SSP104."
function SSP104(; kwargs...)
    s = 10
    α0 = diagm(-1 => fill(1 // 1, s - 1))
    α0[6, 5] = 2 // 5
    α0[6, 1] = 3 // 5
    β0 = 1 // 6 * diagm(-1 => fill(1 // 1, s - 1))
    β0[6, 5] = 1 // 15
    A = (I(s) - α0) \ β0
    b = fill(1 // 10, s)
    c = sum(eachcol(A))
    r = 6
    runge_kutta_method(A, b, c, r; kwargs...)
end

"Rational (optimal, low-storage) `s`-stage 2nd order SSP."
function rSSPs2(s = 2; kwargs...)
    s ≥ 2 || error("Explicit second order SSP family requires s ≥ 2")
    r = s - 1
    α = [fill(0, 1, s); 1 // 1 * I(s)]
    α[s+1, s] = (s - 1) // s
    β = α .// r
    α[s+1, 1] = 1 // s
    A = (I(s) - α[1:s, :]) \ β[1:s, :]
    b = β[s+1, :] + A'α[s+1, :]
    c = sum(eachcol(A))
    runge_kutta_method(A, b, c, r; kwargs...)
end

"Rational (optimal, low-storage) `s^2`-stage 3rd order SSP."
function rSSPs3(s = 4; kwargs...)
    if !(round(sqrt(s)) ≈ sqrt(s)) || s < 4
        error("Explicit third order SSP family requires s = n^2, n > 1")
    end
    n = s^2
    r = n - s
    α = [fill(0, 1, n); 1 // 1 * I(n)]
    α[s*(s+1)÷2+1, s*(s+1)÷2] = (s - 1) // (2s - 1)
    β = α .// r
    α[s*(s+1)÷2+1, (s-1)*(s-2)÷2+1] = s // (2s - 1)
    A = (I(n) - α[1:n, :]) \ β[1:n, :]
    b = β[n+1, :] + A'α[n+1, :]
    c = sum(eachcol(A))
    runge_kutta_method(A, b, c, r; kwargs...)
end

"Wray's RK3."
function Wray3(; kwargs...)
    A = zeros(Rational, 3, 3)
    A[2, 1] = 8 // 15
    A[3, 1] = (8 // 15) - (17 // 60)
    A[3, 2] = 5 // 12
    b = [(8 // 15) - (17 // 60), 0, 3 // 4]
    c = [0, A[2, 1], A[3, 1] + A[3, 2]]
    r = 0
    runge_kutta_method(A, b, c, r; kwargs...)
end

"RK56."
function RK56(; kwargs...)
    r = 0
    A = [
        0 0 0 0 0 0
        1//4 0 0 0 0 0
        1//8 1//8 0 0 0 0
        0 0 1//2 0 0 0
        3//16 -3//8 3//8 9//16 0 0
        -3//7 8//7 6//7 -12//7 8//7 0
    ]
    b = [7 // 90, 0, 16 // 45, 2 // 15, 16 // 45, 7 // 90]
    c = [0, 1 // 4, 1 // 4, 1 // 2, 3 // 4, 1]
    runge_kutta_method(A, b, c, r; kwargs...)
end

"Dormand-Price pair."
function DOPRI6(; kwargs...)
    A = [
        0 0 0 0 0 0
        1//5 0 0 0 0 0
        3//40 9//40 0 0 0 0
        44//45 -56//15 32//9 0 0 0
        19372//6561 -25360//2187 64448//6561 -212//729 0 0
        9017//3168 -355//33 46732//5247 49//176 -5103//18656 0
    ]
    b = [35 // 384, 0, 500 // 1113, 125 // 192, -2187 // 6784, 11 // 84]
    c = sum(eachcol(A))
    r = 0
    runge_kutta_method(A, b, c, r; kwargs...)
end

## ===================Half explicit methods========================

"Brasey and Hairer."
function HEM3(; kwargs...)
    r = 0
    A = [0 0 0; 1//3 0 0; -1 2 0]
    b = [0, 3 // 4, 1 // 4]
    c = sum(eachcol(A))
    runge_kutta_method(A, b, c, r; kwargs...)
end

"HEM3BS."
function HEM3BS(; kwargs...)
    r = 0
    A = [0 0 0; 1//2 0 0; -1 2 0]
    b = [1 // 6, 2 // 3, 1 // 6]
    c = sum(eachcol(A))
    runge_kutta_method(A, b, c, r; kwargs...)
end

"Brasey and Hairer, 5 stage, 4th order."
function HEM5(; kwargs...)
    r = 0
    A = [
        0 0 0 0 0
        3//10 0 0 0 0
        (1+sqrt(6))/30 (11-4*sqrt(6))/30 0 0 0
        (-79-31*sqrt(6))/150 (-1-4*sqrt(6))/30 (24+11*sqrt(6))/25 0 0
        (14+5*sqrt(6))/6 (-8+7*sqrt(6))/6 (-9-7*sqrt(6))/4 (9-sqrt(6))/4 0
    ]
    b = [0, 0, (16 - sqrt(6)) / 36, (16 + sqrt(6)) / 36, 1 / 9]
    c = sum(eachcol(A))
    runge_kutta_method(A, b, c, r; kwargs...)
end

## ==================Miscellaneous Methods================

"Midpoint 22 method."
function Mid22(; kwargs...)
    A = [0 0; 1//2 0]
    b = [0, 1]
    c = [0, 1 // 2]
    r = 1 // 2
    runge_kutta_method(A, b, c, r; kwargs...)
end

"Minimal truncation error 22 method (Heun)."
function MTE22(; kwargs...)
    A = [0 0; 2//3 0]
    b = [1 // 4, 3 // 4]
    c = [0, 2 // 3]
    r = 1 // 2
    runge_kutta_method(A, b, c, r; kwargs...)
end

"Heun33."
function Heun33(; kwargs...)
    A = [0 0 0; 1//3 0 0; 0 2//3 0]
    b = [1 // 4, 0, 3 // 4]
    c = sum(eachcol(A))
    r = 0
    runge_kutta_method(A, b, c, r; kwargs...)
end

"RK3 satisfying C(2) for i=3."
function RK33C2(; kwargs...)
    A = [0 0 0; 2//3 0 0; 1//3 1//3 0]
    b = [1 // 4, 0, 3 // 4]
    c = [0, 2 // 3, 2 // 3]
    r = 0
    runge_kutta_method(A, b, c, r; kwargs...)
end

"RK3 satisfying the second order condition for the pressure."
function RK33P2(; kwargs...)
    A = [0 0 0; 1//3 0 0; -1 2 0]
    b = [0, 3 // 4, 1 // 4]
    c = [0, 1 // 3, 1]
    r = 0
    runge_kutta_method(A, b, c, r; kwargs...)
end

"Classical fourth order."
function RK44(; kwargs...)
    A = [0 0 0 0; 1//2 0 0 0; 0 1//2 0 0; 0 0 1 0]
    b = [1 // 6, 1 // 3, 1 // 3, 1 // 6]
    c = sum(eachcol(A))
    r = 0
    runge_kutta_method(A, b, c, r; kwargs...)
end

"RK4 satisfying C(2) for i=3."
function RK44C2(; kwargs...)
    A = [0 0 0 0; 1//4 0 0 0; 0 1//2 0 0; 1 -2 2 0]
    b = [1 // 6, 0, 2 // 3, 1 // 6]
    c = [0, 1 // 4, 1 // 2, 1]
    r = 0
    runge_kutta_method(A, b, c, r; kwargs...)
end

"RK4 satisfying C(2) for i=3 and c2=c3."
function RK44C23(; kwargs...)
    A = [0 0 0 0; 1//2 0 0 0; 1//4 1//4 0 0; 0 -1 2 0]
    b = [1 // 6, 0, 2 // 3, 1 // 6]
    c = [0, 1 // 2, 1 // 2, 1]
    r = 0
    runge_kutta_method(A, b, c, r; kwargs...)
end

"RK4 satisfying the second order condition for the pressure (but not third order)."
function RK44P2(; kwargs...)
    A = [0 0 0 0; 1 0 0 0; 3//8 1//8 0 0; -1//8 -3//8 3//2 0]
    b = [1 // 6, -1 // 18, 2 // 3, 2 // 9]
    c = [0, 1, 1 // 2, 1]
    r = 0
    runge_kutta_method(A, b, c, r; kwargs...)
end

## ==================="Non-SSP" Methods of Wong & Spiteri========================

"NSSP21."
function NSSP21(; kwargs...)
    A = [
        0 0
        3//4 0
    ]
    b = [0, 1]
    c = [0, 3 // 4]
    r = 0
    runge_kutta_method(A, b, c, r; kwargs...)
end

"NSSP32."
function NSSP32(; kwargs...)
    r = 0
    A = [
        0 0 0
        1//3 0 0
        0 1 0
    ]
    b = [1 // 2, 0, 1 // 2]
    c = [0, 1 // 3, 1]
    runge_kutta_method(A, b, c, r; kwargs...)
end

"NSSP33."
function NSSP33(; kwargs...)
    r = 0
    A = [
        0 0 0
        -4//9 0 0
        7//6 -1//2 0
    ]
    b = [1 // 4, 0, 3 // 4]
    c = [0, -4 // 9, 2 // 3]
    runge_kutta_method(A, b, c, r; kwargs...)
end

"NSSP53."
function NSSP53(; kwargs...)
    r = 0
    A = [
        0 0 0 0 0
        1//7 0 0 0 0
        0 3//16 0 0 0
        0 0 1//3 0 0
        0 0 0 2//3 0
    ]
    b = [1 // 4, 0, 0, 0, 3 // 4]
    c = [0, 1 // 7, 3 // 16, 1 // 3, 2 // 3]
    runge_kutta_method(A, b, c, r; kwargs...)
end

end
