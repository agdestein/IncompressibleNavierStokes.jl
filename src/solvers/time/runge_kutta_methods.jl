"""
    RungeKuttaMethod

Abstract Runge Kutta method.

Original by David Ketcheson
Extended by Benjamin Sanderse
"""
abstract type RungeKuttaMethod end

##================SSP Methods=========================

# Explicit Methods
struct FE11 <: RungeKuttaMethod end
struct SSP22 <: RungeKuttaMethod end
struct SSP42 <: RungeKuttaMethod end
struct SSP33 <: RungeKuttaMethod end
struct SSP43 <: RungeKuttaMethod end
struct SSP104 <: RungeKuttaMethod end
struct rSSPs2 <: RungeKuttaMethod end
struct rSSPs3 <: RungeKuttaMethod end
struct Wray3 <: RungeKuttaMethod end
struct RK56 <: RungeKuttaMethod end
struct DOPRI6 <: RungeKuttaMethod end

# Implicit Methods
struct BE11 <: RungeKuttaMethod end
struct SDIRK34 <: RungeKuttaMethod end
struct ISSPm2 <: RungeKuttaMethod end
struct ISSPs3 <: RungeKuttaMethod end

# Half explicit methods
struct HEM3 <: RungeKuttaMethod end
struct HEM3BS <: RungeKuttaMethod end
struct HEM5 <: RungeKuttaMethod end

# Classical Methods
struct GL1 <: RungeKuttaMethod end
struct GL2 <: RungeKuttaMethod end
struct GL3 <: RungeKuttaMethod end
struct RIA1 <: RungeKuttaMethod end
struct RIA2 <: RungeKuttaMethod end
struct RIA3 <: RungeKuttaMethod end
struct RIIA1 <: RungeKuttaMethod end
struct RIIA2 <: RungeKuttaMethod end
struct RIIA3 <: RungeKuttaMethod end
struct LIIIA2 <: RungeKuttaMethod end
struct LIIIA3 <: RungeKuttaMethod end

#Chebyshev methods
struct CHDIRK3 <: RungeKuttaMethod end
struct CHCONS3 <: RungeKuttaMethod end
struct CHC3 <: RungeKuttaMethod end
struct CHC5 <: RungeKuttaMethod end

# Miscellaneous Methods
struct Mid22 <: RungeKuttaMethod end
struct MTE22 <: RungeKuttaMethod end
struct CN22 <: RungeKuttaMethod end
struct Heun33 <: RungeKuttaMethod end
struct RK33C2 <: RungeKuttaMethod end
struct RK33P2 <: RungeKuttaMethod end
struct RK44 <: RungeKuttaMethod end
struct RK44C2 <: RungeKuttaMethod end
struct RK44C23 <: RungeKuttaMethod end
struct RK44P2 <: RungeKuttaMethod end

# DSRK Methods
struct DSso2 <: RungeKuttaMethod end
struct DSRK2 <: RungeKuttaMethod end
struct DSRK3 <: RungeKuttaMethod end

# "Non-SSP" Methods of Wong & Sπteri
struct NSSP21 <: RungeKuttaMethod end
struct NSSP32 <: RungeKuttaMethod end
struct NSSP33 <: RungeKuttaMethod end
struct NSSP53 <: RungeKuttaMethod end

r = 0

"""
    A, b, c, r = tableau(rk_method, s = 1)

Set up Butcher arrays `A`, `b`, and `c` for the given `rk_method`. Also returns SSP coefficient r
For families of methods, optional input `s` is the number of stages.
"""
function tableau end

##================Explicit Methods=========================
function tableau(::FE11, s = 1)
    #Forward Euler
    s = 1
    r = 1
    A = fill(0, 1, 1)
    b = [1]
    c = [0]
    A, b, c, r
end
function tableau(::SSP22, s = 1)
    s = 2
    r = 1
    A = [0 0; 1 0]
    b = [1 // 2, 1 // 2]
    c = sum(A; dims = 2)
    A, b, c, r
end
function tableau(::SSP42, s = 1)
    s = 4
    r = 3
    A = [0 0 0 0; 1//3 0 0 0; 1//3 1//3 0 0; 1//3 1//3 1//3 0]
    b = fill(1 // 4, s)
    c = sum(A; dims = 2)
    A, b, c, r
end
function tableau(::SSP33, s = 1)
    s = 3
    r = 1
    A = [0 0 0; 1 0 0; 1//4 1//4 0]
    b = [1 // 6, 1 // 6, 2 // 3]
    c = sum(A; dims = 2)
    A, b, c, r
end
function tableau(::SSP43, s = 1)
    s = 4
    r = 2
    A = [0 0 0 0; 1//2 0 0 0; 1//2 1//2 0 0; 1//6 1//6 1//6 0]
    b = [1 // 6, 1 // 6, 1 // 6, 1 // 2]
    c = sum(A; dims = 2)
    A, b, c, r
end
function tableau(::SSP104, s = 1)
    s = 10
    r = 6
    α0 = diag(-1 => ones(1, s - 1))
    α0[6, 5] = 2 // 5
    α0[6, 1] = 3 // 5
    β0 = 1 // 6 * diag(-1 => ones(1, s - 1))
    β0[6, 5] = 1 // 15
    A = (I(s) - α0) \ β0
    b = fill(1 // 10, s)
    c = sum(A; dims = 2)
    A, b, c, r
end
function tableau(::rSSPs2, s = 1)
    #Rational (optimal, low-storage) s-stage 2nd order SSP
    s ≥ 2 || error("Explicit second order SSP family requires s ≥ 2")
    r = s - 1
    α = [zeros(1, s); I(s)]
    α[s+1, s] = (s - 1) // s
    β = α .// r
    α[s+1, 1] = 1 // s
    A = (I(s) - α[1:s, :]) \ β[1:s, :]
    b = β[s+1, :] + α[s+1, :] * A
    b = b'
    c = sum(A; dims = 2)
    A, b, c, r
end
function tableau(::rSSPs3, s = 1)
    #Rational (optimal, low-storage) s^2-stage 3rd order SSP
    if round(sqrt(s)) != sqrt(s) || s < 4
        error("Explicit third order SSP family requires s = n^2, n > 1")
    end
    n = s^2
    r = n - s
    α = [zeros(1, n); I(n)]
    α[s*(s+1)//2+1, s*(s+1)//2] = (s - 1) // (2s - 1)
    β = α // r
    α[s*(s+1)//2+1, (s-1)*(s-2)//2+1] = s // (2s - 1)
    A = (I(n) - α[1:n, :]) \ β[1:n, :]
    b = β[n+1, :] + α[n+1, :] * A
    b = b'
    c = sum(A; dims = 2)
    A, b, c, r
end
function tableau(::Wray3, s = 1)
    # Wray"s RK3
    A = zeros(Rational, 3, 3)
    A[2, 1] = 8 // 15
    A[3, 1] = (8 // 15) - (17 // 60)
    A[3, 2] = 5 // 12
    b = [(8 // 15) - (17 // 60), 0, 3 // 4]
    c = [0, A[2, 1], A[3, 1] + A[3, 2]]
    r = 0 # ?
    A, b, c, r
end
function tableau(::RK56, s = 1)
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
    A, b, c, r
end
function tableau(::DOPRI6, s = 1)
    # Dormand-Price pair
    A = [
        0 0 0 0 0 0
        1//5 0 0 0 0 0
        3//40 9//40 0 0 0 0
        44//45 -56//15 32//9 0 0 0
        19372//6561 -25360//2187 64448//6561 -212//729 0 0
        9017//3168 -355//33 46732//5247 49//176 -5103//18656 0
    ]
    b = [35 // 384, 0, 500 // 1113, 125 // 192, -2187 // 6784, 11 // 84]
    c = sum(A; dims = 2)
    r = 0 #?
    A, b, c, r
end

##================Implicit Methods=========================
function tableau(::BE11, s = 1)
    #Backward Euler
    s = 1
    r = 1.e10
    A = fill(1, 1, 1)
    b = [1]
    c = [1]
    A, b, c, r
end
function tableau(::SDIRK34, s = 1)
    #3-stage, 4th order singly diagonally implicit (SSP)
    s = 3
    r = 1.7588
    g = 0.5 * (1 - cos(π / 18) / sqrt(3) - sin(π / 18))
    q = (0.5 - g)^2
    A = [
        g 0 0
        (0.5-g) g 0
        2g (1-4g) g
    ]
    b = [1 / 24q, 1 - 1 / 12q, 1 / 24q]
    c = sum(A; dims = 2)
    A, b, c, r
end
function tableau(::ISSPm2, s = 1)
    #Optimal DIRK SSP schemes of order 2
    r = 2 * s
    i = repmat(1:s, 1, s)
    j = repmat(1:s, s, 1)
    A = 1 // s * (j < i) + 1 // (2 * s) * (i == j)
    b = fill(1 // s, s)
    c = sum(A; dims = 2)
    A, b, c, r
end
function tableau(::ISSPs3, s = 1)
    #Optimal DIRK SSP schemes of order 3
    if s < 2
        error("Implicit third order SSP schemes require s>=2")
    end
    r = s - 1 + sqrt(s^2 - 1)
    i = repeat(1:s, 1, s)
    j = repeat(1:s, s, 1)
    A = 1 / sqrt(s^2 - 1) * (j < i) + 0.5 * (1 - sqrt((s - 1) / (s + 1))) * (i == j)
    b = fill(1 / s, s)
    c = sum(A; dims = 2)
    A, b, c, r
end

##===================Half explicit methods========================
function tableau(::HEM3, s = 1)

    # Brasey and Hairer
    A = [0 0 0; 1//3 0 0; -1 2 0]
    b = [0, 3 // 4, 1 // 4]
    c = sum(A; dims = 2)
    A, b, c, r
end
function tableau(::HEM3BS, s = 1)
    A = [0 0 0; 1//2 0 0; -1 2 0]
    b = [1 // 6, 2 // 3, 1 // 6]
    c = sum(A; dims = 2)
    A, b, c, r
end
function tableau(::HEM5, s = 1)

    # Brasey and Hairer, 5 stage, 4th order
    A = [
        0 0 0 0 0
        3//10 0 0 0 0
        (1+sqrt(6))/30 (11-4*sqrt(6))/30 0 0 0
        (-79-31*sqrt(6))/150 (-1-4*sqrt(6))/30 (24+11*sqrt(6))/25 0 0
        (14+5*sqrt(6))/6 (-8+7*sqrt(6))/6 (-9-7*sqrt(6))/4 (9-sqrt(6))/4 0
    ]
    b = [0, 0, (16 - sqrt(6)) / 36, (16 + sqrt(6)) / 36, 1 / 9]
    c = sum(A; dims = 2)
    A, b, c, r
end

##================Classical Methods=========================

#Gauss-Legendre methods -- order 2s
function tableau(::GL1, s = 1)
    r = 2
    A = fill(1 // 2, 1, 1)
    b = [1]
    c = [1 // 2]
    A, b, c, r
end
function tableau(::GL2, s = 1)
    r = 0
    A = [
        1/4 1/4-sqrt(3)/6
        1/4+sqrt(3)/6 1/4
    ]
    b = [1 / 2, 1 / 2]
    c = [1 / 2 - sqrt(3) / 6, 1 / 2 + sqrt(3) / 6]
    A, b, c, r
end
function tableau(::GL3, s = 1)
    r = 0
    A = [
        5/36 (80-24*sqrt(15))/360 (50-12*sqrt(15))/360
        (50+15*sqrt(15))/360 2/9 (50-15*sqrt(15))/360
        (50+12*sqrt(15))/360 (80+24*sqrt(15))/360 5/36
    ]
    b = [5 / 18, 4 / 9, 5 / 18]
    c = [(5 - sqrt(15)) / 10, 1 / 2, (5 + sqrt(15)) / 10]

    #Radau IA methods -- order 2s-1
    A, b, c, r
end
function tableau(::RIA1, s = 1)
    # this is implicit Euler
    r = 1
    A = fill(1, 1, 1)
    b = [1]
    c = [0]
    A, b, c, r
end
function tableau(::RIA2, s = 1)
    r = 0
    A = [
        1//4 -1//4
        1//4 5//12
    ]
    b = [1 // 4, 3 // 4]
    c = [0, 2 // 3]
    A, b, c, r
end
function tableau(::RIA3, s = 1)
    r = 0
    A = [
        1/9 (-1-sqrt(6))/18 (-1+sqrt(6))/18
        1/9 (88+7*sqrt(6))/360 (88-43*sqrt(6))/360
        1/9 (88+43*sqrt(6))/360 (88-7*sqrt(6))/360
    ]
    b = [1 // 9, (16 + sqrt(6)) / 36, (16 - sqrt(6)) / 36]
    c = [0, (6 - sqrt(6)) / 10, (6 + sqrt(6)) / 10]

    #Radau IIA methods -- order 2s-1
    A, b, c, r
end
function tableau(::RIIA1, s = 1)
    r = 1
    A = 1
    b = 1
    c = 1
    A, b, c, r
end
function tableau(::RIIA2, s = 1)
    r = 0
    A = [
        5//12 -1//12
        3//4 1//4
    ]
    b = [3 // 4, 1 // 4]
    c = [1 // 3, 1]
    A, b, c, r
end
function tableau(::RIIA3, s = 1)
    r = 0
    A = [
        (88-7*sqrt(6))/360 (296-169*sqrt(6))/1800 (-2+3*sqrt(6))/225
        (296+169*sqrt(6))/1800 (88+7*sqrt(6))/360 (-2-3*sqrt(6))/225
        (16-sqrt(6))/36 (16+sqrt(6))/36 1/9
    ]
    b = [(16 - sqrt(6)) / 36, (16 + sqrt(6)) / 36, 1 / 9]
    c = [(4 - sqrt(6)) / 10, (4 + sqrt(6)) / 10, 1]

    #Lobatto IIIA methods -- order 2s-2
    A, b, c, r
end
function tableau(::LIIIA2, s = 1)
    r = 0
    A = [
        0 0
        1//2 1//2
    ]
    b = [1 // 2, 1 // 2]
    c = [0, 1]
    A, b, c, r
end
function tableau(::LIIIA3, s = 1)
    r = 0
    A = [
        0 0 0
        5//24 1//3 -1//24
        1//6 2//3 1//6
    ]
    b = [1 // 6, 2 // 3, 1 // 6]
    c = [0, 1 // 2, 1]
    A, b, c, r
end

##= Chebyshev methods
function tableau(::CHDIRK3, s = 1)
    # Chebyshev based DIRK (not algebraically stable)
    A = [
        0 0 0
        1//4 1//4 0
        0 1 0
    ]
    b = [1 // 6, 2 // 3, 1 // 6]
    c = [0, 1 // 2, 1]
    r = 0
    A, b, c, r
end
function tableau(::CHCONS3, s = 1)
    A = [
        1//12 -1//6 1//12
        5//24 1//3 -1//24
        1//12 5//6 1//12
    ]
    b = [1 // 6, 2 // 3, 1 // 6]
    c = [0, 1 // 2, 1]
    r = 0
    A, b, c, r
end
function tableau(::CHC3, s = 1)
    # Chebyshev quadrature and C(3) satisfied
    # note this equals Lobatto IIIA
    A = [
        0 0 0
        5//24 1//3 -1//24
        1//6 2//3 1//6
    ]
    b = [1 // 6, 2 // 3, 1 // 6]
    c = [0, 1 // 2, 1]
    A, b, c, r
end
function tableau(::CHC5, s = 1)
    A = [
        0 0 0 0 0
        0.059701779686442 0.095031716019062 -0.012132034355964 0.006643368370744 -0.002798220313558
        0.016666666666667 0.310110028629970 0.200000000000000 -0.043443361963304 0.016666666666667
        0.036131553646891 0.260023298295923 0.412132034355964 0.171634950647605 -0.026368446353109
        0.033333333333333 0.266666666666667 0.400000000000000 0.266666666666667 0.033333333333333
    ]
    b = [1 // 30, 4 // 15, 2 // 5, 4 // 15, 1 // 30]
    c = [0, 0.146446609406726, 0.500000000000000, 0.853553390593274, 1.000000000000000]
    A, b, c, r
end

##==================Miscellaneous Methods================
function tableau(::Mid22, s = 1)
    #Midpoint 22 method
    s = 2
    r = 0.5
    A = [
        0 0
        1//2 0
    ]
    b = [0, 1]
    c = [0, 1 // 2]
    A, b, c, r
end
function tableau(::MTE22, s = 1)
    #Minimal truncation error 22 method (Heun)
    s = 2
    r = 0.5
    A = [
        0 0
        2//3 0
    ]
    b = [1 // 4, 3 // 4]
    c = [0, 2 // 3]
    A, b, c, r
end
function tableau(::CN22, s = 1)
    #Crank-Nicholson
    s = 2
    r = 2
    A = [
        0 0
        1//2 1//2
    ]
    b = [1 // 2, 1 // 2]
    c = [0, 1]
    A, b, c, r
end
function tableau(::Heun33, s = 1)
    s = 3
    r = 0
    A = [0 0 0; 1//3 0 0; 0 2//3 0]
    b = [1 // 4, 0, 3 // 4]
    c = sum(A; dims = 2)
    A, b, c, r
end
function tableau(::RK33C2, s = 1)
    # RK3 satisfying C(2) for i=3
    A = [0 0 0; 2//3 0 0; 1//3 1//3 0]
    b = [1 // 4, 0, 3 // 4]
    c = [0, 2 // 3, 2 // 3]
    A, b, c, r
end
function tableau(::RK33P2, s = 1)
    # RK3 satisfying the second order condition for the pressure
    A = [0 0 0; 1//3 0 0; -1 2 0]
    b = [0, 3 // 4, 1 // 4]
    c = [0, 1 // 3, 1]
    A, b, c, r
end
function tableau(::RK44, s = 1)
    #Classical fourth order
    s = 4
    r = 0
    A = [0 0 0 0; 1//2 0 0 0; 0 1//2 0 0; 0 0 1 0]
    b = [1 // 6, 1 // 3, 1 // 3, 1 // 6]
    c = sum(A; dims = 2)
    A, b, c, r
end
function tableau(::RK44C2, s = 1)
    # RK4 satisfying C(2) for i=3
    A = [0 0 0 0; 1//4 0 0 0; 0 1//2 0 0; 1 -2 2 0]
    b = [1 // 6, 0, 2 // 3, 1 // 6]
    c = [0, 1 // 4, 1 // 2, 1]
    A, b, c, r
end
function tableau(::RK44C23, s = 1)
    # RK4 satisfying C(2) for i=3 and c2=c3
    A = [0 0 0 0; 1//2 0 0 0; 1//4 1//4 0 0; 0 -1 2 0]
    b = [1 // 6, 0, 2 // 3, 1 // 6]
    c = [0, 1 // 2, 1 // 2, 1]
    A, b, c, r
end
function tableau(::RK44P2, s = 1)
    # RK4 satisfying the second order condition for the pressure (but not third
    # order)
    A = [0 0 0 0; 1 0 0 0; 3//8 1//8 0 0; -1//8 -3//8 3//2 0]
    b = [1 // 6, -1 // 18, 2 // 3, 2 // 9]
    c = [0, 1, 1 // 2, 1]
    A, b, c, r
end

##===================DSRK Methods========================
function tableau(::DSso2, s = 1)
    #CBM"s DSRKso2
    s = 2
    isdsrk = 1
    A = [
        3//4 -1//4
        1 0
    ]
    W = [
        1//2 0
        1 0
    ]
    b = [1, 0]
    c = [1 // 2, 1]
    A, b, c, r
end
function tableau(::DSRK2, s = 1)
    #CBM"s DSRK2
    s = 2
    A = [
        1//2 -1//2
        1//2 1//2
    ]
    W = [
        0 0
        1//2 1//2
    ]
    b = [1 // 2, 1 // 2]
    c = [0, 1]
    A, b, c, r
end
function tableau(::DSRK3, s = 1)
    #Zennaro"s DSRK3
    s = 3
    isdsrk = 1
    A = [
        5//2 -2 -1//2
        -1 2 -1//2
        1//6 2//3 1//6
    ]
    W = [
        0 0 0
        7//24 1//6 1//24
        1//6 2//3 1//6
    ]
    b = [1 // 6, 2 // 3, 1 // 6]
    c = [0, 1 // 2, 1]
    A, b, c, r
end

##==================="Non-SSP" Methods of Wong & Sπteri========================
function tableau(::NSSP21, s = 1)
    m = 2
    r = 0
    A = [
        0 0
        3//4 0
    ]
    b = [0, 1]
    c = [0, 3 // 4]
    A, b, c, r
end
function tableau(::NSSP32, s = 1)
    m = 3
    r = 0
    A = [
        0 0 0
        1//3 0 0
        0 1 0
    ]
    b = [1 // 2, 0, 1 // 2]
    c = [0, 1 // 3, 1]
    A, b, c, r
end
function tableau(::NSSP33, s = 1)
    m = 3
    r = 0
    A = [
        0 0 0
        -4//9 0 0
        7//6 -1//2 0
    ]
    b = [1 // 4, 0, 3 // 4]
    c = [0, -4 // 9, 2 // 3]
    A, b, c, r
end
function tableau(::NSSP53, s = 1)
    m = 5
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
    A, b, c, r
end
