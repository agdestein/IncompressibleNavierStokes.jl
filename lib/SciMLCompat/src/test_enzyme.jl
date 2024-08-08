using IncompressibleNavierStokes 
using INSEnzyme
using KernelAbstractions
using Enzyme
Enzyme.API.runtimeActivity!(true)

## The test for psolve is missing
# [...]



# here you just have to run the tests for the functions implemented in INSEnzyme.jl
B = get_backend(rand(Float32, 10))
myapply_bc_u! = _get_enz_bc_u!(B, setup)

nreps = 10000
_, time_INS, allocation_INS, gc_INS, memory_counters_INS = @timed for i in 1:nreps
    A = (rand(Float32,size(cache_p)[1],size(cache_p)[1]),rand(Float32,size(cache_p)[1],size(cache_p)[1]))
    IncompressibleNavierStokes.apply_bc_u!(A, 0.0f0, setup);
end
_, time_enz, allocation_enz, gc_enz, memory_counters_enz = @timed for i in 1:nreps
    A = (rand(Float32,size(cache_p)[1],size(cache_p)[1]),rand(Float32,size(cache_p)[1],size(cache_p)[1]))
    myapply_bc_u!(A);
end
# Compare the execution times and the memory allocations
# assert that the time is less than 10% more than the time of the INS implementation
@assert time_enz < 1.1*time_INS "OK: time_enz = $time_enz, time_INS = $time_INS"
# assert that the memory allocation is less than 10% more than the memory allocation of the INS implementation
@assert allocation_enz < 1.1*allocation_INS "OK: allocation_enz = $allocation_enz, allocation_INS = $allocation_INS"

# Check if the implementation is correct
for i in 1:nreps
    A = (rand(Float32,size(cache_p)[1],size(cache_p)[1]),rand(Float32,size(cache_p)[1],size(cache_p)[1]));
    A0 = (copy(A[1]), copy(A[2])) ;                  ;
    B = (copy(A[1]), copy(A[2]))                    ;
    IncompressibleNavierStokes.apply_bc_u!(A, T(0), setup)  ;
    myapply_bc_u!(B);
    @assert A[1] == B[1]                  
    @assert A[2] == B[2]
    @assert A[1] != A0[1]
    @assert A[2] != A0[2]
end

# Check if it is differentiable
A = (rand(Float32,size(cache_p)[1],size(cache_p)[1]),rand(Float32,size(cache_p)[1],size(cache_p)[1]))
dA = Enzyme.make_zero(A)
@timed Enzyme.autodiff(Enzyme.Reverse, myapply_bc_u!, Const, DuplicatedNoNeed(A, dA))






myapply_bc_p! = get_bc_p!(cache_p, setup) 

# Speed test
@timed for i in 1:10000
    A = rand(Float32,size(cache_p)[1],size(cache_p)[2]);
    IncompressibleNavierStokes.apply_bc_p!(A, 0.0f0, setup);
end
@timed for i in 1:10000
    A = rand(Float32,size(cache_p)[1],size(cache_p)[2]);
    myapply_bc_p!(A);
end

# Check if the implementation is correct
for i in 1:10000
    A = rand(Float32,size(cache_p)[1],size(cache_p)[2]) ;
    A0 = copy(A)                   ;
    B = copy(A)                    ;
    IncompressibleNavierStokes.apply_bc_p!(A, T(0), setup)  ;
    myapply_bc_p!(B);
    @assert A == B                  
    @assert A != A0                  
end

# Check if it is differentiable
A = rand(Float32,size(cache_p)[1],size(cache_p)[2]);
dA = Enzyme.make_zero(A);
@timed Enzyme.autodiff(Enzyme.Reverse, myapply_bc_p!, Const, DuplicatedNoNeed(A, dA))











F = rand(Float32,size(cache_p)[1],size(cache_p)[2])
z = Enzyme.make_zero(F)
u = random_field(setup, T(0))
my_f = get_divergence!(F, setup)
(; grid, Re) = setup
(; Δ, Δu, A) = grid
my_f(F, u, z)#, stack(Δ))

@timed for i in 1:1000
    F0 = rand(Float32,size(cache_p)[1],size(cache_p)[2]);
    F = copy(F0);
    u = random_field(setup, T(0));
    IncompressibleNavierStokes.divergence!(F, u, setup);
    @assert F != F0
end
@timed for i in 1:1000
    F = rand(Float32,size(cache_p)[1],size(cache_p)[2]);
    u = random_field(setup, T(0));
    z = Enzyme.make_zero(F)
    my_f(F, u, z)#, stack(Δ));
end
# Check if the implementation is correct
using Statistics
for i in 1:1000
    F = rand(Float32,size(cache_p)[1],size(cache_p)[2]);
    u = random_field(setup, T(0)) 
    A0 = copy(F)                   ;
    B = copy(F)                    ;
    IncompressibleNavierStokes.divergence!(F, u, setup)  ;
    z = Enzyme.make_zero(F)
    my_f(B, u, z)#, stack(Δ));
    @assert F == B                  
end

F = rand(Float32,size(cache_p)[1],size(cache_p)[2]);
dF = Enzyme.make_zero(F);
u = random_field(setup, T(0));
du = Enzyme.make_zero(u);
d = Enzyme.make_zero(F);
dd = Enzyme.make_zero(d);
z = Enzyme.make_zero(F);
dΔ = Enzyme.make_zero(stack(Δ));
# Test if it is differentiable
@timed Enzyme.autodiff(Enzyme.Reverse, my_f, Const, DuplicatedNoNeed(F, dF), DuplicatedNoNeed(u, du), DuplicatedNoNeed(d, dd))






u = random_field(setup, T(0))
p = rand(T,(n+2,n+2))
myapplypressure! = get_applypressure!(u, setup)
myapplypressure!(u, p)#, Δu)
IncompressibleNavierStokes.applypressure!(u, p, setup)

# Speed test
@timed for i in 1:1000
    u = random_field(setup, T(0))
    p = rand(T,(n+2,n+2))
    IncompressibleNavierStokes.applypressure!(u, p, setup)
end
@timed for i in 1:1000
    u = random_field(setup, T(0))
    p = rand(T,(n+2,n+2))
    myapplypressure!(u, p)#, Δu)
end

# Compare with INS
for i in 1:1000
    u = random_field(setup, T(0))
    p = rand(T,(n+2,n+2))
    u0 = copy.(u)
    IncompressibleNavierStokes.applypressure!(u, p, setup)
    myapplypressure!(u0, p)#, Δu)
    @assert u == u0
end

# Check if it is differentiable
u = random_field(setup, T(0))
p = rand(T,(n+2,n+2))
du = Enzyme.make_zero(u)
dp = Enzyme.make_zero(p)
dΔu = Enzyme.make_zero(Δu)
@timed Enzyme.autodiff(Enzyme.Reverse, myapplypressure!, Const, DuplicatedNoNeed(u, du), DuplicatedNoNeed(p, dp))




(; Δ, Δu, A) = grid
ν = 1 / Re

u = random_field(setup, T(0))
F = random_field(setup, T(0))
my_f = get_momentum!(F, u, nothing, setup)
sΔ = stack(Δ)
sΔu = stack(Δu)
my_f(F, u, T(0))#, sΔ, sΔu)#, ν)#, A, T(0))

# Check if it is differentiable
u = random_field(setup, T(0))
F = random_field(setup, T(0))
du = Enzyme.make_zero(u)
dF = Enzyme.make_zero(F)
dΔu = Enzyme.make_zero(Δu)
dΔ = Enzyme.make_zero(Δ)
dν = Enzyme.make_zero(ν)
dA = Enzyme.make_zero(A)
dsΔ = Enzyme.make_zero(sΔ)
dsΔu = Enzyme.make_zero(sΔu)
 @timed Enzyme.autodiff(Enzyme.Reverse, my_f, Const, DuplicatedNoNeed(F, dF), DuplicatedNoNeed(u, du), Const(T(0)), DuplicatedNoNeed(sΔ,dsΔ), DuplicatedNoNeed(sΔu, dsΔu))
@timed Enzyme.autodiff(Enzyme.Reverse, my_f, Const, DuplicatedNoNeed(F, dF), DuplicatedNoNeed(u, du), Const(T(0)))

@timed for i in 1:1000
    u = random_field(setup, T(0))
    F = random_field(setup, T(0))
    IncompressibleNavierStokes.momentum!(F, u, nothing, T(0), setup)
end
@timed for i in 1:1000
    u = random_field(setup, T(0))
    F = random_field(setup, T(0))
    my_f(F, u, T(0))#, sΔ, sΔu)
end

# Check if the implementation is correct
for i in 1:1000
    u = random_field(setup, T(0))
    F = random_field(setup, T(0))
    u0 = copy.(u)
    F0 = copy.(F)
    IncompressibleNavierStokes.momentum!(F, u, nothing, T(0), setup)
    my_f(F0, u, T(0))#, sΔ, sΔu)
    @assert F == F0
end
