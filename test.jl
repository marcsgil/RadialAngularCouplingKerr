using FreeParaxialPropagation,KerrPropagation
using LinearAlgebra

overlap(ψ₁,ψ₂,area_element) = area_element*abs(ψ₁ ⋅ ψ₂)

function overlap(ψ₁::AbstractArray{T1,3},ψ₂::AbstractArray{T2,3},area_element) where {T1,T2}
    map( (a,b) -> overlap(a,b,area_element), eachslice(ψ₁,dims=3), eachslice(ψ₂,dims=3) )
end

rs = LinRange(-5,5,1024)
zs = LinRange(0,.1,64)

ψ₀ = lg(rs,rs,0,l=2) |> cu
ψ1s = kerr_propagation(ψ₀,rs,rs,zs,2048,χ=1,k=2)
ψ2s = kerr_propagation(ψ₀,rs,rs,zs,2048,χ=.0,k=2)

ψ2s_ex = 

ψ0s = free_propagation(ψ₀,rs,rs,zs,k=2)
δψ1s = ψ1s - ψ0s
δψ2s = ψ2s - ψ0s

os1 = stack(overlap(lg(rs,rs,zs,p=p,l=2,w0 = 1/√3,k=2) |> cu,δψ1s,(rs[2]-rs[1])^2) for p in 0:2)
os2 = stack(overlap(lg(rs,rs,zs,p=p,l=2,w0 = 1/√3,k=2) |> cu,δψ2s,(rs[2]-rs[1])^2) for p in 0:2)

ψ1s ≈ ψ2s
δψ1s ≈ δψ2s
os1 ≈ os2
##


overlap(lg(rs,rs,zs,l=2,k=2) |> cu,ψ2s,(rs[2]-rs[1])^2)