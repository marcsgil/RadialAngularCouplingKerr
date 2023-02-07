using FreeParaxialPropagation
using KerrPropagation
using LinearAlgebra

overlap(ψ₁,ψ₂,area_element) = area_element*abs(ψ₁ ⋅ ψ₂)

function overlap(ψ₁::AbstractArray{T,3},ψ₂::AbstractArray{T,3},area_element) where T
    [ overlap(view(ψ₁,:,:,n),view(ψ₂,:,:,n),area_element) for n in axes(ψ₁,3) ]
end
##
rs = LinRange(-6,6,1024)
zs = LinRange(0,.5,32)

χ = 10^2
l₀ = 1
##
ψ₀ = LG(rs,rs,0,l=l₀) |> cu
ψs = convert(Array{ComplexF64},kerr_propagation(ψ₀,rs,rs,zs,1024,χ=χ))
ψ0s = convert(Array{ComplexF64},free_propagation(ψ₀,rs,rs,zs))
δψs = ψs - ψ0s
##
using Plots
os = reduce(hcat,[overlap(LG(rs,rs,zs,p=p,l=l₀,γ₀ = 1/√3),δψs,(rs[2]-rs[1])^2) for p in 0:abs(l₀)+2])

plot(zs,os)
##
using InteractiveBeamVizualization
interactive_vizualization(ψs, rs, rs, zs)