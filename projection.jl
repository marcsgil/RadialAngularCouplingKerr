using FreeParaxialPropagation
using KerrPropagation
using LinearAlgebra

using Plots,LaTeXStrings

default()
default(label=false,width=3,size=(600,400), markersize = 4, msw=0, 
palette=:seaborn_bright, tickfontsize=12, labelfontsize=18,
legendfontsize=12, fontfamily="Computer Modern",dpi=1000,grid=false,framestyle = :box)
##

overlap(ψ₁,ψ₂,area_element) = area_element*abs(ψ₁ ⋅ ψ₂)

function overlap(ψ₁::AbstractArray{T1,3},ψ₂::AbstractArray{T2,3},area_element) where {T1,T2}
    map( (a,b) -> overlap(a,b,area_element), eachslice(ψ₁,dims=3), eachslice(ψ₂,dims=3) )
end
##
rs = LinRange(-5,5,512)
zs = LinRange(0,0.5,32)

χ = 1
l₀ = 2
##
ψ₀ = lg(rs,rs,0,l=l₀) |> cu
ψs = kerr_propagation(ψ₀,rs,rs,zs,2048,χ=χ)
ψ0s = free_propagation(ψ₀,rs,rs,zs)
δψs = ψs - ψ0s
##


#os = stack(overlap(lg(rs,rs,zs,p=p,l=l₀,w0 = 1/√3) |> cu,δψs,(rs[2]-rs[1])^2) for p in 0:abs(l₀)+2)

plot(zs,os,
xlabel=L"\tilde{z}",
label=[L"\left| c_{%$l} \right|" for _ in 1:1, l in 0:size(os,2)-1],
title = L"l=%$l₀, \tilde{g} = %$χ")