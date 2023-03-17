using FreeParaxialPropagation
using KerrPropagation
using LinearAlgebra,JLD2

using Plots,LaTeXStrings

default()
default(label=false,width=3,size=(600,400), markersize = 4, msw=0, 
palette=:seaborn_bright, tickfontsize=12, labelfontsize=12,
legendfontsize=8, fontfamily="Computer Modern",dpi=1000,grid=false,framestyle = :box)
##

overlap(ψ₁,ψ₂,area_element) = area_element*abs(ψ₁ ⋅ ψ₂)

function overlap(ψ₁::AbstractArray{T1,3},ψ₂::AbstractArray{T2,3},area_element) where {T1,T2}
    map( (a,b) -> overlap(a,b,area_element), eachslice(ψ₁,dims=3), eachslice(ψ₂,dims=3) )
end
##
χ = .01
l₀ = 2
Z = .01
##
rs = LinRange(-8,8,1024)
zs = LinRange(0,Z,64)

ψ₀ = lg(rs,rs,0,l=l₀) |> CuArray
ψs = kerr_propagation(ψ₀,rs,rs,zs,2048,χ=χ,k=2)


ψ0s = free_propagation(ψ₀,rs,rs,zs,k=2)
δψs = (ψs - ψ0s)/χ
            
os = stack(overlap(lg(rs,rs,zs,p=p,l=l₀,w0 = 1/√3,k=2) |> CuArray,δψs,(rs[2]-rs[1])^2) for p in 0:abs(l₀)+3)

modified_os = hcat(ntuple(i->os[i,:]/zs[i], size(os,1) )...)'


p = plot(zs,os,
    xlabel=L"\tilde{z}\times 10^{-2}",
    ylabel=L"|c_p \ | \times 10^{-2}",
    xformatter = x->x*10^2,
    yformatter = y->y*10^2,
    label=[L"p = %$p" for _ in 1:1, p in 0:size(os,2)-1],
    title = L"l=%$l₀, \tilde{g} = %$χ")

q = plot(zs,modified_os,
    xlabel=L"\tilde{z}\times 10^{-2}",
    ylabel=L"|c_p \ | \times 10^{-2}",
    xformatter = x->x*10^2,
    yformatter = y->y*10^2,
    label=[L"p = %$p" for _ in 1:1, p in 0:size(os,2)-1],
    title = L"l=%$l₀, \tilde{g} = %$χ")
        
png(p,"Plots/l=$(l₀)_g=$(χ)_z=$(last(zs))")

png(q,"MPlots/l=$(l₀)_g=$(χ)_z=$(last(zs))")


#jldsave("Data/l=$(l₀)_g=$(χ)_z=$(last(zs)).jld2";ψs,os,zs,χ,l₀)
save("Gifs/l=$(l₀)_g=$(χ)_z=$(last(zs)).gif",FreeParaxialPropagation.animate(ψs,ratio=.5))