using FreeParaxialPropagation
using CUDA
using KerrPropagation
using LinearAlgebra,JLD2

using Plots,LaTeXStrings

default()
default(label=false,width=3,size=(600,400), markersize = 4.5, msw=0, 
palette=:Set1_5, tickfontsize=10, labelfontsize=12,
legendfontsize=10, fontfamily="Computer Modern",dpi=1000,grid=false,framestyle = :box)
##
overlap(ψ₁,ψ₂,area_element) = area_element*abs(ψ₁ ⋅ ψ₂)

function overlap(ψ₁::AbstractArray{T1,3},ψ₂::AbstractArray{T2,3},area_element) where {T1,T2}
    map( (a,b) -> overlap(a,b,area_element), eachslice(ψ₁,dims=3), eachslice(ψ₂,dims=3) )
end

function Λ(η,p,q,l)
    L = abs(l)
    if q ≥ p
        (-1)^p * (2√η/(1+η))^(L+1) * √( prod(q+1:q+L)/prod(p+1:p+L) ) * 
        sum(  (-1)^n * binomial(q,n) * binomial(p+q+L-n,q+L) 
        * ((1-η)/(1+η))^(p+q-2n) for n in 0:p)
    else
        Λ(1/η,q,p,l)
    end    
end

function c(p,l)
    L = abs(l)
    2 * (-1)^p * binomial(2L,L-p) * √binomial(L+p,p) / ( π * 3^((3L+1)/2) )
end

function Φ(z,p)
    iszero(p) ? atan(z) : ( 1- ( (1-im*z)/(1+im*z) )^p ) / (2im*p)
end

function C1(z,p,l)
    im * z * c(p,l) / 4
end

function C2(z,p,l)
    sum( c(q,l) * Φ(z,r) * Λ(3,r,p,l) * Λ(3,r,q,l) for q in 0:abs(l), r in 0:10^3 ) * im / 4
end
##
g = .1
l₀ = 2
Z = .01
##
rs = LinRange(-5,5,1024)
zs = LinRange(0,Z,64)
zs_scatter = LinRange(0,Z,length(zs)÷4)

ψ₀ = lg(rs,rs,0,l=l₀) |> CuArray
ψs = kerr_propagation(ψ₀,rs,rs,zs_scatter,2048,g=g,k=2)
FreeParaxialPropagation.animate(ψs)


ψ0s = free_propagation(ψ₀,rs,rs,zs_scatter,k=2)
δψs = (ψs - ψ0s)/g
            
Cs = stack(overlap(lg(rs,rs,zs_scatter,p=p,l=l₀,w0 = 1/√3,k=2)|> CuArray ,δψs,(rs[2]-rs[1])^2) for p in 0:abs(l₀)+2)

C1s = [abs(C1(z,p,l₀)) for z in zs, p in 0:abs(l₀)]
C2s = [abs(C2(z,p,l₀)) for z in zs, p in 0:abs(l₀)+2]
##
p = scatter(zs_scatter,Cs,
    xlabel=L"\tilde{z}",
    ylabel=L"| c_{p%$l₀} \ | \ \ \ \left( 10^{-2} \ \right)",
    xformatter = x->x,
    yformatter = y->100*y,
    annotations = ((.15,.85), Plots.text(L"g=%$g",15)),
    marker=:diamond
    )
plot!(p,zs,C1s)
##

##
q = scatter(zs_scatter,Cs,
    xlabel=L"\tilde{z}",
    ylabel=L"| c_{p%$l₀} \ | \ \ \ \left( 10^{-2} \ \right)",
    xformatter = x->x,
    yformatter = y->100*y,
    annotations = ((.15,.85), Plots.text(L"g=%$g",15)),
    marker=:diamond
    )
plot!(q,zs,C1s)
##

        
#png(p,"Plots/l=$(l₀)_g=$(g)_z=$(last(zs))")

#png(q,"MPlots/l=$(l₀)_g=$(g)_z=$(last(zs))")


jldsave("Data/l=$(l₀)_g=$(g)_z=$Z.jld2";ψs,Cs,C1s,C2s,zs,zs_scatter,g,l₀)
save("Gifs/l=$(l₀)_g=$(g)_z=$Z.gif",FreeParaxialPropagation.animate(ψs,ratio=.5))
##

function make_plot(zs,zs_scatter,Cs,Ds,g,show_tick_legend,isfirst)
    p = plot(zs,Cs,
    ylabel=L"100 \times |C_{p%$l₀} \ |/g ",
    yformatter = y->100*y,
    annotations = ((.15,.85), Plots.text(L"g=%$g",12)),size=(354,200),left_margin=4Plots.mm)

    if isfirst
        plot!(p,title="(b)")
    end

    if show_tick_legend
        plot!(xlabel=L"Z",xformatter = x->x,bottom_margin=-3Plots.mm)
    else
        plot!(p,xformatter=_->"",bottom_margin=-8.1Plots.mm)
    end

    scatter!(p,zs_scatter,Ds,line=:dash,marker=:diamond)
end

ps = []

for g ∈ (1,10,80)
    _load(name) = load("Data/l=2_g=$(g)_z=8.jld2",name)
    push!(ps,make_plot(_load("zs"),_load("zs_scatter"),
    _load("Cs"),_load("C2s"),g,length(ps)==2,length(ps)==0))
end

p2 = plot(ps[1],ps[2],ps[3],layout=(3,1),size=(400,800))
#png("Plots/big_plot.png")

plot(p1,p2,layout=(1,2),size=(800,800))
