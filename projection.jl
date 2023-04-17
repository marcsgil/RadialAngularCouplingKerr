using StructuredLight
using CUDA
using LinearAlgebra,JLD2

using Plots,LaTeXStrings

default()
default(label=false,width=4,size=(600,400), markersize = 6, msw=0, 
palette=:Set1_5, tickfontsize=15, labelfontsize=18,xlabelfontsize=22,
legendfontsize=12, fontfamily="Computer Modern",dpi=1000,grid=false,framestyle = :box)
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

function get_Cs(g,l₀,Z,R)
    rs = LinRange(-R,R,1024)
    zs = LinRange(0,Z,64)
    zs_scatter = LinRange(0,Z,length(zs)÷4)

    ψ₀ = lg(rs,rs,0,l=l₀) |> CuArray
    ψs = kerr_propagation(ψ₀,rs,rs,zs_scatter,2048,g=g,k=2)


    ψ0s = free_propagation(ψ₀,rs,rs,zs_scatter,k=2)
    δψs = (ψs - ψ0s)/g
                
    Cs = stack(overlap(lg(rs,rs,zs_scatter,p=p,l=l₀,w0 = 1/√3,k=2)|> CuArray ,δψs,(rs[2]-rs[1])^2) for p in 0:abs(l₀)+2)

    C1s = [abs(C1(z,p,l₀)) for z in zs, p in 0:abs(l₀)]
    C2s = [abs(C2(z,p,l₀)) for z in zs, p in 0:abs(l₀)+2]
    
    zs,zs_scatter,Cs,C1s,C2s
end
##
zs1,zs_scatter1,Cs,C1s,C2s = get_Cs(30,2,5,15)
##
p = scatter(zs_scatter1,Cs,
    xlabel=L"\tilde{z}",
    ylabel=L"| c_{p2} \ | \ \ \ \left(  \times \ 10^{-2}  \right)",
    xformatter = x->x,
    yformatter = y->100*y,
    annotations = ((.15,.85), Plots.text(L"g=30",18)),
    marker=:diamond,
    xticks=(0:1.0:5,[L"%$x" for x in 0:1.0:5])
    )
plot!(p,zs1,C2s)
##
zs2,zs_scatter2,Ds,D1s,D2s = get_Cs(-30,2,5,15)
##
q = scatter(zs_scatter2,Ds,
    xlabel=L"\tilde{z}",
    ylabel=L"| c_{p2} \ | \ \ \ \left(  \times \ 10^{-2}  \right)",
    xformatter = x->x,
    yformatter = y->100*y,
    annotations = ((.15,.85), Plots.text(L"g=-30",18)),
    marker=:diamond,
    xticks=(0:1.0:5,[L"%$x" for x in 0:1.0:5])
)
plot!(q,zs2,D2s)
##

plot(p,q,size=(1200,400),left_margin = 10Plots.mm,bottom_margin=10Plots.mm)
##
M = 2/π
n = 2
g = -4000
l₀ = 5
zs1,zs_scatter1,Cs,C1s,C2s = get_Cs(g,l₀,.01,10)
##
default()
default(label=false,width=4,size=(600,400), markersize = 6, msw=0, 
palette= :Set1_5, tickfontsize=15, labelfontsize=18,xlabelfontsize=22,
legendfontsize=12, fontfamily="Computer Modern",dpi=1000,grid=false,framestyle = :box)
colors = palette(:Set1_9)[1:l₀+3]'
p = scatter(zs_scatter1,Cs,
    xlabel=L"\tilde{z} \ \ \ \left(  \times \ 10^{-2}  \right)",
    ylabel=L"| c_{p%$l₀} \ | \ \ \ \left(  \times \ 10^{-3}  \right)",
    xformatter = x->100*x,
    yformatter = y->1000*y,
    annotations = ((.20,.85), Plots.text(L"g \ \ = %$g",18)),
    marker=:diamond, color = colors
    )
plot!(p,zs1,C2s,color = colors, label = reshape([L"p=%$(n-1)" for n in axes(Cs,2) ],1,size(Cs,2)),
legend=:outerright)
#png("e")