using StructuredLight
using CUDA
CUDA.allowscalar(false)
using LinearAlgebra,JLD2,QuadGK

using Plots,LaTeXStrings

default()
default(label=false,width=4, size=(700,400), markersize = 7, 
msw=0, tickfontsize=15, labelfontsize=20,
legendfontsize=12, fontfamily="Computer Modern",dpi=1000,grid=false,framestyle = :box)

magnitude(x) = floor(Int, log10(x))

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

function phase_propagation(rs,zs,l,g,free_ψs)
    f(z) = lg(rs,rs,z,l=l,k=2)
    integrand(z) = abs2.(f(z))

    Zs = vcat(0,zs)
    θs = Array{eltype(rs)}(undef,length(rs),length(rs),length(zs))

    Threads.@threads for n in eachindex(zs)
        θs[:,:,n] = quadgk(integrand,Zs[n],Zs[n+1]) |> first
    end

    θs = CuArray(θs)

    cumsum!(θs,θs,dims=3)

    f(ψ,θ) = ψ .* cis.(g*θ/4)

    stack( map(f,eachslice(free_ψs,dims=3),eachslice(θs,dims=3)) )
end

function get_Cs(rs,zs,g,l)
    GC.gc()
    ψ₀ = lg(rs,rs,l=l) |> CuArray

    free_ψs = free_propagation(ψ₀,rs,rs,zs,k=2)

    phase_δψs = (phase_propagation(rs,zs,l,g,free_ψs) - free_ψs)/g

    δψs = (kerr_propagation(ψ₀,rs,rs,zs,2048,g=g,k=2) - free_ψs)/g

    C2s = [abs(C2(z,p,l)) for z in zs, p in 0:abs(l)+2]

    Cs = similar(C2s)
    C1s = similar(C2s)

    for p in axes(Cs,2)
        ψs_proj = lg(rs,rs,zs,p=p-1,l=l,w0 = 1/√3,k=2)|> CuArray
        Cs[:,p] = overlap(ψs_proj,δψs,(rs[2]-rs[1])^2)
        C1s[:,p] = overlap(ψs_proj,phase_δψs,(rs[2]-rs[1])^2)
    end

    Cs,C1s,C2s
end

function make_plot(zs,Cs_line,Cs_scatter,pos,text,g,l)
    Z = last(zs)
    mag_x = magnitude(Z)
    mag_y = magnitude(maximum(Cs))
    colors = palette(:Set1_9)[1:l+3]'

    if iszero(mag_x)
        xlabel = L"\tilde{z}"
    else
        xlabel = L"\tilde{z} \ \ \ \left( \times \ 10^{%$(magnitude(Z))} \right)"
    end

    plot(zs,Cs_line,
        color = colors,
        xformatter = x-> x/10.0^mag_x,
        xlabel = xlabel,
        yformatter = y-> y/10.0^mag_y,
        ylabel = L"\left| c_{p%$l} \right| \ \ \ \left( \times \ 10^{%$(magnitude(maximum(Cs)))} \right)",
        annotations = (pos, Plots.text(text,18)),
        label = reshape([L"p=%$p" for p in 0:l+2],1,l+3),
        legend = :outerright,
        bottom_margin = 4Plots.mm,
        left_margin = 4Plots.mm
    )
    scatter!(zs[1:3:end],Cs_scatter[1:3:end,:],color=colors,marker=:diamond)
end
##
rs = LinRange(-15,15,512)
zs = LinRange(0,1e-2,64)
g_eff(g_l,l) = π / 2 * g_l * factorial( abs(l) ) / abs(l)^(abs(l)) / exp(-abs(l))
g_l = 800π
l = 2
g = g_eff(g_l,l)
##
Cs,C1s,C2s = get_Cs(rs,zs,g,l)
##
make_plot(zs,C1s,Cs,(.2,.85),L"g_l=800 \pi",g,l)
make_plot(zs,C2s,Cs,(.2,.85),L"g_l=800 \pi",g,l)
##
ψ₀ = lg(rs,rs,l=l) |> cu
ψs = kerr_propagation(ψ₀,rs,rs,zs,2048,g=g,k=2)
show_animation(ψs,ratio=1/4)