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

function C3(z,p,l)
    im * z * c(p,l) / 4
end

function C2(z,p,l)
    sum( c(q,l) * Φ(z,r) * Λ(3,r,p,l) * Λ(3,r,q,l) for q in 0:abs(l), r in 0:10^3 ) * im / 4
end

function phase_propagation(rs,zs,l,g,free_ψs)
    integrand(z) = lg(rs,rs,z,l=l,k=2) .|> abs2

    Zs = vcat(0,zs)
    θs = Array{eltype(rs)}(undef,length(rs),length(rs),length(zs))

    Threads.@threads for n in eachindex(zs)
        θs[:,:,n] = quadgk(integrand,Zs[n],Zs[n+1]) |> first
    end

    θs = CuArray(θs)

    cumsum!(θs,θs,dims=3)

    f(ψ,θ) = @. ψ * cis(g*θ/4)

    stack( map(f,eachslice(free_ψs,dims=3),eachslice(θs,dims=3)) )
end

function get_Cs(rs,zs,g,l)
    ψ₀ = lg(rs,rs,l=l) |> CuArray

    free_ψs = free_propagation(ψ₀,rs,rs,zs,k=2)

    #phase_δψs = (phase_propagation(rs,zs,l,g,free_ψs) - free_ψs)/g

    δψs = (kerr_propagation(ψ₀,rs,rs,zs,2048,g=g,k=2) - free_ψs)/g

    C2s = [abs(C2(z,p,l)) for z in zs, p in 0:abs(l)+2]
    C3s = [abs(C3(z,p,l)) for z in zs, p in 0:abs(l)+2]

    Cs = similar(C2s)
    C1s = similar(C2s)

    for p in axes(Cs,2)
        ψs_proj = lg(rs,rs,zs,p=p-1,l=l,w0 = 1/√3,k=2)|> CuArray
        Cs[:,p] = overlap(ψs_proj,δψs,rs,rs) .|> abs
        #C1s[:,p] = overlap(ψs_proj,phase_δψs,(rs[2]-rs[1])^2)
    end

    Cs,C1s,C2s,C3s
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
        ylabel = L"\left| c_{p%$l} \right| \ \ \ \left( \ 10^{%$(magnitude(maximum(Cs)))} \right)",
        annotations = (pos, Plots.text(text,18)),
        label = reshape([L"p=%$p" for p in 0:l+2],1,l+3),
        legend = :outerright,
        bottom_margin = 4Plots.mm,
        left_margin = 4Plots.mm,
        xticks = 0:Z/5:Z,
        legendfontsize=16,
        size=(700,400)
    )
    scatter!(zs[1:3:end],Cs_scatter[1:3:end,:],color=colors,marker=:diamond)
end

g_eff(g_l,l) = g_l * factorial( abs(l) ) / abs(l)^(abs(l)) / exp(-abs(l))
##
rs = LinRange(-15,15,1024)
zs = LinRange(0,0.1,64)

g = 0.01
l = 2
##
Cs,C1s,C2s,C3s = get_Cs(rs,zs,g,l)
##
make_plot(zs,C3s,Cs,(.15,.85),L"(a)",g,l)
png("Plots/phase_g_l=$g_l,l=$l")
make_plot(zs,C2s,Cs,(.2,.85),L"g_l=%$1",g,l)
png("Plots/analytic_g_l=$g_l,l=$l")
##
#Experiment
w0 = 1.7e-3
λ = 780e-9
k = 2π/λ
zᵣ = k * w0^2 / 2
z = 7e-2
z/zᵣ
Δϕ = π
zs = LinRange(0,1e-2,64)
rs = LinRange(-4,4,512)
g_l = round(-2π * Δϕ / (z/zᵣ), sigdigits=2) |> Int
l = 0
g = g_eff(g_l,l)
##
Cs,C1s,C2s = get_Cs(rs,zs,g,l)
##
make_plot(zs,C1s,Cs,(.2,.85),L"g_l=%$g_l",g,l)
vline!([z/zᵣ],line=(:dash,:black))
plot!(ψ_image,
    xticks=false,
    yticks=false,
    framestyle=:none,
    inset=bbox(0.7, 0.2, 0.3, 0.3), subplot=2)

##
png("Plots/phase_g_l=$g_l,l=$l")
make_plot(zs,C2s,Cs,(.2,.85),L"g_l=%$g_l",g,l)
png("Plots/analytic_g_l=$g_l,l=$l")
##
ψ₀ = lg(rs,rs,l=l) |> CuArray
ψ = kerr_propagation(ψ₀,rs,rs,zs,2048,g=g,k=2)

show_animation(ψ)
##
plot( 1:5 )
plot!( -5:8, (-5:8).^2, inset = (1, bbox(0.1,0.0,0.4,0.4)), subplot = 2)