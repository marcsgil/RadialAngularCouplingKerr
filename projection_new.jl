using StructuredLight, CUDA, QuadGK
using Plots,LaTeXStrings

default()
default(label=false,
fontfamily="Computer Modern",
dpi=1000,
grid=false,
framestyle = :box, 
markersize = 6, 
tickfontsize=15, 
labelfontsize=20,
legendfontsize=16,
width=5,
size=(700,400),
bottom_margin = 4Plots.mm,
left_margin = 4Plots.mm,)

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

function C(p,l)
    L = abs(l)
    2 * (-1)^p * binomial(2L,L-p) * √binomial(L+p,p) / ( π * 3^((3L+1)/2) )
end

function Φ(z,p)
    iszero(p) ? atan(z) : ( 1- ( (1-im*z)/(1+im*z) )^p ) / (2im*p)
end

function nonlinear_c(z,p,l)
    sum( C(q,l) * Φ(z,r) * Λ(3,r,p,l) * Λ(3,r,q,l) for q in 0:abs(l), r in 0:10^3 ) * im / 4
end

function linear_c(z,p,l)
    im * z * C(p,l) / 4
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

magnitude(x) = floor(Int, log10(x))

function make_plot(zs,cs_line,cs_scatter,pos,text,l)
    Z = last(zs)
    mag_x = magnitude(Z)
    mag_y = magnitude(max(maximum(cs_line),maximum(cs_scatter)))
    colors = palette(:Set1_9)[1:l+3]'

    if iszero(mag_x)
        xlabel = L"\tilde{z}"
    elseif mag_x == -1
        xlabel = L"10 \tilde{z}"
    else
        xlabel = L"10^{%$(-mag_x)} \tilde{z}"
    end

    if iszero(mag_y)
        ylabel = L"\left| c_{p%$l} \right|"
    elseif mag_y == -1
        ylabel = L"10 \left| c_{p%$l} \right|"
    else
        ylabel = L"10^{%$(-mag_y)} \left| c_{p%$l} \right|"
    end

    plot(zs,cs_line,
        color = colors,
        xformatter = x-> x/10.0^mag_x,
        xticks = 0:Z/5:Z,
        yformatter = y-> y/10.0^mag_y,
        annotations = (pos, Plots.text(text,18)),
        lines = [:solid :dash :dot :dashdot :dashdotdot],
        ; xlabel,ylabel
    )
    scatter!(zs[1:3:end],cs_scatter[1:3:end,:],color=colors,
    label = reshape([L"p=%$p" for p in 0:l+2],1,l+3),
    legend = :outerright,
    marker = [ :circ :utriangle :diamond :dtriangle :hexagon ])
end
##
#Figure 1 (a)
GC.gc()
rs = LinRange(-5,5,1024)
zs = LinRange(0,0.1,64)
g = 0.01
l = 2
ψ₀ = lg(rs,rs,l=l) |> CuArray

free_ψs = free_propagation(ψ₀,rs,rs,zs,k=2)

δψs = (kerr_propagation(ψ₀,rs,rs,zs,2048,g=g,k=2) - free_ψs)/g

lc = [abs(linear_c(z,p,l)) for z in zs, p in 0:abs(l)+2]

num_c = similar(lc)

for p in axes(num_c,2)
    ψs_proj = lg(rs,rs,zs,p=p-1,l=l,w0 = 1/√3,k=2)|> CuArray
    num_c[:,p] = overlap(ψs_proj,δψs,rs,rs) .|> abs
end

make_plot(zs,lc,num_c,(.15,.85),L"(a)",l)
png("physical_case_1")
##
#Figure 1 (b)
GC.gc()
rs = LinRange(-15,15,1024)
zs = LinRange(0,5,64)
g = 0.01
l = 2
ψ₀ = lg(rs,rs,l=l) |> CuArray

free_ψs = free_propagation(ψ₀,rs,rs,zs,k=2)

δψs = (kerr_propagation(ψ₀,rs,rs,zs,2048,g=g,k=2) - free_ψs)/g

nlc = [abs(nonlinear_c(z,p,l)) for z in zs, p in 0:abs(l)+2]

num_c = similar(nlc)

for p in axes(num_c,2)
    ψs_proj = lg(rs,rs,zs,p=p-1,l=l,w0 = 1/√3,k=2)|> CuArray
    num_c[:,p] = overlap(ψs_proj,δψs,rs,rs) .|> abs
end

make_plot(zs,nlc,num_c,(.15,.85),L"(b)",l)
png("physical_case_2")
##
#Figure 2 (a)
GC.gc()
rs = LinRange(-15,15,1024)
zs = LinRange(0,5,64)
g = 30
l = 2
ψ₀ = lg(rs,rs,l=l) |> CuArray

free_ψs = free_propagation(ψ₀,rs,rs,zs,k=2)

δψs = (kerr_propagation(ψ₀,rs,rs,zs,2048,g=g,k=2) - free_ψs)/g

nlc = [abs(nonlinear_c(z,p,l)) for z in zs, p in 0:abs(l)+2]

num_c = similar(nlc)

for p in axes(num_c,2)
    ψs_proj = lg(rs,rs,zs,p=p-1,l=l,w0 = 1/√3,k=2)|> CuArray
    num_c[:,p] = overlap(ψs_proj,δψs,rs,rs) .|> abs
end

make_plot(zs,nlc,num_c,(.15,.85),L"g = %$g",l)
png("extreme_case_1")
##
#Figure 2 (b)
GC.gc()
rs = LinRange(-15,15,1024)
zs = LinRange(0,5,64)
g = -30
l = 2
ψ₀ = lg(rs,rs,l=l) |> CuArray

free_ψs = free_propagation(ψ₀,rs,rs,zs,k=2)

δψs = (kerr_propagation(ψ₀,rs,rs,zs,2048,g=g,k=2) - free_ψs)/g

nlc = [abs(nonlinear_c(z,p,l)) for z in zs, p in 0:abs(l)+2]

num_c = similar(nlc)

for p in axes(num_c,2)
    ψs_proj = lg(rs,rs,zs,p=p-1,l=l,w0 = 1/√3,k=2)|> CuArray
    num_c[:,p] = overlap(ψs_proj,δψs,rs,rs) .|> abs
end

make_plot(zs,nlc,num_c,(.15,.85),L"g = %$g",l)
png("extreme_case_2")