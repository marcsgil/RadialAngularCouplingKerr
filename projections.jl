using StructuredLight, CUDA, QuadGK
using Plots,LaTeXStrings

default()
default(label=false,
fontfamily="Computer Modern",
dpi=1000,
grid=false,
framestyle = :box, 
markersize = 5, 
tickfontsize=15, 
labelfontsize=20,
legendfontsize=14,
width=4,
size=(700,400),
bottom_margin = 3Plots.mm,
left_margin = 3.5Plots.mm,)

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

function phase_propagation(rs,zs,l,g,free_ψ)
    integrand(z) = lg(rs,rs,z,l=l,k=2) .|> abs2

    Zs = vcat(0,zs)
    θs = Array{eltype(rs)}(undef,length(rs),length(rs),length(zs))

    Threads.@threads for n in eachindex(zs)
        θs[:,:,n] = quadgk(integrand,Zs[n],Zs[n+1]) |> first
    end

    θs = CuArray(θs)

    cumsum!(θs,θs,dims=3)

    f(ψ,θ) = @. ψ * cis(g*θ/4)

    stack( map(f,eachslice(free_ψ,dims=3),eachslice(θs,dims=3)) )
end

g_eff(g_l,l) = g_l * factorial( abs(l) ) / abs(l)^(abs(l)) / exp(-abs(l))

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

function calculate_projections(rs,zs,g,l,type::Symbol)
    @assert type ∈ (:A,:B,:C)

    ψ₀ = lg(rs,rs,l=l) |> CuArray

    free_ψ = free_propagation(ψ₀,rs,rs,zs,k=2)

    δψ = (kerr_propagation(ψ₀,rs,rs,zs,2048,g=g,k=2) .- free_ψ)./g

    numerical_c = Matrix{real(eltype(δψ))}(undef,(length(zs),abs(l)+3))

    if type == :A

        analytic_c = [abs(linear_c(z,p,l)) for z in zs, p in 0:abs(l)+2]

        for p in axes(numerical_c,2)
            corrected_ψ = lg(rs,rs,zs,p=p-1,l=l,w0 = 1/√3,k=2)|> CuArray
            numerical_c[:,p] = overlap(corrected_ψ,δψ,rs,rs) .|> abs
        end

    elseif type == :B

        analytic_c = [abs(nonlinear_c(z,p,l)) for z in zs, p in 0:abs(l)+2]

        for p in axes(numerical_c,2)
            corrected_ψ = lg(rs,rs,zs,p=p-1,l=l,w0 = 1/√3,k=2)|> CuArray
            numerical_c[:,p] = overlap(corrected_ψ,δψ,rs,rs) .|> abs
        end

    elseif type == :C

        phase_δψ = (phase_propagation(rs,zs,l,g,free_ψ) .- free_ψ)./g

        analytic_c = similar(numerical_c)
        
        for p in axes(numerical_c,2)
            corrected_ψ = lg(rs,rs,zs,p=p-1,l=l,w0 = 1/√3,k=2)|> CuArray
            numerical_c[:,p] = overlap(corrected_ψ,δψ,rs,rs) .|> abs
            analytic_c[:,p] = overlap(corrected_ψ,phase_δψ,rs,rs) .|> abs
        end

    end

    analytic_c,numerical_c
end
##
#Figure 1 (a)
rs = LinRange(-5,5,1024)
zs = LinRange(0,0.1,64)
g = 0.01
l = 2
analytic_c,numerical_c = calculate_projections(rs,zs,g,l,:A)

p1 = make_plot(zs,analytic_c,numerical_c,(.15,.85),L"(a)",l)
png(p1, "Plots/png/physical_case_1")
Plots.svg(p1, "Plots/svg/physical_case_1")
##
#Figure 1 (b)
rs = LinRange(-15,15,1024)
zs = LinRange(0,5,64)
g = 0.01
l = 2
analytic_c,numerical_c = calculate_projections(rs,zs,g,l,:B)

p2 = make_plot(zs,analytic_c,numerical_c,(.15,.85),L"(b)",l)

png(p2, "Plots/png/physical_case_2")
Plots.svg(p2, "Plots/svg/physical_case_2")
##
#Figure 2 (a)
rs = LinRange(-15,15,1024)
zs = LinRange(0,5,64)
g = 30
l = 2
analytic_c,numerical_c = calculate_projections(rs,zs,g,l,:B)

p3 = make_plot(zs,analytic_c,numerical_c,(.15,.85),L"g = %$g",l)

png(p3,"Plots/png/extreme_case_1")
Plots.svg(p3,"Plots/svg/extreme_case_1")
##
#Figure 2 bottom
rs = LinRange(-15,15,1024)
zs = LinRange(0,5,64)
g = -30
l = 2
analytic_c,numerical_c = calculate_projections(rs,zs,g,l,:B)

p4 = make_plot(zs,analytic_c,numerical_c,(.15,.85),L"g = %$g",l)
png(p4, "Plots/png/extreme_case_2")
Plots.svg("Plots/svg/extreme_case_2")
##
#Experiment l = 0
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

analytic_c,numerical_c = calculate_projections(rs,zs,g,l,:C)

p5 = make_plot(zs,analytic_c,numerical_c,(.20,.85),L"g_l = %$g_l",l)
vline!([z/zᵣ],line=(:dash,:black))
png(p5, "Plots/png/kaiser_l=0")
Plots.svg(p5, "Plots/svg/kaiser_l=0")
##
#Experiment l = 2
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
l = 2
g = g_eff(g_l,l)

analytic_c,numerical_c = calculate_projections(rs,zs,g,l,:C)

p6 = make_plot(zs,analytic_c,numerical_c,(.20,.85),L"g_l = %$g_l",l)
vline!([z/zᵣ],line=(:dash,:black))
png(p6, "Plots/png/kaiser_l=2")
Plots.svg(p6, "Plots/svg/kaiser_l=2")