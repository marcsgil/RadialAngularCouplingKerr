using StructuredLight,QuadGK,CUDA,LinearAlgebra

using Plots,LaTeXStrings
##
default()
default(label=false,width=4,size=(600,400), markersize = 6, msw=0, 
palette=:Set1_5, tickfontsize=15, labelfontsize=18,xlabelfontsize=22,
legendfontsize=12, fontfamily="Computer Modern",dpi=1000,grid=false,framestyle = :box)
##
rs = LinRange(-10,10,1024)
zs = LinRange(0,5,32)
l=0
##
function phase_approach(rs,l,z,g)
    f(z) = lg(rs,rs,z,l=l,k=2)
    integrand(z) = abs2.(f(z))

    Z = vcat(0,z)
    θs = Array{eltype(rs)}(undef,length(rs),length(rs),length(zs))

    for n in eachindex(z)
        θs[:,:,n] = quadgk(z->abs2.(f(z)),Z[n],Z[n+1]) |> first
    end

    
    [f(zs[n]) .* cis.(g*view(θs,:,:,n)/4) for n in eachindex(zs)] |> stack
end

ψ_phase = phase_approach(rs,l,zs,1) |> cu
ψ = kerr_propagation(ψ_phase[:,:,1]|> cu, rs,rs,zs,2048,k=2)
ψ_free = lg(rs,rs,zs,l=l) |> cu

#interactive_visualization(ψ)
##
function my_max(ψ,rs)
    M = (rs[2]-rs[1])*√sum(abs2,ψ)
    iszero(M) ? 1 : M
end

δψ = map(ψ->ψ/my_max(ψ,rs),eachslice(ψ-ψ_free,dims=3)) |> stack
δψ_phase = map(ψ->ψ/my_max(ψ,rs),eachslice(ψ_phase-ψ_free,dims=3)) |> stack

overlap(δψ,δψ_phase,(rs[2]-rs[1])^2)
#overlap(ψ,ψ_free,(rs[2]-rs[1])^2)