using FreeParaxialPropagation
using KerrPropagation
##
using ThreadsX,Integrals, LinearAlgebra
kerrCube(x) = abs2(x)*x

function approximatePertubation(x,y,z,l;k=1,χ=1)
    f(z,p) = (im*k*χ/2) * kerrCube(LG(x,y,z,l=l))
    prob = IntegralProblem(f,0,z)
    solve(prob,QuadGKJL()).u
end

function overlap(ψ₁,ψ₂)
    sqrt( abs2( ψ₁ ⋅ ψ₂ )/( sum(abs2,ψ₁) * sum(abs2,ψ₂) ) )
end
##
rs = LinRange(-5,5,1024)
zs = LinRange(0,.5,32)

χ = 1e-6
l = 2
##
ψ₀ = LG(rs,rs,0,l=l) |> cu

ψs = convert(Array{ComplexF64},kerrPropagation(ψ₀,rs,rs,zs,128, χ = χ))
δψs = ψs - LG(rs,rs,zs,l=l)
##
approx_ψs = ThreadsX.map(r->approximatePertubation(r[1],r[2],last(zs),l,χ = χ),Iterators.product(rs,rs))


overlap(view(ψs,:,:,length(zs)) - view(ψ0s,:,:,length(zs)), approx_ψs)
##
using InteractiveBeamVizualization
interactive_vizualization(δψs, rs, rs, zs)
##