using FreeParaxialPropagation
using KerrPropagation
using InteractiveBeamVizualization
##
using ThreadsX, Integrals, LinearAlgebra
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

χ = 0
l = 2
##
ψs = LG(rs,rs,0,l=l) |> cu

ψs = convert(Array{ComplexF64},kerrPropagation(ψs[:,:,1],rs,rs,zs,128, χ = 1))
ψs[:,:,1]
δψs = ψs - LG(rs,rs,zs,l=l)
interactive_vizualization(δψs, rs, rs, zs)
##
approx_ψs = ThreadsX.map(r->approximatePertubation(r[1],r[2],last(zs),l,χ = χ),Iterators.product(rs,rs))

overlap(view(ψs,:,:,length(zs)) - view(ψ0s,:,:,length(zs)), approx_ψs)