using FreeParaxialPropagation,KerrPropagation,CUDA,Tullio
##
N = 2048
rs = LinRange(-20,20,N)

function get_beam(rs,n,m)
    ψ₀ = lg(rs,rs,0)
    @tullio ψ₀[j,k] = ψ₀[j,k] * cis(- 2π*n * exp( -m*( rs[j]^2 + rs[k]^2 )) )
end
##
ψ₀ = get_beam(rs,1,2) |> cu
vizualize(ψ₀)
ψ = free_propagation(ψ₀,rs,rs,3,k=2)
vizualize(ψ)
##
g(n) = 800π * n
gaussian = lg(rs,rs,0)|> cu
ψ₀ = kerr_propagation(gaussian,rs,rs,LinRange(0,.01,4),2048,k=2,g=-g(1))[:,:,end]
vizualize(ψ₀)
ψ = free_propagation(ψ₀,rs,rs,LinRange(0,4,64),k=2)
animate(ψ)
##