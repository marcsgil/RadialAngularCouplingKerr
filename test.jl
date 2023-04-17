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
rs = LinRange(-10,10,512)
ψ₀ = lg(rs,rs,0,l=21)
M = maximum(abs2,ψ₀)

visualize(ψ₀)
n = 2
ψ₁ = kerr_propagation(ψ₀,rs,rs,.01*zᵣ,512,g=400*n*π/M);

ψ = free_propagation(ψ₁,rs,rs,5,8)
visualize(ψ)
##

Ms = [maximum(x->abs(x)^2,lg(rs,rs,l=l,p=0)) for l in 0:20]
[sum(abs2,lg(rs,rs,l=l,p=0))*(rs[2]-rs[1])^2 for l in 0:20]

plot(0:20,Ms,ylabel=L"\max \ | u_{0l} | ^ 3",xlabel=L"l",xticks=0:2:20,left_margin=3Plots.mm)

Ms[1]/Ms[3]
png("Plots/power_decay2.png")