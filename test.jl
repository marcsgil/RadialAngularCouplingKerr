using FreeParaxialPropagation,KerrPropagation,CUDA


λ = 780e-9
k = 2π/λ
w0 = 1.6e-3
zr = k * w0^2 / 2
g = -1000

rs = LinRange(-8*w0,8*w0,1024)
zs = LinRange(0,7e-2,2)
ψ₀ = lg(rs,rs,0,w0=w0,l=2) |> CuArray

ψs = kerr_propagation(ψ₀,rs,rs,zs,2048,g=g,k=k)
ψ0s = free_propagation(ψ₀,rs,rs,zs,k=k)
δψs = (ψs - ψ0s)

vizualize((@view δψs[:,:,end]))
##
far_ψ = free_propagation((@view δψs[:,:,end]),rs,rs,1.8*zr,k=k)

vizualize(far_ψ)