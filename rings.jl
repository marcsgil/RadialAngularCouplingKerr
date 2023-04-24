using StructuredLight,CUDA,Plots

function non_linear_phase(ψ,n)
    M = maximum(abs2,ψ)
    [ cis( π * n * abs2(ψ[j,k])/M ) for j ∈ axes(ψ,1), k ∈ axes(ψ,2) ]
end

function get_images(rs,l₀,n,z₀,z,scalling)
    ψ₀ = lg(rs,rs,z₀,l = l₀)

    ψ₁ = ψ₀ .* non_linear_phase(ψ₀,n) |> cu
    free_propagation(ψ₁,rs,rs,z,scalling)
end

zᵣ = 1/2;

rs = LinRange(-18,18,2048)
##
ψ = get_images(rs,0,1,0,2,5)
ψ_image = visualize(ψ)
##
I = abs2.(view(ψ,size(ψ,1)÷2+1,size(ψ,2)÷2+1:size(ψ,2)))
plot(I)