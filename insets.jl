using StructuredLight

function non_linear_phase!(ψ,n)
    M = maximum(abs2,ψ)
    for j ∈ axes(ψ,1), k ∈ axes(ψ,2) 
        ψ[j,k] *= cis( π * n * abs2(ψ[j,k])/M )
    end
end

function get_images(rs,l₀,n,z₀,z,scalling)
    ψ₀ = lg(rs,rs,z₀,l = l₀)

    non_linear_phase!(ψ₀,n)
    free_propagation(ψ₀,rs,rs,z,scalling)
end

zᵣ = 1/2;

rs = LinRange(-18,18,2048)
##
#Figure 3 top
ψ = get_images(rs,0,1,0,2,1)
ψ_image = visualize(ψ)
##
#Figure 3 bottom
ψ = get_images(rs,2,1,0,2,1.5)
ψ_image = visualize(ψ)