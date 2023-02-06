function free_propagation_step!(ψ,phases,plan,iplan)
    plan*ψ
    map!(*,ψ,ψ,phases)
    iplan*ψ
    nothing
end

phase_evolution(ψ,factor) = cis(abs2(ψ)*factor)*ψ

function nls_propagation_step!(ψ,phases,plan,iplan,factor)
    map!(ψ->phase_evolution(ψ,factor/2),ψ,ψ)
    free_propagation_step!(ψ,phases,plan,iplan)
    map!(ψ->phase_evolution(ψ,factor/2),ψ,ψ)
end

function kerrPropagation(ψ₀,xs,ys,zs,subdivisions::Integer;k=1,χ=1)
    #solves 2ik ∂_z ψ = - ∇² ψ - χ |ψ|² ψ with initial condition ψ₀

    #@assert iszero(first(zs))

    steps = length(zs)*subdivisions
    Δz = last(zs)/steps

    phase_evolution_factor = χ*Δz/(2k)
    
    free_prop_factor = -Δz/(2k)
    phases = map(ks -> cis(free_prop_factor*sum(abs2,ks)), Iterators.product(reciprocal_grid(xs),reciprocal_grid(ys))) |> ifftshift

    if typeof(ψ₀) <: CuArray
        phases = cu(phases)
    end

    plan = plan_fft!(ψ₀)
    iplan = plan_ifft!(ψ₀)

    results = similar(ψ₀,size(ψ₀)...,length(zs))
    results[:,:,1] = ψ₀

    prev_ψ = ifftshift(ψ₀)
    
    for n in 2:steps
        nls_propagation_step!(prev_ψ,phases,plan,iplan,phase_evolution_factor)

        if n % subdivisions == 0
            fftshift!(view(results,ntuple(i->:,ndims(ψ₀))..., n ÷ subdivisions), prev_ψ)
        end
    end

    
    results
end