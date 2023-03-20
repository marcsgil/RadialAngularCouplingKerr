using FreeParaxialPropagation,LinearAlgebra

function Λ_num(η,p,q,l)
    rs = LinRange(-5,5,2048)
    ψ₁ = lg(rs,rs,p=p,l=l)
    ψ₂ = lg(rs,rs,p=q,l=l,w0=1/√η)
    ψ₁ ⋅ ψ₂ * (rs[2]-rs[1])^2
end

Λ_num(.3,1,3,5)

function Λ(η,p,q,l)
     (√η)^(abs(l)+1) /√(prod(p+1:p+abs(l))*prod(q+1:q+abs(l))) * sum(     
       (-1)^(j+k) * binomial(p+abs(l),p-j) * binomial(q+abs(l),q-k) * η^k 
       * factorial(abs(l) + j + k)/( (η + 1)/2 )^(abs(l) + j + k+1) / ( factorial(j) * factorial(k) ) for j in 0:p, k in 0:q )
end

Λ(.3,1,3,5)

function Λ2(η,p,q,l)
    (-1)^p * (2√η/(1+η))^(abs(l)+1) /√(prod(p+1:p+abs(l))*prod(q+1:q+abs(l))) * sum(  
        (-1)^n * factorial(p+q+abs(l)-n)/( factorial(n) * factorial(p-n) * factorial(q-n) ) * ((1-η)/(1+η))^(p+q-2n)   for n in 0:p)
end
##
x = rand()
Λ(x,1,3,-5) ≈ Λ2(x,1,3,-5)
