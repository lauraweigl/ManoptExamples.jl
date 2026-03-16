### A Pluto.jl notebook ###
# v0.20.20

using Markdown
using InteractiveUtils

# ╔═╡ 3d0776f8-1bac-11f1-237d-cd2b0f404020
using Pkg; Pkg.activate();

# ╔═╡ fe7d4e51-8c93-467c-99e9-c546900775b8
begin
	using LinearAlgebra
	using SparseArrays
	using Manopt
	using ManoptExamples
	using Manifolds
	using OffsetArrays
	using RecursiveArrayTools
    using WGLMakie, Makie, GeometryTypes, Colors
	using CairoMakie
end;

# ╔═╡ 683a3a9d-64f8-49f8-9d4c-fa86b3d08335
begin
	N=200
	max_iter = 50
	max_iter_newton = 20

	α = 1.0
	h_ref = 0.1
	
	S = Manifolds.Sphere(2)
	R = Manifolds.Euclidean(1)
	powerS = PowerManifold(S, NestedPowerRepresentation(), N) # power manifold of S
	powerR = PowerManifold(R, NestedPowerRepresentation(), N)
	
	mutable struct variational_space
		manifold::AbstractManifold
		degree::Integer
	end

	test_spaces = variational_space(S, 1)

	ansatz_spaces = variational_space(S, 1)
	
	start_interval = -pi/2 + 0.1
	end_interval = pi/2 - 0.1
	discrete_time = range(; start=start_interval, stop = end_interval, length=N+2) # equidistant discrete time points
	
	theta = pi/4
	y0 = [sin(theta)*cos(start_interval),sin(theta)*sin(start_interval),cos(theta)] # startpoint of geodesic
	yT = [sin(theta)*cos(end_interval),sin(theta)*sin(end_interval),cos(theta)] # endpoint of geodesic
end;

# ╔═╡ 18a99268-7d83-483a-b279-ab7185850094
mutable struct DifferentiableMapping{F1<:Function,F2<:Function,T}
	value::F1
	derivative::F2
	h_ref::T
	proximity_parameter::T
end;

# ╔═╡ c54635ae-2c08-4bf6-b536-1b50ec3489db
mutable struct DifferentiableRetraction{F1<:Function,F2<:Function,F3<:Function}
	value::F1
	derivative::F2
	second_derivative::F3
end;

# ╔═╡ 0cb7b735-2118-440d-aa24-9da29192b1c7
begin
	retraction_pointwise_at(x, ξ) = (x+ξ)/norm(x+ξ)
	retraction_derivative_at(x, ξ, ϕ) = ϕ/norm(x+ξ) - ((x+ξ)*(x+ξ)'*ϕ)/norm(x+ξ)^3
	retraction_second_derivative_at(x, ξ, δξ, ϕ) = -((x+ξ)'*δξ/norm(x+ξ)^3)*ϕ + 3*(((x+ξ)'*δξ*(x+ξ)'*ϕ)/norm(x+ξ)^5)*(x+ξ) - 1/norm(x+ξ)^3*(((x+ξ)'*ϕ)*δξ + ((δξ'*ϕ)*(x+ξ)))

	retraction = DifferentiableRetraction(retraction_pointwise_at, retraction_derivative_at, retraction_second_derivative_at)
end;

# ╔═╡ 2d8fdd96-fc60-40b7-9517-141cae486cf5
begin
	function legendre_derivative(y,h_ref) 
		return -log(1-h_ref-y)
	end
	legendre_derivative_star(y,h_ref) = 1-h_ref - exp(-y)
	legendre_derivative_star_prime(y) = exp(-y)
end;

# ╔═╡ 5ccea4c1-a34c-4d72-b095-5cc3b849506e
begin
	function F_at(Integrand, x, retr_dot, retr_prime_dot, T)
	  	return Integrand.proximity_parameter * retr_prime_dot'*retr_dot
	end

	function F_rest_at(Integrand, x, ξ, ψ, T)
		return ψ[1]*T[3] - legendre_derivative(x[3],Integrand.h_ref)*T[3]
	end

	function F_ψ_at(Integrand, x, ξ, ψ, T)
		return (x[3] + ξ[3])*T[1] - legendre_derivative_star(ψ[1], Integrand.h_ref)*T[1]
	end

	function F12_at(Integrand, x, ξ, ψ, B, T)
		return B[1]*T[3]
	end

	function F22_at(Integrand, x, ξ, ψ, B, T)
		return legendre_derivative_star_prime(ψ[1])*B[1]*T[1]
	end
	
	function F_prime_at(Integrand,x,retr_dot,retr_prime_dotTF,retr_prime_dotAF,retr_doubleprime_dot)
		return Integrand.proximity_parameter * (retr_doubleprime_dot'*retr_dot + retr_prime_dotTF'*retr_prime_dotAF)
	end

	integrand_F = DifferentiableMapping(F_at, F_prime_at, h_ref, α)
	integrand_F_rest = DifferentiableMapping(F_rest_at, F_prime_at, h_ref, α)
	integrand_ξψ = DifferentiableMapping(F_ψ_at, F12_at, h_ref, α)
	integrand_ψψ = DifferentiableMapping(F_ψ_at, F22_at, h_ref, α)

end;

# ╔═╡ 362cb1d2-eda2-4e1b-a995-c10f68e74e34
function evaluate(y, i, tloc)
	return (1.0-tloc)*y[i-1]+tloc*y[i]
end;

# ╔═╡ f87800a1-55d5-4994-a9d5-c248cf9fc78e
function assemble_Newton_matrix(M, x, ξ, ψ, integrand_F, integrand_Frest, integrand_ξψ, integrand_ψψ, time_interval, retraction, testspace, eval)
	
	Ox = OffsetArray([y0, x..., yT], 0:(length(x)+1))
	Oξ = OffsetArray([zeros(3), ξ..., zeros(3)], 0:(length(ξ)+1))
	Oψ = OffsetArray([legendre_derivative(y0[3], 0.1), ψ..., legendre_derivative(yT[3], 0.1)], 0:(length(ψ)+1))
	
	A11 = spzeros(Int(manifold_dimension(M)), Int(manifold_dimension(M)))

	ManoptExamples.get_jacobian!(M, Ox, Oξ, eval, A11, integrand_F, time_interval, retraction; row_index = 1, column_index = 1, test_space = testspace, ansatz_space = testspace)
	
	#println(Matrix(A11))
	
	A12 = spzeros(2*N, N)

	ManoptExamples.get_jacobian_ψ!(M, Ox, Oξ, Oψ, eval, A12, integrand_ξψ, time_interval; row_index = 1, column_index = 2, test_space = testspace, ansatz_space = variational_space(Manifolds.Euclidean(1), 1))
	
	A22 = zeros(N,N)

	ManoptExamples.get_jacobian_ψ!(M, Ox, Oξ, Oψ, eval, A22, integrand_ψψ, time_interval; row_index = 2, column_index = 2, test_space = variational_space(Manifolds.Euclidean(1), 1), ansatz_space = variational_space(Manifolds.Euclidean(1), 1))

	return vcat(hcat(A11 , A12), 
			  hcat(A12', -A22))
end

# ╔═╡ e71b9f35-6d84-46fe-813c-70bb95d94c58
function assemble_rhs(M, x, ξ, ψ, integrand_F, integrand_Frest, integrand_ψ, time_interval, retraction, testspace, eval)

		b = zeros(Int(manifold_dimension(M)))

		Ox = OffsetArray([y0, x..., yT], 0:(length(x)+1))
		Oξ = OffsetArray([zeros(3), ξ..., zeros(3)], 0:(length(ξ)+1))
		Oψ = OffsetArray([legendre_derivative(y0[3], 0.1), ψ..., legendre_derivative(yT[3], 0.1)], 0:length(ψ)+1)
	
		ManoptExamples.get_right_hand_side!(M, Ox, Oξ, eval, b, integrand_F, time_interval, retraction; test_space = testspace)

		bψ = zeros(Int(manifold_dimension(M)))
		
		ManoptExamples.get_right_hand_side!(M, Ox, Oξ, Oψ, eval, bψ, integrand_Frest, time_interval, retraction; test_space = testspace)
	
		b = b .+ bψ
		
		nb = zeros(length(ψ))

		ManoptExamples.get_right_hand_side_ψ!(M, Ox, Oξ, Oψ, eval, nb, integrand_ψ, time_interval; test_space = variational_space(Manifolds.Euclidean(1),1))
		

		return vcat(b, nb)
end;

# ╔═╡ 45d6a596-88c1-4ec4-8de1-a657977e857d
begin
function solve_SPP(M, x, ξ, ψ, integrand_F, integrand_Frest, integrand_ξψ, integrand_ψψ, time, retraction, testspace, eval)
	ξ_start = copy(ξ)
	ψ_start = copy(ψ)
	
	ξ_res = copy(ξ)
	
	for i in 1:max_iter_newton
		
		A = assemble_Newton_matrix(M, x, ξ_res, ψ, integrand_F, integrand_Frest, integrand_ξψ, integrand_ψψ, time, retraction, testspace, eval)
		
		rhs = assemble_rhs(M, x, ξ_res, ψ, integrand_F, integrand_Frest, integrand_ξψ, time, retraction, testspace, eval)
		#rhs2 = Any[]
		#τ = 0.001
		#for k in 1:length(ψ)
			#ψ_2 = copy(ψ)
			#ψ_2[k] += τ
			#push!(rhs2, (assemble_rhs(M, x, ξ_res, ψ_2, integrand_F, integrand_Frest, integrand_ξψ, time, retraction, testspace, eval)-rhs)/τ)
		#end
		#AA = reduce(hcat, rhs2)
		#println("")
		#println("Finite Differenzen: ", AA)
		
		X = A \ (-rhs)

		directionξ = get_vector(powerS, x, X[1:2*N], DefaultOrthogonalBasis())
		directionψ = get_vector(powerR, ψ, X[2*N+1:end], DefaultOrthogonalBasis())
	
		if norm(directionξ) < 1e-11
			println("Newton konvergiert")
			return ξ_res
		end

		if i == 2
			θ = norm([directionξ..., directionψ...])/norm([(ξ_res-ξ_start)..., (ψ-ψ_start)...])
			println("θ =", θ)
		
			if θ < 0.1
				integrand_F.proximity_parameter *= 0.1/θ
			elseif θ > 0.9
				integrand_F.proximity_parameter *= 0.5
				return ξ_start
			else
				integrand_F.proximity_parameter *= 1.0
			end
		end
		
		ξ_res .= ξ_res .+ directionξ
		retract!(powerR, ψ, ψ, directionψ, ExponentialRetraction())
	end
	println("Newton konvergiert nicht")
	return ξ_start
	
end
end

# ╔═╡ c5c6c894-b975-4ff2-b411-46bf52049e67
begin
	y(t) = [sin(theta)*cos(t), sin(theta)*sin(t), cos(theta)]
	discretized_y = [y(ti) for ti in discrete_time[2:end-1]];
end;

# ╔═╡ 8600c30d-ec97-42d5-baa2-3c735526d5a2
begin
	x_res = copy(powerS, discretized_y)
	for k in 1:max_iter
		
		discretized_ξ = [zeros(3) for ti in discrete_time[2:end-1]];
		discretized_ψ = [legendre_derivative(x_res[i][3], h_ref) for i in 1:N] 
		
		ξ = solve_SPP(powerS, x_res, discretized_ξ, discretized_ψ, integrand_F, integrand_F_rest, integrand_ξψ, integrand_ψψ, discrete_time, retraction, test_spaces, evaluate)
		
		acc = false 
		λ = 1.0
		x_test = copy(powerS, x_res)
		
		while acc == false
			x_test = retract(powerS, x_res, λ*ξ, ProjectionRetraction())
			x_third = zeros(length(x_test))
			for i in 1:length(x_test)
				x_third[i] = x_test[i][3]
			end
			if any(x -> x > (1-h_ref), x_third)
				λ = λ/2
			else
				acc = true
			end
		end
		#println("Dämpfungsfaktor λ = ", λ)
		#retract!(powerS, x_res, x_res, λ*ξ, ProjectionRetraction())
		if (!all(x -> x == 0.0, ξ)) && distance(powerS, x_res, x_test)/norm(x_res) < 1e-6
			println("Äußere Iteration konvergiert")
			break
		end
		println("α = ", integrand_F.proximity_parameter)
		x_res .= copy(powerS, x_test)
	end 
end;

# ╔═╡ ddad9dbb-4087-4819-b2ae-7ac9187424c2
begin
n = 30
u = range(0,stop=2*π,length=n);
v = range(0,stop=π,length=n);
	
sx = [cos(ui) * sin(vj) for ui in u, vj in v]
sy = [sin(ui) * sin(vj) for ui in u, vj in v]
sz = [cos(vj) for ui in u, vj in v]

π1(x) = 1.01*x[1]
π2(x) = 1.01*x[2]
π3(x) = 1.01*x[3]

geodesic_start = [y0, discretized_y ...,yT]

geodesic_final = [y0, x_res..., yT]

fig = Figure(resolution = (1400, 900), padding=0)
ax = Axis3(fig[1, 1]; aspect =:data)
hidedecorations!(ax)
hidespines!(ax)


	x = acos(1-h_ref)

	circx = [cos(ui)*sin(x) for ui in u]
	circy = [sin(ui)*sin(x) for ui in u]
	circz = fill(cos(x), n)

	wireframe!(ax, sx, sy, sz, color = RGBA(0.5,0.5,0.7,0.1); transparency=true)

	scatterlines!(ax, circx, circy, circz; markersize =2, color=:black, linewidth=2)

	#scatterlines!(ax, π1.(geodesic_start), π2.(geodesic_start), π3.(geodesic_start); markersize =8, color=:orange, linewidth=2)
	
	scatterlines!(ax, π1.(geodesic_final), π2.(geodesic_final), π3.(geodesic_final); markersize =8, color=:orange, linewidth=2)
	
	scatter!(ax, π1.([y0]), π2.([y0]), π3.([y0]); markersize = 10, color=:green)
	scatter!(ax, π1.([yT]), π2.([yT]), π3.([yT]); markersize = 10, color=:red)
	
	ax.azimuth[] += 14.445
	ax.elevation[] = 35.02

	limits!(ax, -1.5, 1.5, -1.5, 1.5, -1.5, 1.5)
	fig
end

# ╔═╡ Cell order:
# ╠═3d0776f8-1bac-11f1-237d-cd2b0f404020
# ╠═fe7d4e51-8c93-467c-99e9-c546900775b8
# ╠═683a3a9d-64f8-49f8-9d4c-fa86b3d08335
# ╠═18a99268-7d83-483a-b279-ab7185850094
# ╠═c54635ae-2c08-4bf6-b536-1b50ec3489db
# ╠═0cb7b735-2118-440d-aa24-9da29192b1c7
# ╠═2d8fdd96-fc60-40b7-9517-141cae486cf5
# ╠═5ccea4c1-a34c-4d72-b095-5cc3b849506e
# ╠═362cb1d2-eda2-4e1b-a995-c10f68e74e34
# ╠═f87800a1-55d5-4994-a9d5-c248cf9fc78e
# ╠═e71b9f35-6d84-46fe-813c-70bb95d94c58
# ╠═45d6a596-88c1-4ec4-8de1-a657977e857d
# ╠═c5c6c894-b975-4ff2-b411-46bf52049e67
# ╠═8600c30d-ec97-42d5-baa2-3c735526d5a2
# ╠═ddad9dbb-4087-4819-b2ae-7ac9187424c2
