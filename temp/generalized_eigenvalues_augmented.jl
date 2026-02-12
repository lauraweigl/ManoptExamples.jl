### A Pluto.jl notebook ###
# v0.20.20

using Markdown
using InteractiveUtils

# ╔═╡ 3e19b659-8855-4020-9173-fe4a1ffbbc08
using Pkg; Pkg.activate();

# ╔═╡ eb35384f-0822-4953-8d34-4ed5a0df71c5
begin
	using LinearAlgebra
	using RandomMatrices
	using Manopt
	using ManoptExamples
	using Manifolds
	using Random
	using WGLMakie, Makie, GeometryTypes, Colors
end;

# ╔═╡ 981a77c4-c6de-47f4-ba9d-75fc29d68fa3
md"""
In this example we compute a generalized eigenvalue by applying Newton's method on vector bundles which was introduced in \ref{paper}. This example reproduces the results from \ref{paper}.
"""

# ╔═╡ 0b74ee34-09f0-4aba-839b-6345be190fef
md"""
Let $A, \, B \in L(X,Y)$ where $(X,\|\cdot\|_X)$ and $(Y,\|\cdot\|_Y)$ are Banach spaces and $\|\cdot\|_X$ is Fréchet differentiable on $X\setminus\{0\}$, so that the unit sphere $\mathbb{S}^X := \{ x \in X \mid \|x\|_X = 1\}$ is an embedded submanifold of $X$.
A generalized eigenvalue problem consists of finding nonzero vectors $x\in X$ and numbers $\mu \in \mathbb C$ such that 

$$Ax = \mu Bx \quad \Leftrightarrow \quad Ax \in \mathrm{span}(Bx).$$
Consider the vector bundle $\mathcal E$ with base manifold $\mathcal Y=Y$, the quotient spaces $E_y = Y / \mathrm{span}(y)$ as fibres, whose elements are denoted by $[v]_y=v+\mathrm{span}(y)\in E_y$, and a vector bundle projection $p: \mathcal E \to Y, \; E_y \mapsto y$. Clearly, $0_y=[0]_y=\mathrm{span}(y)$. 
Consider the mapping

$$\begin{align*}
    F : \mathbb{S}^X &\to \mathcal E \\
    x &\mapsto (Bx, [Ax]_{Bx}).
\end{align*}$$

Then we obtain $y(x) = p(F(x)) = Bx$, and a zero $x\in \mathbb{S}^X$ of $F$ satisfies

$$F(x) = 0_{y(x)} \in E_{y(x)} \Leftrightarrow Ax \in  [0]_{Bx} \subset Y\;  \Leftrightarrow Ax \in \mathrm{span}(Bx), \text{ i.e. $x$ is an eigenvector. }$$

We will apply Newton's method to $F$ to find a real eigenvalue $\mu \in \mathbb R$ (if any exists). 
"""

# ╔═╡ 244385d0-9bcc-4429-be76-881e85b772ae
md"""
As an illustrating example we choose $N=101$, $X=Y = \mathbb R^N$ and $A,B \in \mathbb R^{N\times N}$ with

$$A_{ij} := \begin{cases}
i & \text{if } i=j \\
1 & \text{else}
\end{cases}, \qquad B_{ij} := \begin{cases}
-1 & \text{if } i > j \\
\phantom{-}0 & \text{else}
\end{cases}.$$
"""

# ╔═╡ 587cf35f-882f-4e5a-8527-a53ba7c5644f
begin
    N = 101
	M = Manifolds.Sphere(N-1)
	
    eig_A = Matrix{Float64}(I, N, N)
    for i in 1:N
        eig_A[i, i] = i
    end

    for i in 1:N
		for j in 1:N
			if i > j
				eig_A[i,j] = 1.0
			end
			if i < j
				eig_A[i,j] = 1.0
			end
		end
	end
	
	B = zeros(N,N)
	for i in 1:N
		for j in 1:N
			if i > j
				B[i,j] = -1.0
			end
		end
	end
end;

# ╔═╡ 7bf4f00d-7473-42bd-a342-7d49f6a7a4d7
md"""
For the computation of the Newton direction $\delta x$ we have to solve the Newton equation 

$$Q_{F'(x)} \circ F''(x)\delta x + F'(x) = 0_{y(x)} \text{ in } E_{y(x)}$$

while taking into account that $\delta x \in T_x\mathbb{S}^X\subset X$.

In \ref{paper} we derived a matrix representation given by 

$$Q_{F(x)}\circ F'(x)\delta x = \left[A\delta x-\frac{\langle Bx,Ax\rangle_Y}{\langle Bx,Bx\rangle_Y} B\delta x\right]_{Bx}.$$

Using this, the representation of the tangent space of the sphere given by 

$$T_x\mathbb S^X = x^\perp = \{ v \in X \mid \langle x, v \rangle_2 = x^Tv = 0\}$$

and a Lagrangian multiplier $\lambda \in \mathbb R$, we can write the Newton equation as a saddle point system:

$$\begin{pmatrix}
                A-\frac{\langle Bx, Ax\rangle_2}{\langle Bx,Bx\rangle_2}B & -Bx\\
                x^T & 0
            \end{pmatrix}
            \begin{pmatrix}
                \delta x \\ \lambda
            \end{pmatrix}
            + \begin{pmatrix}
                Ax \\ 0
            \end{pmatrix}
            = \begin{pmatrix}
                0 \\ 0
            \end{pmatrix}.$$

To increase the numerical stability we use an estimate $\widehat\lambda$ for the Lagrangian multiplier $\lambda$ given by

$$\widehat\lambda(x) := \frac{\langle Ax, Bx\rangle_2}{\langle Bx,Bx\rangle_2}$$

for $x \in \mathbb{S}^X$ and consider instead the linear system

$$\begin{pmatrix}
                A-\widehat\lambda(x)B & -Bx\\
                x^T & 0
            \end{pmatrix}
            \begin{pmatrix}
                \delta x \\ \delta\lambda
            \end{pmatrix}
            + \begin{pmatrix}
                Ax - \widehat\lambda(x) Bx \\ 0
            \end{pmatrix}
            = \begin{pmatrix}
                0 \\ 0
            \end{pmatrix}.$$
"""

# ╔═╡ d12f9edf-42f5-4065-9d69-2188b0970457
md"""
The following routine returns the estimator $\widehat{\lambda}(x)$.
"""

# ╔═╡ f677d57b-a924-42da-8e56-534e90d28b2d
λ_estimator(A,B,x) = ((A*x)'*(B*x))/((B*x)'*(B*x));

# ╔═╡ 4a1e4638-aeba-4c88-b217-6b35643b0f36
md"""
As a vector back-transport (needed for the simplified right hand side) we use the projection onto $\mathrm{span}(Bx)$.
"""

# ╔═╡ 7339914a-bb5b-4b1b-81c6-b9696419481c
transport_by_projection(q, p) = I - (1/norm(B*p)^2)*(B*p)*((B*p)');

# ╔═╡ 539829de-30b6-4c5c-b2ef-f24574a95a65
md"""
`NewtonEquation`

In this example we implement a functor to compute the Newton matrix and the right hand side for the Newton equation. It returns the matrix and the right hand side in base representation.
Moreover, for the computation of the simplified Newton direction (which is necessary for affine covariant damping) a method returning the right hand side for the simplified Newton equation is provided.
"""

# ╔═╡ e16a578c-3d64-11f0-057c-c1f978ba732a
begin
struct NewtonEquation{F, T, L, VT, NM, Nrhs}
	A_eig::F
	B_eig::T
	λ_est::L
	transport::VT
	A::NM
	b::Nrhs
end

function NewtonEquation(M, A_eig, B_eig, λ_est, transport)
	n = manifold_dimension(M)
	A = zeros(N+1,N+1)
	b = zeros(N+1)
	return NewtonEquation{typeof(A_eig), typeof(B_eig), typeof(λ_est), typeof(transport), typeof(A), typeof(b)}(A_eig, B_eig, λ_est, transport, A, b)
end
	
function (ne::NewtonEquation)(M, VB, p)
    ne.A .= hcat(vcat(ne.A_eig - ne.λ_est(ne.A_eig, ne.B_eig, p)*ne.B_eig, p'), vcat(-ne.B_eig*p, 0))
    ne.b .= vcat(ne.A_eig*p- ne.λ_est(ne.A_eig, ne.B_eig, p)*ne.B_eig*p, 0)
end
	
function (ne::NewtonEquation)(M, VB, p, p_trial)
	rhs_p_trial = ne.A_eig*p_trial - ne.λ_est(ne.A_eig, ne.B_eig, p_trial)*ne.B_eig*p_trial
    return vcat(ne.transport(p_trial, p)'*rhs_p_trial, 0)
end
end;

# ╔═╡ ea6fbf58-dea1-4436-b3a5-b55e25d49ea3
md"""
We solve the linear system directly and return the Newton direction $\delta x$ which consists of the first $N$ entries of the returned vector.
"""

# ╔═╡ 7bf95fe2-20cb-4824-9d5d-b520b4e6e1ad
begin
	function solve_augmented_system(problem, newtonstate) 
		return ((problem.newton_equation.A) \ (-problem.newton_equation.b))[1:end-1]
	end
end;

# ╔═╡ 427233d0-567d-4cac-930d-7c80178f2d09
begin
	x0 = ones(N)
	x0 = x0/norm(x0)
	
	NE = NewtonEquation(M, eig_A, B, λ_estimator, transport_by_projection)
		
	st_res = vectorbundle_newton(M, TangentBundle(M), NE, x0; sub_problem=solve_augmented_system, sub_state=AllocatingEvaluation(),
	stopping_criterion=(StopAfterIteration(50)|StopWhenChangeLess(M,1e-12)),
	retraction_method=ProjectionRetraction(),
	stepsize=Manopt.AffineCovariantStepsize(M, θ_des=0.05),
	debug=[:Iteration, (:Change, "Change: %1.8e"), "\n", :Stop, (:Stepsize, "Stepsize: %1.8e"), "\n",],
	record=[:Iterate, :Change, :Stepsize],
	return_state=true
)
end

# ╔═╡ 66bb735a-a41d-4742-8581-fe48907b7489
change = get_record(st_res, :Iteration, :Change)[2:end];

# ╔═╡ 285c7cdc-fdc7-43e8-99fa-17fd708d4328
begin
	f = Figure(;)
	
    row, col = fldmod1(1, 2)
	
	Axis(f[row, col], yscale = log10, title = string("Semilogarithmic Plot of the norms of the Newton direction"), xminorgridvisible = true, xticks = (1:length(change)), xlabel = "Iteration", ylabel = "‖δx‖")
    scatterlines!(change[1:end], color = :blue)
	f
end

# ╔═╡ d7173619-6b1e-4d7c-890b-d62499254048
stepsize = get_record(st_res, :Iteration, :Stepsize)[2:end];

# ╔═╡ dd0600d6-4063-4529-af81-afc924ed51e9
begin
	f2 = Figure(;)

	row2, col2 = fldmod1(1, 2)
	
	Axis(f2[row2, col2], title = string("Stepsizes"), xminorgridvisible = true, xticks = (1:length(stepsize)), xlabel = "Iteration", ylabel = "Stepsize")
    scatterlines!(stepsize[1:end], color = :blue)
	f2
end

# ╔═╡ dd7fea0d-9ada-4bec-91c8-2338969f71e6
res = get_record(st_res, :Iteration, :Iterate)[end];

# ╔═╡ e051777b-18c0-46c0-9903-eeb4d8f65bc0
md"""
We compute an estimate for a generalized eigenvalue via $\widehat\lambda$
"""

# ╔═╡ fe6488ec-4099-42c0-8fe4-d48cafb1a68b
eigenvalue = λ_estimator(eig_A, B, res)

# ╔═╡ 3743d687-a712-42d0-8743-5ae4bde845b3
md"""
and check the result by plugging it into the generalized eigenvalue equation 
"""

# ╔═╡ a7a13b69-dd2c-4f24-989d-dc3356633184
norm(eig_A*res - eigenvalue*B*res)

# ╔═╡ Cell order:
# ╠═3e19b659-8855-4020-9173-fe4a1ffbbc08
# ╟─981a77c4-c6de-47f4-ba9d-75fc29d68fa3
# ╠═eb35384f-0822-4953-8d34-4ed5a0df71c5
# ╟─0b74ee34-09f0-4aba-839b-6345be190fef
# ╟─244385d0-9bcc-4429-be76-881e85b772ae
# ╠═587cf35f-882f-4e5a-8527-a53ba7c5644f
# ╟─7bf4f00d-7473-42bd-a342-7d49f6a7a4d7
# ╟─d12f9edf-42f5-4065-9d69-2188b0970457
# ╠═f677d57b-a924-42da-8e56-534e90d28b2d
# ╟─4a1e4638-aeba-4c88-b217-6b35643b0f36
# ╠═7339914a-bb5b-4b1b-81c6-b9696419481c
# ╟─539829de-30b6-4c5c-b2ef-f24574a95a65
# ╠═e16a578c-3d64-11f0-057c-c1f978ba732a
# ╟─ea6fbf58-dea1-4436-b3a5-b55e25d49ea3
# ╠═7bf95fe2-20cb-4824-9d5d-b520b4e6e1ad
# ╠═427233d0-567d-4cac-930d-7c80178f2d09
# ╠═66bb735a-a41d-4742-8581-fe48907b7489
# ╠═285c7cdc-fdc7-43e8-99fa-17fd708d4328
# ╠═d7173619-6b1e-4d7c-890b-d62499254048
# ╠═dd0600d6-4063-4529-af81-afc924ed51e9
# ╠═dd7fea0d-9ada-4bec-91c8-2338969f71e6
# ╟─e051777b-18c0-46c0-9903-eeb4d8f65bc0
# ╠═fe6488ec-4099-42c0-8fe4-d48cafb1a68b
# ╟─3743d687-a712-42d0-8743-5ae4bde845b3
# ╠═a7a13b69-dd2c-4f24-989d-dc3356633184
