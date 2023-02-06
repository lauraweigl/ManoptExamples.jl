@doc raw"""
    RiemannianMeanCost{P} where P

A functor representing the Riemannian center of mass (or Riemannian mean) cost function.

For a given set of points ``d_1,\ldots,d_N`` this cost function is defined as

```math
f(p) = \sum_{j=i}^N d_{mathcal M}^2(d_i, p),
```

where ``d_{\mathcal M}`` is the [`distance`]() on a Riemannian manifold.

# Constructor

    RiemannianMeanCost(data::AbstractVector{<:P}) where {P}

Initialize the cost function to a data set `data` of points on a manfiold of type `P`.
"""
struct RiemannianMeanCost{P,V<:AbstractVector{<:P}}
    data::V
end
function (rmc::RiemannianMeanCost)(M, p)
    return sum(distance(M, p, di)^2 for di in rmc.data)
end

@doc raw"""
    RiemannianMeanGradient!!{P} where P

A functor representing the Riemannian center of mass (or Riemannian mean) cost function.

For a given set of points ``d_1,\ldots,d_N`` this cost function is defined as

```math
\operatorname{grad}f(p) = \sum_{j=i}^N \log_p d_i
```

where ``d_{\mathcal M}`` is the [`distance`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions.html#ManifoldsBase.distance-Tuple{AbstractManifold,%20Any,%20Any})
on a Riemannian manifold and we employ [`grad_distance`](@ref) to compute the single summands.

This functor provides both the allocating variant `grad_f(M,p)` as well as
the in-place variant `grad_f!(M, X, p)` which computes the gradient in-place of `X`.

# Constructors

    RiemannianMeanGradient!!(data::AbstractVector{P}, initial_vector::T=nothing) where {P,T}

Generate the Riemannian mean gradient based on some data points `data` an intial tangent vector `initial_vector`.
If you do not provide an initial tangent vector to allocate the intermediate storage of a tangent vector,
you can only use the allocating variant.

    RiemannianMeanGradient!!(
        M::AbstractManifold,
        data::AbstractVector{P};
        initial_vector::T=zero_vector(M, first(data)),
    ) where {P,T}

Initialize the Riemannian mean gradient, where the internal storage for tangent vectors can
be created automatically, since the Riemannian manifold `M` is provideed.
"""
struct RiemannianMeanGradient!!{P,T,V<:AbstractVector{<:P}}
    X::T
    data::V
end
function RiemannianMeanGradient!!(
    data::V, initial_vector::T
) where {P,T,V<:AbstractVector{<:P}}
    return RiemannianMeanGradient!!{P,T,V}(initial_vector, data)
end
function RiemannianMeanGradient!!(
    M::AbstractManifold, data::V; initial_vector::T=zero_vector(M, first(data))
) where {P,T,V<:AbstractVector{<:P}}
    return RiemannianMeanGradient!!{P,T,V}(initial_vector, data)
end
function (rmg::RiemannianMeanGradient!!)(M, p)
    return sum(grad_distance(M, di, p) for di in rmg.data)
end
function (rmg::RiemannianMeanGradient!!{T})(M, X::T, p) where {T}
    zero_vector!(M, X, p)
    for di in rng.data
        grad_distance!(M, rmg.X, di, p)
        X .+= rmd.X
    end
    return X
end

function Riemannian_mean_objective() end
