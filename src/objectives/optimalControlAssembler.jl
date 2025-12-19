raw"""
Helper function that builds a basis of the tangent space of M at p
"""

function build_base(M::AbstractManifold, p)
    Bl = get_basis(M, p, DefaultOrthonormalBasis())
    return get_vectors(M, p, Bl)
end

@doc raw"""
This function is called by Newton's method to compute one block of the matrix for the Newton step

Input:

M:                      Product manifold\\
y:                      iterate\\
eval:                   function that evaluates y at left and right boundary point of i-th interval, signature: eval(y, i, scaling), must return an element of M
A:                      Matrix to be written into\\
integrand:	            integrand of the functional as a struct, must have a field value and a field derivative\\
transport:	            vectortransport used to compute the connection term (as a struct, must have a field value and a field derivative)\\
time_intervals:			time interval with discrete time points

Keyword arguments:

row_index:                  row index of block inside system\\
column_index:               column index of block inside system\\
test_space:    				space of test functions as a struct, must have a field manifold (base manifold of the tangent spaces) and a field degree (degree of test functions (1: linear, 0: constant))\\
ansatz_space:   			space of ansatz functions as a struct, must have a field manifold (base manifold of the tangent spaces) and a field degree (degree of ansatz functions (1: linear, 0: constant))\\

...
"""

# Die Funktion braucht man, wenn man eine Testfunktion in der Ableitung des Vektortransports hat und eine außerhalb, also z.B. A'(y)ϕ(P'(y)δy p)

function get_Jac_Lyy!(M::ProductManifold, y, eval, A, integrand, transport, time_interval; row_index = nothing, column_index = nothing, test_space = nothing, ansatz_space = nothing)
    isnothing(row_index) && error("Please provide the row index of the block to be assembled")
    isnothing(column_index) && error("Please provide the column index of the block to be assembled")
    isnothing(test_space) && error("Please provide the space of the test functions")
    isnothing(ansatz_space) && error("Please provide the space of the ansatz functions")

    degree_test_function = test_space.degree
    degree_ansatz_function = ansatz_space.degree

    # Schleife über Intervalle
    for i in 1:(length(time_interval) - 1)

        # Evaluation of the current iterate. This routine has to be provided from outside, because Knowledge about the basis functions is needed
        yl = eval(y, i, 0.0)
        yr = eval(y, i, 1.0)

        #yl=ArrayPartition(getindex.(y.x, (i-1...,)))
        #yr=ArrayPartition(getindex.(y.x, (i...,)))


        base_ansatz_space_left = build_base(ansatz_space.manifold, yl[M, column_index])
        base_ansatz_space_right = build_base(ansatz_space.manifold, yr[M, column_index])
        base_test_space_left = build_base(test_space.manifold, yl[M, row_index])
        base_test_space_right = build_base(test_space.manifold, yr[M, row_index])

        h = time_interval[i + 1] - time_interval[i]

        if degree_test_function == 1 && degree_ansatz_function == 1
        # In the following, all combinations of test and basis functions have to be considered.
            assemble_local_jacobian_Lyy!(M, yl, yr, A, h, i, base_ansatz_space_left, 1, 0, base_test_space_left, 1, 0, integrand, transport; row_index = row_index)
            assemble_local_jacobian_Lyy!(M, yl, yr, A, h, i, base_ansatz_space_right, 0, 1, base_test_space_left, 1, 0, integrand, transport; row_index = row_index)
            assemble_local_jacobian_Lyy!(M, yl, yr, A, h, i, base_ansatz_space_left, 1, 0, base_test_space_right, 0, 1, integrand, transport; row_index = row_index)
            assemble_local_jacobian_Lyy!(M, yl, yr, A, h, i, base_ansatz_space_right, 0, 1, base_test_space_right, 0, 1, integrand, transport; row_index = row_index)
        end

    end
    return
end

"""
 A:      Matrix to be written into\\
row_index: row index of block inside system\\
column_index: column index of block inside system\\

h:       length of interval\\
i:       index of interval\\

yl:      left value of iterate\\
yr:      right value of iterate\\

B:       basis vector for basis function\\
bfl:     0/1 scaling factor at left boundary\\
bfr:     0/1 scaling factor at right boundary \\

T:       basis vector for test function\\
tfl:     0/1 scaling factor at left boundary\\
tfr:     0/1 scaling factor at right boundary \\
...
"""
function assemble_local_jacobian_Lyy!(M, y_left, y_right, A, h, i, base_ansatz, bfl, bfr, base_test, tfl, tfr, integrand, transport; row_index = nothing, M_component = M[row_index])

    dim_ansatz = length(base_ansatz)
    dim_test = length(base_test)

    if tfr == 1
        idxc = dim_test * (i - 1)
    else
        idxc = dim_test * (i - 2)
    end
    if bfr == 1
        idx = dim_ansatz * (i - 1)
    else
        idx = dim_ansatz * (i - 2)
    end

    ydot = (y_right - y_left) / h # approximate time derivative of y
    quadrature_weight = 0.5 * h
    nA1 = size(A, 1)
    nA2 = size(A, 2)

    #	Schleife über Komponenten der Testfunktion
    for k in 1:dim_test
        # Schleife über Komponenten der Basisfunktion
        for j in 1:dim_ansatz
            # Sicherstellen, dass wir in Indexgrenzen der Matrix bleiben
            if idx + j >= 1 && idxc + k >= 1 && idx + j <= nA2 && idxc + k <= nA1

                # approximation of time derivative of ansatz and test functions (=0 am jeweils anderen Rand)
                #Tdot = (tfr - tfl) * base_test[k] / h
                Bdot = (bfr - bfl) * base_ansatz[j] / h

                # modification for covariant derivative:
                # derivative of the vector transport w.r.t. y at left quadrature point
                # P'(yl)bfl*base_ansatz[j] (tfl*base_test(k))

                Pprime_left = transport.derivative(M_component, y_left, bfl * base_ansatz[j], tfl * base_test[k])
                Pprime_right = transport.derivative(M_component, y_right, bfr * base_ansatz[j], tfr * base_test[k])

                # approximation of the time derivative of the vector transport

                Pprimedot = (Pprime_right - Pprime_left) / h

                # Einsetzen in die rechte Seite am rechten und linken Quadraturpunkt

                tmp = integrand.derivative(integrand, y_left, ydot, bfl * base_ansatz[j], Bdot, bfl * Pprime_left, Pprimedot)
                #tmp+=integrand.derivative(integrand,yr,ydot,bfr*B[j],Bdot,bfr*Pprimel,Pprimedot_neu)


                # Update des Matrixeintrags

                #tmp+=integrand.derivative(integrand,yl,ydot,bfl*B[j], Bdot,bfl*Pprimer,Pprimedot_neu)
                tmp += integrand.derivative(integrand, y_right, ydot, bfr * base_ansatz[j], Bdot, bfr * Pprime_right, Pprimedot)

                A[idxc + k, idx + j] += quadrature_weight * tmp
            end
        end
    end
    return
end

function get_rhs_simplified_y!(eval, b, row_idx, degT, h, nCells, y, y_trial, integrand, transport)
    S = integrand.precodomain
    # loop: time intervals
    for i in 1:nCells
        yl = eval(y, i, 0.0)
        yr = eval(y, i, 1.0)

        yl_trial = eval(y_trial, i, 0.0)
        yr_trial = eval(y_trial, i, 1.0)

        Tcl = get_basis(S, yl.x[row_idx], DefaultOrthonormalBasis())
        Tl = get_vectors(S, yl.x[row_idx], Tcl)

        Tcr = get_basis(S, yr.x[row_idx], DefaultOrthonormalBasis())
        Tr = get_vectors(S, yr.x[row_idx], Tcr)

        if degT == 1
            assemble_local_rhs_OC!(b, row_idx, h, i, yl_trial, yr_trial, Tl, 1, 0, integrand, transport, yl, yr)
            assemble_local_rhs_OC!(b, row_idx, h, i, yl_trial, yr_trial, Tr, 0, 1, integrand, transport, yl, yr)
        end
        if degT == 0
            assemble_local_rhs_OC!(b, row_idx, h, i, yl_trial, yr_trial, Tr, 1, 1, integrand, transport, yl, yr)
        end
    end
    return
end

function assemble_local_rhs_OC!(b, row_idx, h, i, yl, yr, T, tlf, trf, integrand, transport, yl_vorher, yr_vorher)
    dimc = manifold_dimension(integrand.precodomain)
    S = integrand.precodomain
    if trf == 1
        idx = dimc * (i - 1)
    else
        idx = dimc * (i - 2)
    end
    ydotr = (yr - yl) / h
    # trapezoidal rule
    quadwght = 0.5 * h
    for k in 1:dimc
        # finite differences, taking into account values of test function at both endpoints
        if idx + k > 0 && idx + k <= length(b)

            Pl = transport.value(S, yl.x[row_idx], transport.derivative(integrand.domain, yl.x[row_idx], tlf * T[k], tlf * yl.x[3]), yl.x[row_idx])
            Pr = transport.value(S, yr.x[row_idx], transport.derivative(integrand.domain, yr.x[row_idx], trf * T[k], trf * yr.x[3]), yr.x[row_idx])
            Pdot = (Pr - Pl) / h

            tmp = integrand.value(integrand, yl, ydotr, tlf * Pl, Pdot)
            tmp += integrand.value(integrand, yr, ydotr, trf * Pr, Pdot)
            # Update rhs
            b[idx + k] += quadwght * tmp
        end
    end
    return
end


function get_jacobian_simplified!(M, y, y_trial, eval, A, integrand, transport, time_interval; row_index = nothing, column_index = nothing, test_space=nothing, ansatz_space=nothing)
    isnothing(row_index) && error("Please provide the row index of the block to be assembled")
    isnothing(column_index) && error("Please provide the column index of the block to be assembled")
    isnothing(test_space) && error("Please provide the space of the test functions")
    isnothing(ansatz_space) && error("Please provide the space of the ansatz functions")

    degree_test_function = test_space.degree
    degree_ansatz_function = ansatz_space.degree

    # loop: time intervals
    for i in 1:(length(time_interval) - 1)

        # Evaluation of the current iterate. This routine has to be provided from outside, because knowledge about the basis functions is needed
        yl = eval(y, i, 0.0)
        yr = eval(y, i, 1.0)

        yl_trial = eval(y_trial, i, 0.0)
        yr_trial = eval(y_trial, i, 1.0)

        base_ansatz_space_left = build_base(ansatz_space.manifold, yl_trial[M, column_index])
        base_ansatz_space_right = build_base(ansatz_space.manifold, yr_trial[M, column_index])
        base_test_space_left = build_base(test_space.manifold, yl[M, row_index])
        base_test_space_right = build_base(test_space.manifold, yr[M, row_index])

        dim = manifold_dimension(test_space.manifold)

        for k in 1:dim
            base_test_space_left[k] = transport.value(test_space.manifold, yl[M,row_index], base_test_space_left[k], yl_trial[M,row_index])
            base_test_space_right[k] = transport.value(test_space.manifold, yr[M,row_index], base_test_space_right[k], yr_trial[M,row_index])
        end

        h = time_interval[i + 1] - time_interval[i]

        # In the following, all combinations of test and basis functions have to be considered.

        # The case, where both test and basis functions are linear. We have 2x2=4 combinations, since there are two test/basis functions on each interval
        if degree_test_function == 1 && degree_ansatz_function == 1
            assemble_local_jacobian_Lyy!(M, yl_trial, yr_trial, A, h, i, base_ansatz_space_left, 1, 0, base_test_space_left, 1, 0, integrand, transport; row_index = row_index) # ich glaube hier müsste man beim Vektortransport komponenten nehmen, dann passts aber wahrschieblich im oberen Fall nicht
            assemble_local_jacobian_Lyy!(M, yl_trial, yr_trial, A, h, i, base_ansatz_space_right, 0, 1, base_test_space_left, 1, 0, integrand, transport; row_index = row_index)
            assemble_local_jacobian_Lyy!(M, yl_trial, yr_trial, A, h, i, base_ansatz_space_left, 1, 0, base_test_space_right, 0, 1, integrand, transport; row_index = row_index)
            assemble_local_jacobian_Lyy!(M, yl_trial, yr_trial, A, h, i, base_ansatz_space_right, 0, 1, base_test_space_right, 0, 1, integrand, transport; row_index = row_index)
        end
        # Other cases could be added here. In the rod example I did not need them, thus I havent implemented them
    end
    return
end
