function get_jacobian!(M::AbstractManifold, x, xi, eval, A, integrand, time_interval, retraction; row_index = nothing, column_index = nothing, test_space = nothing, ansatz_space = nothing)

    isnothing(test_space) && error("Please provide the space of the test functions")
    isnothing(ansatz_space) && error("Please provide the space of the ansatz functions")

    # loop: time intervals
    for i in 1:(length(time_interval) - 1)

        h = time_interval[i + 1] - time_interval[i]

        xl = eval(x, i, 0.0)
        xr = eval(x, i, 1.0)

        xil = eval(xi, i, 0.0)
        xir = eval(xi, i, 1.0)

        base_test_space_left = build_base(test_space.manifold, xl)
        base_test_space_right = build_base(test_space.manifold, xr)


        if test_space.degree == 1
            assemble_local_jacobian!(M, xl, xr, xil, xir, A, h, i, base_test_space_left, 1, 0, base_test_space_left, 1, 0, integrand, retraction)
            assemble_local_jacobian!(M, xl, xr, xil, xir, A, h, i, base_test_space_right, 0, 1, base_test_space_left, 1, 0, integrand, retraction)
            assemble_local_jacobian!(M, xl, xr, xil, xir, A, h, i, base_test_space_left, 1, 0, base_test_space_right, 0, 1, integrand, retraction)
            assemble_local_jacobian!(M, xl, xr, xil, xir, A, h, i, base_test_space_right, 0, 1, base_test_space_right, 0, 1, integrand, retraction)
        else
            error("The case degree ≠ 1 is not yet implemented")
        end
    end
    return
end

function get_jacobian_ψ!(M::AbstractManifold, x, xi, psi, eval, A, integrand, time_interval; row_index = nothing, column_index = nothing, test_space = nothing, ansatz_space = nothing)

    isnothing(test_space) && error("Please provide the space of the test functions")
    isnothing(ansatz_space) && error("Please provide the space of the ansatz functions")

    # loop: time intervals
    for i in 1:(length(time_interval) - 1)

        h = time_interval[i + 1] - time_interval[i]

        xl = eval(x, i, 0.0)
        xr = eval(x, i, 1.0)

        xil = eval(xi, i, 0.0)
        xir = eval(xi, i, 1.0)

        psil = eval(psi, i, 0.0)
        psir = eval(psi, i, 1.0)

        base_test_space_left = build_base(test_space.manifold, xl)
        base_test_space_right = build_base(test_space.manifold, xr)

        base_ansatz_space_left = build_base(ansatz_space.manifold, psil)
        base_ansatz_space_right = build_base(ansatz_space.manifold, psir)


        if test_space.degree == 1
            assemble_local_jacobian!(M, xl, xr, xil, xir, psil, psir, A, h, i, base_ansatz_space_left, 1, 0, base_test_space_left, 1, 0, integrand)
            assemble_local_jacobian!(M, xl, xr, xil, xir, psil, psir, A, h, i, base_ansatz_space_right, 0, 1, base_test_space_left, 1, 0, integrand)
            assemble_local_jacobian!(M, xl, xr, xil, xir, psil, psir, A, h, i, base_ansatz_space_left, 1, 0, base_test_space_right, 0, 1, integrand)
            assemble_local_jacobian!(M, xl, xr, xil, xir, psil, psir, A, h, i, base_ansatz_space_right, 0, 1, base_test_space_right, 0, 1, integrand)
        else
            error("The case degree ≠ 1 is not yet implemented")
        end
    end
    return
end

function get_right_hand_side!(M::AbstractManifold, x, xi, eval, b, integrand, time_interval, retraction; row_index = nothing, test_space = nothing)

    isnothing(test_space) && error("Please provide the space of the test functions")

    # loop: time intervals
    for i in 1:(length(time_interval) - 1)
        x_left = eval(x, i, 0.0)
        x_right = eval(x, i, 1.0)
        xi_left = eval(xi, i, 0.0)
        xi_right = eval(xi, i, 1.0)
        
        base_test_function_left = build_base(test_space.manifold, x_left) # testspace = T_xM
        base_test_function_right = build_base(test_space.manifold, x_right)

        h = time_interval[i + 1] - time_interval[i]

        if test_space.degree == 1
            assemble_local_right_hand_side!(M, x_left, x_right, xi_left, xi_right, b, h, i, base_test_function_left, 1, 0, integrand, retraction)
            assemble_local_right_hand_side!(M, x_left, x_right, xi_left, xi_right, b, h, i, base_test_function_right, 0, 1, integrand, retraction)
        else
            error("The case degree ≠ 1 is not yet implemented")
        end
    end
    return
end

function get_right_hand_side!(M::AbstractManifold, x, xi, psi, eval, b, integrand, time_interval, retraction; row_index = nothing, test_space = nothing)

    isnothing(test_space) && error("Please provide the space of the test functions")

    # loop: time intervals
    for i in 1:(length(time_interval) - 1)
        x_left = eval(x, i, 0.0)
        x_right = eval(x, i, 1.0)
        xi_left = eval(xi, i, 0.0)
        xi_right = eval(xi, i, 1.0)
        psi_left = eval(psi, i, 0.0)
        psi_right = eval(psi, i, 1.0)
        
        base_test_function_left = build_base(test_space.manifold, x_left) # testspace = T_xM
        base_test_function_right = build_base(test_space.manifold, x_right)

        h = time_interval[i + 1] - time_interval[i]

        if test_space.degree == 1
            assemble_local_right_hand_side!(M, x_left, x_right, xi_left, xi_right, psi_left, psi_right, b, h, i, base_test_function_left, 1, 0, integrand)
            assemble_local_right_hand_side!(M, x_left, x_right, xi_left, xi_right, psi_left, psi_right, b, h, i, base_test_function_right, 0, 1, integrand)
        else
            error("The case degree ≠ 1 is not yet implemented")
        end
    end
    return
end

function get_right_hand_side_ψ!(M::AbstractManifold, x, xi, psi, eval, b, integrand, time_interval; test_space = nothing)

    isnothing(test_space) && error("Please provide the space of the test functions")

    # loop: time intervals
    for i in 1:(length(time_interval) - 1)
        x_left = eval(x, i, 0.0)
        x_right = eval(x, i, 1.0)
        xi_left = eval(xi, i, 0.0)
        xi_right = eval(xi, i, 1.0)
        psi_left = eval(psi, i, 0.0)
        psi_right = eval(psi, i, 1.0)
        
        base_test_function_left = build_base(test_space.manifold, psi_left) # testspace = R
        base_test_function_right = build_base(test_space.manifold, psi_right)

        h = time_interval[i + 1] - time_interval[i]

        if test_space.degree == 1
            assemble_local_right_hand_side!(M, x_left, x_right, xi_left, xi_right, psi_left, psi_right, b, h, i, base_test_function_left, 1, 0, integrand)
            assemble_local_right_hand_side!(M, x_left, x_right, xi_left, xi_right, psi_left, psi_right, b, h, i, base_test_function_right, 0, 1, integrand)
        else
            error("The case degree ≠ 1 is not yet implemented")
        end
    end
    return
end

function assemble_local_jacobian!(M, x_left, x_right, xi_left, xi_right, A, h, i, base_ansatz, bfl, bfr, base_test, tfl, tfr, integrand, retraction)
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

    retra_dot = (retraction.value(x_right,xi_right) - retraction.value(x_left,xi_left)) / h # approximate time derivative 
 
    quadrature_weight = 0.5 * h
    nA1 = size(A, 1)
    nA2 = size(A, 2)

    #	loop: components of test functions
    for k in 1:dim_test
        # loop: components of ansatz functions
        for j in 1:dim_ansatz
            # ensure that indices remain in bounds
            if idx + j >= 1 && idxc + k >= 1 && idx + j <= nA2 && idxc + k <= nA1

                retr_prime_dotTF = (retraction.derivative(x_right,xi_right,tfr*base_test[k]) - retraction.derivative(x_left,xi_left,tfl*base_test[k]))/h

                retr_prime_dotAF = (retraction.derivative(x_right,xi_right,bfr*base_ansatz[j]) - retraction.derivative(x_left,xi_left,bfl*base_ansatz[j]))/h

                retr_doubleprime_dot = (retraction.second_derivative(x_right,xi_right,bfr*base_ansatz[j],tfr*base_test[k]) - retraction.second_derivative(x_left,xi_left,tfl*base_ansatz[j],bfl*base_test[k]))/h


                # derivative (using the embedding) at right and left quadrature point
                tmp = integrand.derivative(integrand, x_left, retra_dot, retr_prime_dotTF, retr_prime_dotAF, retr_doubleprime_dot)

                tmp += integrand.derivative(integrand, x_right, retra_dot, retr_prime_dotAF, retr_prime_dotTF, retr_doubleprime_dot)

                # Update matrix entry
                A[idxc + k, idx + j] += quadrature_weight * tmp
            end
        end
    end
    return
end

function assemble_local_jacobian!(M, x_left, x_right, xi_left, xi_right, psi_left, psi_right, A, h, i, base_ansatz, bfl, bfr, base_test, tfl, tfr, integrand)
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
 
    quadrature_weight = 0.5 * h
    nA1 = size(A, 1)
    nA2 = size(A, 2)

    #	loop: components of test functions
    for k in 1:dim_test
        # loop: components of ansatz functions
        for j in 1:dim_ansatz
            # ensure that indices remain in bounds
            if idx + j >= 1 && idxc + k >= 1 && idx + j <= nA2 && idxc + k <= nA1

                tmp = integrand.derivative(integrand, x_left, xi_left, psi_left, bfl*base_ansatz[j], tfl*base_test[k])

                tmp += integrand.derivative(integrand, x_right, xi_right, psi_right, bfr*base_ansatz[j], tfr*base_test[k])

                # Update matrix entry
                A[idxc + k, idx + j] += quadrature_weight * tmp
            end
        end
    end
    return
end

function assemble_local_right_hand_side!(M, x_left, x_right, xi_left, xi_right, b, h, i, base_test_space, tlf, trf, integrand, retraction)
    dim_test = length(base_test_space)
    if trf == 1
        idx = dim_test * (i - 1)
    else
        idx = dim_test * (i - 2)
    end

    retra_dot = (retraction.value(x_right,xi_right) - retraction.value(x_left,xi_left)) / h

    # trapezoidal rule
    quadwght = 0.5 * h
    for k in 1:dim_test
        # finite differences, taking into account values of test function at both endpoints
        if idx + k > 0 && idx + k <= length(b)

            retr_prime_dot = (retraction.derivative(x_right,xi_right,trf*base_test_space[k]) - retraction.derivative(x_left,xi_left,tlf*base_test_space[k]))/h

            # left quadrature point
            tmp = integrand.value(integrand, x_left, retra_dot, retr_prime_dot, tlf*base_test_space[k])
            # right quadrature point
            tmp += integrand.value(integrand, x_right, retra_dot, retr_prime_dot, trf*base_test_space[k])
            # Update rhs
            b[idx + k] += quadwght * tmp
        end
    end
    return
end

function assemble_local_right_hand_side!(M, x_left, x_right, xi_left, xi_right, psi_left, psi_right, b, h, i, base_test_space, tlf, trf, integrand)
    dim_test = length(base_test_space)
    if trf == 1
        idx = dim_test * (i - 1)
    else
        idx = dim_test * (i - 2)
    end

    # trapezoidal rule
    quadwght = 0.5 * h
    for k in 1:dim_test
        # finite differences, taking into account values of test function at both endpoints
        if idx + k > 0 && idx + k <= length(b)
            # left quadrature point
            tmp = integrand.value(integrand, x_left, xi_left, psi_left, tlf*base_test_space[k])
            # right quadrature point
            tmp += integrand.value(integrand, x_right, xi_right, psi_right, trf*base_test_space[k])
            # Update rhs
            b[idx + k] += quadwght * tmp
        end
    end
    return
end