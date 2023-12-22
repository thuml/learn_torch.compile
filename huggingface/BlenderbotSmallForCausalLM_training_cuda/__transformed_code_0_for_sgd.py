def __transformed_code_0_for_sgd(params, d_p_list, momentum_buffer_list, has_sparse_grad, foreach, weight_decay, momentum, lr, dampening, nesterov, maximize):
    _ = None; func = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function

    return __resume_at_114_25(_multi_tensor_sgd(params, d_p_list,
        momentum_buffer_list, weight_decay=weight_decay, momentum=momentum, lr=
        lr, dampening=dampening, nesterov=nesterov, has_sparse_grad=
        has_sparse_grad, maximize=maximize))
