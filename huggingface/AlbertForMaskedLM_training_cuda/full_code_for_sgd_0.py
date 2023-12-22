
# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_114_6(___stack0):
    foreach = False
    if foreach:
        if torch.jit.is_scripting():
            raise RuntimeError(
                'torch.jit.script not supported with foreach optimizers')
    if foreach:
        if not torch.jit.is_scripting():
            func = _multi_tensor_sgd
        else:
            func = _single_tensor_sgd
    else:
        func = _single_tensor_sgd
    func(params, d_p_list, momentum_buffer_list, weight_decay=weight_decay,
        momentum=momentum, lr=lr, dampening=dampening, nesterov=nesterov,
        has_sparse_grad=has_sparse_grad, maximize=maximize)
    return None

def transformed___resume_at_114_6(___stack0):
    L = {"___stack0": ___stack0}
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_114_6(___stack0)

#============ end of __resume_at_114_6 ============#

def __guard_0_for_sgd(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_type_id(L['lr'], 7644160)) \
        and (L['lr'] == 0.01) \
        and (___check_type_id(L['params'], 7642176)) \
        and (len(L['params']) == 28) \
        and (___check_obj_id(L['foreach'], 7677664)) \
        and (___check_type_id(L['d_p_list'], 7642176)) \
        and (len(L['d_p_list']) == 28) \
        and (___check_obj_id(L['maximize'], 7677632)) \
        and (___check_type_id(L['momentum'], 7640416)) \
        and (L['momentum'] == 0) \
        and (___check_obj_id(L['nesterov'], 7677632)) \
        and (___check_type_id(L['dampening'], 7640416)) \
        and (L['dampening'] == 0) \
        and (hasattr(L['d_p_list'][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][2], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][3], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][4], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][5], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][6], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][7], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][8], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][9], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][10], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][11], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][12], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][13], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][14], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][15], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][16], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][17], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][18], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][19], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][20], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][21], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][22], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][23], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][24], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][25], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][26], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][27], '_dynamo_dynamic_indices') == False) \
        and (___check_type_id(L['weight_decay'], 7640416)) \
        and (L['weight_decay'] == 0) \
        and (___check_obj_id(L['has_sparse_grad'], 7677632)) \
        and (___check_type_id(L['momentum_buffer_list'], 7642176)) \
        and (len(L['momentum_buffer_list']) == 28) \
        and (___check_obj_id(L['momentum_buffer_list'][0], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][1], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][2], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][3], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][4], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][5], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][6], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][7], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][8], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][9], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][10], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][11], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][12], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][13], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][14], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][15], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][16], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][17], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][18], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][19], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][20], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][21], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][22], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][23], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][24], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][25], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][26], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][27], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139966723186192))) \
        and (___compile_config_hash() == 'd4fb4dd0e48f203468ea4acc759f45a1') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['params'][0], L['params'][1], L['params'][2], L['params'][3], L['params'][4], L['params'][5], L['params'][6], L['params'][7], L['params'][8], L['params'][9], L['params'][10], L['params'][11], L['params'][12], L['params'][13], L['params'][14], L['params'][15], L['params'][16], L['params'][17], L['params'][18], L['params'][19], L['params'][20], L['params'][21], L['params'][22], L['params'][23], L['params'][24], L['params'][25], L['params'][26], L['params'][27], L['d_p_list'][0], L['d_p_list'][1], L['d_p_list'][2], L['d_p_list'][3], L['d_p_list'][4], L['d_p_list'][5], L['d_p_list'][6], L['d_p_list'][7], L['d_p_list'][8], L['d_p_list'][9], L['d_p_list'][10], L['d_p_list'][11], L['d_p_list'][12], L['d_p_list'][13], L['d_p_list'][14], L['d_p_list'][15], L['d_p_list'][16], L['d_p_list'][17], L['d_p_list'][18], L['d_p_list'][19], L['d_p_list'][20], L['d_p_list'][21], L['d_p_list'][22], L['d_p_list'][23], L['d_p_list'][24], L['d_p_list'][25], L['d_p_list'][26], L['d_p_list'][27], tensor_check_names=tensor_check_names))

def __transformed_code_0_for_sgd(params, d_p_list, momentum_buffer_list, has_sparse_grad, foreach, weight_decay, momentum, lr, dampening, nesterov, maximize):
    _ = None; func = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    return __resume_at_114_6(_multi_tensor_sgd(params, d_p_list,
        momentum_buffer_list, weight_decay=weight_decay, momentum=momentum, lr=
        lr, dampening=dampening, nesterov=nesterov, has_sparse_grad=
        has_sparse_grad, maximize=maximize))


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def sgd(params, d_p_list, momentum_buffer_list, has_sparse_grad, foreach, weight_decay, momentum, lr, dampening, nesterov, maximize):
    if foreach is None:
        if not torch.jit.is_scripting():
            __temp_63 = _default_to_fused_or_foreach(params, differentiable=
                False, use_fused=False)
            _ = __temp_63[0]
            foreach = __temp_63[1]
        else:
            foreach = False
    if foreach:
        if torch.jit.is_scripting():
            raise RuntimeError(
                'torch.jit.script not supported with foreach optimizers')
    if foreach:
        if not torch.jit.is_scripting():
            func = _multi_tensor_sgd
        else:
            func = _single_tensor_sgd
    else:
        func = _single_tensor_sgd
    func(params, d_p_list, momentum_buffer_list, weight_decay=weight_decay,
        momentum=momentum, lr=lr, dampening=dampening, nesterov=nesterov,
        has_sparse_grad=has_sparse_grad, maximize=maximize)
    return None

def transformed_sgd(params, d_p_list, momentum_buffer_list, has_sparse_grad, foreach, weight_decay, momentum, lr, dampening, nesterov, maximize):
    L = {"params": params, "d_p_list": d_p_list, "momentum_buffer_list": momentum_buffer_list, "has_sparse_grad": has_sparse_grad, "foreach": foreach, "weight_decay": weight_decay, "momentum": momentum, "lr": lr, "dampening": dampening, "nesterov": nesterov, "maximize": maximize}
    if __guard_0_for_sgd(L):
        return __transformed_code_0_for_sgd(params, d_p_list, momentum_buffer_list, has_sparse_grad, foreach, weight_decay, momentum, lr, dampening, nesterov, maximize)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return sgd(params, d_p_list, momentum_buffer_list, has_sparse_grad, foreach, weight_decay, momentum, lr, dampening, nesterov, maximize)

#============ end of sgd ============#
