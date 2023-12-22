
# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_114_8(___stack0):
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

def transformed___resume_at_114_8(___stack0):
    L = {"___stack0": ___stack0}
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_114_8(___stack0)

#============ end of __resume_at_114_8 ============#

def __guard_0_for_sgd(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_type_id(L['lr'], 7644160)) \
        and (L['lr'] == 0.01) \
        and (___check_type_id(L['params'], 7642176)) \
        and (len(L['params']) == 101) \
        and (___check_obj_id(L['foreach'], 7677664)) \
        and (___check_type_id(L['d_p_list'], 7642176)) \
        and (len(L['d_p_list']) == 101) \
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
        and (hasattr(L['d_p_list'][28], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][29], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][30], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][31], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][32], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][33], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][34], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][35], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][36], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][37], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][38], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][39], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][40], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][41], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][42], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][43], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][44], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][45], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][46], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][47], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][48], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][49], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][50], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][51], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][52], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][53], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][54], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][55], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][56], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][57], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][58], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][59], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][60], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][61], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][62], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][63], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][64], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][65], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][66], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][67], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][68], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][69], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][70], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][71], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][72], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][73], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][74], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][75], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][76], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][77], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][78], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][79], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][80], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][81], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][82], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][83], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][84], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][85], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][86], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][87], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][88], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][89], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][90], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][91], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][92], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][93], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][94], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][95], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][96], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][97], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][98], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][99], '_dynamo_dynamic_indices') == False) \
        and (___check_type_id(L['weight_decay'], 7640416)) \
        and (L['weight_decay'] == 0) \
        and (hasattr(L['d_p_list'][100], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['has_sparse_grad'], 7677632)) \
        and (___check_type_id(L['momentum_buffer_list'], 7642176)) \
        and (len(L['momentum_buffer_list']) == 101) \
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
        and (___check_obj_id(L['momentum_buffer_list'][28], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][29], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][30], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][31], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][32], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][33], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][34], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][35], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][36], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][37], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][38], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][39], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][40], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][41], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][42], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][43], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][44], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][45], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][46], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][47], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][48], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][49], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][50], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][51], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][52], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][53], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][54], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][55], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][56], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][57], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][58], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][59], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][60], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][61], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][62], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][63], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][64], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][65], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][66], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][67], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][68], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][69], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][70], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][71], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][72], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][73], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][74], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][75], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][76], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][77], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][78], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][79], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][80], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][81], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][82], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][83], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][84], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][85], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][86], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][87], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][88], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][89], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][90], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][91], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][92], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][93], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][94], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][95], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][96], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][97], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][98], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][99], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][100], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139769951238848))) \
        and (___compile_config_hash() == 'b51f123df5a0db3b4d2ed885515909f1') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['params'][0], L['params'][1], L['params'][2], L['params'][3], L['params'][4], L['params'][5], L['params'][6], L['params'][7], L['params'][8], L['params'][9], L['params'][10], L['params'][11], L['params'][12], L['params'][13], L['params'][14], L['params'][15], L['params'][16], L['params'][17], L['params'][18], L['params'][19], L['params'][20], L['params'][21], L['params'][22], L['params'][23], L['params'][24], L['params'][25], L['params'][26], L['params'][27], L['params'][28], L['params'][29], L['params'][30], L['params'][31], L['params'][32], L['params'][33], L['params'][34], L['params'][35], L['params'][36], L['params'][37], L['params'][38], L['params'][39], L['params'][40], L['params'][41], L['params'][42], L['params'][43], L['params'][44], L['params'][45], L['params'][46], L['params'][47], L['params'][48], L['params'][49], L['params'][50], L['params'][51], L['params'][52], L['params'][53], L['params'][54], L['params'][55], L['params'][56], L['params'][57], L['params'][58], L['params'][59], L['params'][60], L['params'][61], L['params'][62], L['params'][63], L['params'][64], L['params'][65], L['params'][66], L['params'][67], L['params'][68], L['params'][69], L['params'][70], L['params'][71], L['params'][72], L['params'][73], L['params'][74], L['params'][75], L['params'][76], L['params'][77], L['params'][78], L['params'][79], L['params'][80], L['params'][81], L['params'][82], L['params'][83], L['params'][84], L['params'][85], L['params'][86], L['params'][87], L['params'][88], L['params'][89], L['params'][90], L['params'][91], L['params'][92], L['params'][93], L['params'][94], L['params'][95], L['params'][96], L['params'][97], L['params'][98], L['params'][99], L['d_p_list'][0], L['d_p_list'][1], L['d_p_list'][2], L['d_p_list'][3], L['d_p_list'][4], L['d_p_list'][5], L['d_p_list'][6], L['d_p_list'][7], L['d_p_list'][8], L['d_p_list'][9], L['params'][100], L['d_p_list'][10], L['d_p_list'][11], L['d_p_list'][12], L['d_p_list'][13], L['d_p_list'][14], L['d_p_list'][15], L['d_p_list'][16], L['d_p_list'][17], L['d_p_list'][18], L['d_p_list'][19], L['d_p_list'][20], L['d_p_list'][21], L['d_p_list'][22], L['d_p_list'][23], L['d_p_list'][24], L['d_p_list'][25], L['d_p_list'][26], L['d_p_list'][27], L['d_p_list'][28], L['d_p_list'][29], L['d_p_list'][30], L['d_p_list'][31], L['d_p_list'][32], L['d_p_list'][33], L['d_p_list'][34], L['d_p_list'][35], L['d_p_list'][36], L['d_p_list'][37], L['d_p_list'][38], L['d_p_list'][39], L['d_p_list'][40], L['d_p_list'][41], L['d_p_list'][42], L['d_p_list'][43], L['d_p_list'][44], L['d_p_list'][45], L['d_p_list'][46], L['d_p_list'][47], L['d_p_list'][48], L['d_p_list'][49], L['d_p_list'][50], L['d_p_list'][51], L['d_p_list'][52], L['d_p_list'][53], L['d_p_list'][54], L['d_p_list'][55], L['d_p_list'][56], L['d_p_list'][57], L['d_p_list'][58], L['d_p_list'][59], L['d_p_list'][60], L['d_p_list'][61], L['d_p_list'][62], L['d_p_list'][63], L['d_p_list'][64], L['d_p_list'][65], L['d_p_list'][66], L['d_p_list'][67], L['d_p_list'][68], L['d_p_list'][69], L['d_p_list'][70], L['d_p_list'][71], L['d_p_list'][72], L['d_p_list'][73], L['d_p_list'][74], L['d_p_list'][75], L['d_p_list'][76], L['d_p_list'][77], L['d_p_list'][78], L['d_p_list'][79], L['d_p_list'][80], L['d_p_list'][81], L['d_p_list'][82], L['d_p_list'][83], L['d_p_list'][84], L['d_p_list'][85], L['d_p_list'][86], L['d_p_list'][87], L['d_p_list'][88], L['d_p_list'][89], L['d_p_list'][90], L['d_p_list'][91], L['d_p_list'][92], L['d_p_list'][93], L['d_p_list'][94], L['d_p_list'][95], L['d_p_list'][96], L['d_p_list'][97], L['d_p_list'][98], L['d_p_list'][99], L['d_p_list'][100], tensor_check_names=tensor_check_names))

def __transformed_code_0_for_sgd(params, d_p_list, momentum_buffer_list, has_sparse_grad, foreach, weight_decay, momentum, lr, dampening, nesterov, maximize):
    _ = None; func = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    return __resume_at_114_8(_multi_tensor_sgd(params, d_p_list,
        momentum_buffer_list, weight_decay=weight_decay, momentum=momentum, lr=
        lr, dampening=dampening, nesterov=nesterov, has_sparse_grad=
        has_sparse_grad, maximize=maximize))


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def sgd(params, d_p_list, momentum_buffer_list, has_sparse_grad, foreach, weight_decay, momentum, lr, dampening, nesterov, maximize):
    if foreach is None:
        if not torch.jit.is_scripting():
            __temp_84 = _default_to_fused_or_foreach(params, differentiable=
                False, use_fused=False)
            _ = __temp_84[0]
            foreach = __temp_84[1]
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
