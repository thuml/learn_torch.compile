
# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_108_22(___stack0, mod, collect_outputs, cloned_inputs, pred, loss):
    'Failed to decompile.'

def transformed___resume_at_108_22(___stack0, mod, collect_outputs, cloned_inputs, pred, loss):
    L = {"___stack0": ___stack0, "mod": mod, "collect_outputs": collect_outputs, "cloned_inputs": cloned_inputs, "pred": pred, "loss": loss}
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_108_22(___stack0, mod, collect_outputs, cloned_inputs, pred, loss)

#============ end of __resume_at_108_22 ============#

def __guard_3_for_resume_in_forward_and_backward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_type_id(L['self'], 162357232)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140063927393808))) \
        and (___compile_config_hash() == 'd9da5cdf5912d92e85cf14993755c996') \
        and (not ___needs_nopython())

def __transformed_code_3_for_resume_in_forward_and_backward_pass(___stack0, self, mod, collect_outputs, cloned_inputs, pred, loss):
    inputs = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    return __resume_at_108_22(self.optimizer_step(), mod, collect_outputs,
        cloned_inputs, pred, loss)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_144_21(___stack0, self, mod, collect_outputs, cloned_inputs, pred, loss):
    'Failed to decompile.'

def transformed___resume_at_144_21(___stack0, self, mod, collect_outputs, cloned_inputs, pred, loss):
    L = {"___stack0": ___stack0, "self": self, "mod": mod, "collect_outputs": collect_outputs, "cloned_inputs": cloned_inputs, "pred": pred, "loss": loss}
    if __guard_3_for_resume_in_forward_and_backward_pass(L):
        return __transformed_code_3_for_resume_in_forward_and_backward_pass(___stack0, self, mod, collect_outputs, cloned_inputs, pred, loss)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_144_21(___stack0, self, mod, collect_outputs, cloned_inputs, pred, loss)

#============ end of __resume_at_144_21 ============#

def __guard_2_for_resume_in_forward_and_backward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_type_id(L['self'], 162357232)) \
        and (___check_obj_id(L['___stack0'], 31067200)) \
        and (___check_type_id(L['___stack1'], 148954112)) \
        and (hasattr(L['___stack1'].loss, '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].logits, '_dynamo_dynamic_indices') == False) \
        and (___check_type_id(L['self'].grad_scaler, 148080240)) \
        and (___check_obj_id(L['___stack1'].attentions, 7628576)) \
        and (___check_obj_id(L['___stack1'].hidden_states, 7628576)) \
        and (___check_type_id(L['___stack1'].past_key_values, 7617760)) \
        and (len(L['___stack1'].past_key_values) == 6) \
        and (___check_obj_id(L['___stack1'].cross_attentions, 7628576)) \
        and (___check_type_id(L['___stack1'].past_key_values[0], 7617760)) \
        and (len(L['___stack1'].past_key_values[0]) == 2) \
        and (___check_type_id(L['___stack1'].past_key_values[1], 7617760)) \
        and (len(L['___stack1'].past_key_values[1]) == 2) \
        and (___check_type_id(L['___stack1'].past_key_values[2], 7617760)) \
        and (len(L['___stack1'].past_key_values[2]) == 2) \
        and (___check_type_id(L['___stack1'].past_key_values[3], 7617760)) \
        and (len(L['___stack1'].past_key_values[3]) == 2) \
        and (___check_type_id(L['___stack1'].past_key_values[4], 7617760)) \
        and (len(L['___stack1'].past_key_values[4]) == 2) \
        and (___check_type_id(L['___stack1'].past_key_values[5], 7617760)) \
        and (len(L['___stack1'].past_key_values[5]) == 2) \
        and (hasattr(L['___stack1'].past_key_values[0][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[0][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[1][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[1][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[2][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[2][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[3][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[3][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[4][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[4][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[5][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[5][1], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140063927393808))) \
        and (___compile_config_hash() == 'd9da5cdf5912d92e85cf14993755c996') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['___stack1'].loss, L['___stack1'].logits, L['___stack1'].past_key_values[0][0], L['___stack1'].past_key_values[0][1], L['___stack1'].past_key_values[1][0], L['___stack1'].past_key_values[1][1], L['___stack1'].past_key_values[2][0], L['___stack1'].past_key_values[2][1], L['___stack1'].past_key_values[3][0], L['___stack1'].past_key_values[3][1], L['___stack1'].past_key_values[4][0], L['___stack1'].past_key_values[4][1], L['___stack1'].past_key_values[5][0], L['___stack1'].past_key_values[5][1], tensor_check_names=tensor_check_names))

def __transformed_code_2_for_resume_in_forward_and_backward_pass(___stack0, ___stack1, self, mod, collect_outputs, cloned_inputs):
    inputs = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    loss = ___stack1.loss
    pred = ___stack1
    return __resume_at_144_21(___stack1.loss.backward(), self, mod,
        collect_outputs, cloned_inputs, pred, loss)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_44_4(___stack0, ___stack1, self, mod, collect_outputs, cloned_inputs):
    with ___stack0() as __temp_55:
        pred = ___stack1
        loss = self.compute_loss(pred)
    self.grad_scaler.scale(loss).backward()
    self.optimizer_step()
    if collect_outputs:
        return collect_results(mod, pred, loss, cloned_inputs)
    return None

def transformed___resume_at_44_4(___stack0, ___stack1, self, mod, collect_outputs, cloned_inputs):
    L = {"___stack0": ___stack0, "___stack1": ___stack1, "self": self, "mod": mod, "collect_outputs": collect_outputs, "cloned_inputs": cloned_inputs}
    if __guard_2_for_resume_in_forward_and_backward_pass(L):
        return __transformed_code_2_for_resume_in_forward_and_backward_pass(___stack0, ___stack1, self, mod, collect_outputs, cloned_inputs)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_44_4(___stack0, ___stack1, self, mod, collect_outputs, cloned_inputs)

#============ end of __resume_at_44_4 ============#

def __guard_1_for_resume_in_forward_and_backward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['mod'], 140063927089664)) \
        and (L['mod'].training == True) \
        and (___check_type_id(L['self'], 162357232)) \
        and (___check_type_id(L['cloned_inputs'], 7638432)) \
        and (set(L['cloned_inputs'].keys()) == {'labels', 'input_ids'}) \
        and (___check_obj_id(L['self'].autocast, 31067200)) \
        and (hasattr(L['cloned_inputs']['input_ids'], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140063927393808))) \
        and (___compile_config_hash() == 'd9da5cdf5912d92e85cf14993755c996') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['cloned_inputs']['input_ids'], tensor_check_names=tensor_check_names))

def __transformed_code_1_for_resume_in_forward_and_backward_pass(___stack0, self, mod, collect_outputs, cloned_inputs):
    inputs = None; loss = None; pred = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    ___context_manager_0_3 = __import_contextlib.nullcontext()
    ___context_manager_0_3.__enter__()
    try:
        __temp_10 = mod(*(), **{'input_ids': cloned_inputs['input_ids'],
            'labels': cloned_inputs['labels']})
    finally:
        ___context_manager_0_3.__exit__(None, None, None)
    return __resume_at_44_4(__import_contextlib.nullcontext, __temp_10, self,
        mod, collect_outputs, cloned_inputs)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_20_1(___stack0, self, mod, collect_outputs, cloned_inputs):
    with self.autocast() as __temp_63:
        __temp_65 = {}
        __temp_65.update(cloned_inputs)
        pred = mod(*(), **__temp_65)
        loss = self.compute_loss(pred)
    self.grad_scaler.scale(loss).backward()
    self.optimizer_step()
    if collect_outputs:
        return collect_results(mod, pred, loss, cloned_inputs)
    return None

def transformed___resume_at_20_1(___stack0, self, mod, collect_outputs, cloned_inputs):
    L = {"___stack0": ___stack0, "self": self, "mod": mod, "collect_outputs": collect_outputs, "cloned_inputs": cloned_inputs}
    if __guard_1_for_resume_in_forward_and_backward_pass(L):
        return __transformed_code_1_for_resume_in_forward_and_backward_pass(___stack0, self, mod, collect_outputs, cloned_inputs)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_20_1(___stack0, self, mod, collect_outputs, cloned_inputs)

#============ end of __resume_at_20_1 ============#

def __guard_0_for_resume_in_forward_and_backward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['mod'], 140063927089664)) \
        and (L['mod'].training == True) \
        and (___check_type_id(L['self'], 162357232)) \
        and (___check_type_id(L['___stack0'], 7638432)) \
        and (set(L['___stack0'].keys()) == {'labels', 'input_ids'}) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140063927393808))) \
        and (___compile_config_hash() == 'd9da5cdf5912d92e85cf14993755c996') \
        and (not ___needs_nopython())

def __transformed_code_0_for_resume_in_forward_and_backward_pass(___stack0, self, mod, collect_outputs):
    inputs = None; loss = None; pred = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    cloned_inputs = ___stack0
    return __resume_at_20_1(self.optimizer_zero_grad(mod), self, mod,
        collect_outputs, cloned_inputs)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_6_0(___stack0, self, mod, collect_outputs):
    cloned_inputs = ___stack0
    self.optimizer_zero_grad(mod)
    with self.autocast() as __temp_75:
        __temp_77 = {}
        __temp_77.update(cloned_inputs)
        pred = mod(*(), **__temp_77)
        loss = self.compute_loss(pred)
    self.grad_scaler.scale(loss).backward()
    self.optimizer_step()
    if collect_outputs:
        return collect_results(mod, pred, loss, cloned_inputs)
    return None

def transformed___resume_at_6_0(___stack0, self, mod, collect_outputs):
    L = {"___stack0": ___stack0, "self": self, "mod": mod, "collect_outputs": collect_outputs}
    if __guard_0_for_resume_in_forward_and_backward_pass(L):
        return __transformed_code_0_for_resume_in_forward_and_backward_pass(___stack0, self, mod, collect_outputs)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_6_0(___stack0, self, mod, collect_outputs)

#============ end of __resume_at_6_0 ============#

def __guard_0_for_forward_and_backward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_type_id(L['inputs'], 7638432)) \
        and (set(L['inputs'].keys()) == {'labels', 'input_ids'}) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140063927393808))) \
        and (___compile_config_hash() == 'd9da5cdf5912d92e85cf14993755c996') \
        and (not ___needs_nopython())

def __transformed_code_0_for_forward_and_backward_pass(self, mod, inputs, collect_outputs):
    cloned_inputs = None; loss = None; pred = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    return __resume_at_6_0(clone_inputs(inputs), self, mod, collect_outputs)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward_and_backward_pass(self, mod, inputs, collect_outputs):
    cloned_inputs = clone_inputs(inputs)
    self.optimizer_zero_grad(mod)
    with self.autocast() as __temp_88:
        __temp_90 = {}
        __temp_90.update(cloned_inputs)
        pred = mod(*(), **__temp_90)
        loss = self.compute_loss(pred)
    self.grad_scaler.scale(loss).backward()
    self.optimizer_step()
    if collect_outputs:
        return collect_results(mod, pred, loss, cloned_inputs)
    return None

def transformed_forward_and_backward_pass(self, mod, inputs, collect_outputs):
    L = {"self": self, "mod": mod, "inputs": inputs, "collect_outputs": collect_outputs}
    if __guard_0_for_forward_and_backward_pass(L):
        return __transformed_code_0_for_forward_and_backward_pass(self, mod, inputs, collect_outputs)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward_and_backward_pass(self, mod, inputs, collect_outputs)

#============ end of forward_and_backward_pass ============#
