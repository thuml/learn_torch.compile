
# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_144_39(___stack0, self, mod, collect_outputs, cloned_inputs, pred, loss):
    'Failed to decompile.'

def transformed___resume_at_144_39(___stack0, self, mod, collect_outputs, cloned_inputs, pred, loss):
    L = {"___stack0": ___stack0, "self": self, "mod": mod, "collect_outputs": collect_outputs, "cloned_inputs": cloned_inputs, "pred": pred, "loss": loss}
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_144_39(___stack0, self, mod, collect_outputs, cloned_inputs, pred, loss)

#============ end of __resume_at_144_39 ============#

def __guard_2_for_resume_in_forward_and_backward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_type_id(L['self'], 159262688)) \
        and (___check_obj_id(L['___stack0'], 27979072)) \
        and (___check_type_id(L['___stack1'], 145840832)) \
        and (hasattr(L['___stack1'].loss, '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].logits, '_dynamo_dynamic_indices') == False) \
        and (___check_type_id(L['self'].grad_scaler, 144967904)) \
        and (___check_obj_id(L['___stack1'].attentions, 7628576)) \
        and (___check_obj_id(L['___stack1'].hidden_states, 7628576)) \
        and (___check_type_id(L['___stack1'].past_key_values, 7617760)) \
        and (len(L['___stack1'].past_key_values) == 24) \
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
        and (___check_type_id(L['___stack1'].past_key_values[6], 7617760)) \
        and (len(L['___stack1'].past_key_values[6]) == 2) \
        and (___check_type_id(L['___stack1'].past_key_values[7], 7617760)) \
        and (len(L['___stack1'].past_key_values[7]) == 2) \
        and (___check_type_id(L['___stack1'].past_key_values[8], 7617760)) \
        and (len(L['___stack1'].past_key_values[8]) == 2) \
        and (___check_type_id(L['___stack1'].past_key_values[9], 7617760)) \
        and (len(L['___stack1'].past_key_values[9]) == 2) \
        and (___check_type_id(L['___stack1'].past_key_values[10], 7617760)) \
        and (len(L['___stack1'].past_key_values[10]) == 2) \
        and (___check_type_id(L['___stack1'].past_key_values[11], 7617760)) \
        and (len(L['___stack1'].past_key_values[11]) == 2) \
        and (___check_type_id(L['___stack1'].past_key_values[12], 7617760)) \
        and (len(L['___stack1'].past_key_values[12]) == 2) \
        and (___check_type_id(L['___stack1'].past_key_values[13], 7617760)) \
        and (len(L['___stack1'].past_key_values[13]) == 2) \
        and (___check_type_id(L['___stack1'].past_key_values[14], 7617760)) \
        and (len(L['___stack1'].past_key_values[14]) == 2) \
        and (___check_type_id(L['___stack1'].past_key_values[15], 7617760)) \
        and (len(L['___stack1'].past_key_values[15]) == 2) \
        and (___check_type_id(L['___stack1'].past_key_values[16], 7617760)) \
        and (len(L['___stack1'].past_key_values[16]) == 2) \
        and (___check_type_id(L['___stack1'].past_key_values[17], 7617760)) \
        and (len(L['___stack1'].past_key_values[17]) == 2) \
        and (___check_type_id(L['___stack1'].past_key_values[18], 7617760)) \
        and (len(L['___stack1'].past_key_values[18]) == 2) \
        and (___check_type_id(L['___stack1'].past_key_values[19], 7617760)) \
        and (len(L['___stack1'].past_key_values[19]) == 2) \
        and (___check_type_id(L['___stack1'].past_key_values[20], 7617760)) \
        and (len(L['___stack1'].past_key_values[20]) == 2) \
        and (___check_type_id(L['___stack1'].past_key_values[21], 7617760)) \
        and (len(L['___stack1'].past_key_values[21]) == 2) \
        and (___check_type_id(L['___stack1'].past_key_values[22], 7617760)) \
        and (len(L['___stack1'].past_key_values[22]) == 2) \
        and (___check_type_id(L['___stack1'].past_key_values[23], 7617760)) \
        and (len(L['___stack1'].past_key_values[23]) == 2) \
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
        and (hasattr(L['___stack1'].past_key_values[6][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[6][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[7][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[7][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[8][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[8][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[9][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[9][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[10][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[10][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[11][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[11][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[12][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[12][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[13][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[13][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[14][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[14][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[15][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[15][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[16][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[16][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[17][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[17][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[18][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[18][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[19][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[19][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[20][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[20][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[21][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[21][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[22][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[22][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[23][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack1'].past_key_values[23][1], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140087129579024))) \
        and (___compile_config_hash() == '2e7255621b8c647cf36d14f32441f8ea') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['___stack1'].loss, L['___stack1'].logits, L['___stack1'].past_key_values[0][0], L['___stack1'].past_key_values[0][1], L['___stack1'].past_key_values[1][0], L['___stack1'].past_key_values[1][1], L['___stack1'].past_key_values[2][0], L['___stack1'].past_key_values[2][1], L['___stack1'].past_key_values[3][0], L['___stack1'].past_key_values[3][1], L['___stack1'].past_key_values[4][0], L['___stack1'].past_key_values[4][1], L['___stack1'].past_key_values[5][0], L['___stack1'].past_key_values[5][1], L['___stack1'].past_key_values[6][0], L['___stack1'].past_key_values[6][1], L['___stack1'].past_key_values[7][0], L['___stack1'].past_key_values[7][1], L['___stack1'].past_key_values[8][0], L['___stack1'].past_key_values[8][1], L['___stack1'].past_key_values[9][0], L['___stack1'].past_key_values[9][1], L['___stack1'].past_key_values[10][0], L['___stack1'].past_key_values[10][1], L['___stack1'].past_key_values[11][0], L['___stack1'].past_key_values[11][1], L['___stack1'].past_key_values[12][0], L['___stack1'].past_key_values[12][1], L['___stack1'].past_key_values[13][0], L['___stack1'].past_key_values[13][1], L['___stack1'].past_key_values[14][0], L['___stack1'].past_key_values[14][1], L['___stack1'].past_key_values[15][0], L['___stack1'].past_key_values[15][1], L['___stack1'].past_key_values[16][0], L['___stack1'].past_key_values[16][1], L['___stack1'].past_key_values[17][0], L['___stack1'].past_key_values[17][1], L['___stack1'].past_key_values[18][0], L['___stack1'].past_key_values[18][1], L['___stack1'].past_key_values[19][0], L['___stack1'].past_key_values[19][1], L['___stack1'].past_key_values[20][0], L['___stack1'].past_key_values[20][1], L['___stack1'].past_key_values[21][0], L['___stack1'].past_key_values[21][1], L['___stack1'].past_key_values[22][0], L['___stack1'].past_key_values[22][1], L['___stack1'].past_key_values[23][0], L['___stack1'].past_key_values[23][1], tensor_check_names=tensor_check_names))

def __transformed_code_2_for_resume_in_forward_and_backward_pass(___stack0, ___stack1, self, mod, collect_outputs, cloned_inputs):
    inputs = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    loss = ___stack1.loss
    pred = ___stack1
    return __resume_at_144_39(___stack1.loss.backward(), self, mod,
        collect_outputs, cloned_inputs, pred, loss)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_44_4(___stack0, ___stack1, self, mod, collect_outputs, cloned_inputs):
    with ___stack0() as __temp_69:
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
        and (___check_obj_id(L['mod'], 140090398585088)) \
        and (L['mod'].training == True) \
        and (___check_type_id(L['self'], 159262688)) \
        and (___check_type_id(L['cloned_inputs'], 7638432)) \
        and (set(L['cloned_inputs'].keys()) == {'input_ids', 'labels'}) \
        and (___check_obj_id(L['self'].autocast, 27979072)) \
        and (hasattr(L['cloned_inputs']['input_ids'], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140087129579024))) \
        and (___compile_config_hash() == '2e7255621b8c647cf36d14f32441f8ea') \
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
    with self.autocast() as __temp_77:
        __temp_79 = {}
        __temp_79.update(cloned_inputs)
        pred = mod(*(), **__temp_79)
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
        and (___check_obj_id(L['mod'], 140090398585088)) \
        and (L['mod'].training == True) \
        and (___check_type_id(L['self'], 159262688)) \
        and (___check_type_id(L['___stack0'], 7638432)) \
        and (set(L['___stack0'].keys()) == {'input_ids', 'labels'}) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140087129579024))) \
        and (___compile_config_hash() == '2e7255621b8c647cf36d14f32441f8ea') \
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
    with self.autocast() as __temp_89:
        __temp_91 = {}
        __temp_91.update(cloned_inputs)
        pred = mod(*(), **__temp_91)
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
        and (set(L['inputs'].keys()) == {'input_ids', 'labels'}) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140087129579024))) \
        and (___compile_config_hash() == '2e7255621b8c647cf36d14f32441f8ea') \
        and (not ___needs_nopython())

def __transformed_code_0_for_forward_and_backward_pass(self, mod, inputs, collect_outputs):
    cloned_inputs = None; loss = None; pred = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    return __resume_at_6_0(clone_inputs(inputs), self, mod, collect_outputs)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward_and_backward_pass(self, mod, inputs, collect_outputs):
    cloned_inputs = clone_inputs(inputs)
    self.optimizer_zero_grad(mod)
    with self.autocast() as __temp_102:
        __temp_104 = {}
        __temp_104.update(cloned_inputs)
        pred = mod(*(), **__temp_104)
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
