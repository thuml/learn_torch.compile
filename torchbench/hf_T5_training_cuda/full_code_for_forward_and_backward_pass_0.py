
# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_102_7(___stack0, mod, collect_outputs, cloned_inputs, pred, loss):
    'Failed to decompile.'

def transformed___resume_at_102_7(___stack0, mod, collect_outputs, cloned_inputs, pred, loss):
    L = {"___stack0": ___stack0, "mod": mod, "collect_outputs": collect_outputs, "cloned_inputs": cloned_inputs, "pred": pred, "loss": loss}
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_102_7(___stack0, mod, collect_outputs, cloned_inputs, pred, loss)

#============ end of __resume_at_102_7 ============#

def __guard_3_for_resume_in_forward_and_backward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_type_id(L['self'], 116913984)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140086189056224))) \
        and (___compile_config_hash() == 'f2426a96de4e7a6f9de1fdbfe83f26e9') \
        and (not ___needs_nopython())

def __transformed_code_3_for_resume_in_forward_and_backward_pass(___stack0, self, mod, collect_outputs, cloned_inputs, pred, loss):
    inputs = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    return __resume_at_102_7(self.optimizer_step(), mod, collect_outputs,
        cloned_inputs, pred, loss)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_138_6(___stack0, self, mod, collect_outputs, cloned_inputs, pred, loss):
    'Failed to decompile.'

def transformed___resume_at_138_6(___stack0, self, mod, collect_outputs, cloned_inputs, pred, loss):
    L = {"___stack0": ___stack0, "self": self, "mod": mod, "collect_outputs": collect_outputs, "cloned_inputs": cloned_inputs, "pred": pred, "loss": loss}
    if __guard_3_for_resume_in_forward_and_backward_pass(L):
        return __transformed_code_3_for_resume_in_forward_and_backward_pass(___stack0, self, mod, collect_outputs, cloned_inputs, pred, loss)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_138_6(___stack0, self, mod, collect_outputs, cloned_inputs, pred, loss)

#============ end of __resume_at_138_6 ============#

def __guard_2_for_resume_in_forward_and_backward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_type_id(L['self'], 116913984)) \
        and (___check_obj_id(L['___stack0'], 34155104)) \
        and (hasattr(L['___stack1'], '_dynamo_dynamic_indices') == False) \
        and (___check_type_id(L['self'].grad_scaler, 150856400)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140086189056224))) \
        and (___compile_config_hash() == 'f2426a96de4e7a6f9de1fdbfe83f26e9') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['___stack1'], tensor_check_names=tensor_check_names))

def __transformed_code_2_for_resume_in_forward_and_backward_pass(___stack0, ___stack1, self, mod, collect_outputs, cloned_inputs, pred):
    inputs = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    loss = ___stack1
    return __resume_at_138_6(___stack1.backward(), self, mod, collect_outputs,
        cloned_inputs, pred, loss)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_48_5(___stack0, ___stack1, self, mod, collect_outputs, cloned_inputs, pred):
    with ___stack0() as __temp_61:
        loss = ___stack1
    self.grad_scaler.scale(loss).backward()
    self.optimizer_step()
    if collect_outputs:
        return collect_results(mod, pred, loss, cloned_inputs)
    return None

def transformed___resume_at_48_5(___stack0, ___stack1, self, mod, collect_outputs, cloned_inputs, pred):
    L = {"___stack0": ___stack0, "___stack1": ___stack1, "self": self, "mod": mod, "collect_outputs": collect_outputs, "cloned_inputs": cloned_inputs, "pred": pred}
    if __guard_2_for_resume_in_forward_and_backward_pass(L):
        return __transformed_code_2_for_resume_in_forward_and_backward_pass(___stack0, ___stack1, self, mod, collect_outputs, cloned_inputs, pred)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_48_5(___stack0, ___stack1, self, mod, collect_outputs, cloned_inputs, pred)

#============ end of __resume_at_48_5 ============#

def __guard_1_for_resume_in_forward_and_backward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['mod'], 140089032502352)) \
        and (L['mod'].training == False) \
        and (___check_type_id(L['self'], 116913984)) \
        and (___check_type_id(L['cloned_inputs'], 7642176)) \
        and (len(L['cloned_inputs']) == 2) \
        and (___check_obj_id(L['self'].autocast, 34155104)) \
        and (hasattr(L['cloned_inputs'][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['cloned_inputs'][1], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140086189056224))) \
        and (___compile_config_hash() == 'f2426a96de4e7a6f9de1fdbfe83f26e9') \
        and (not ___needs_nopython()) \
        and (___check_type_id(G['__import_transformers_dot_modeling_utils'].XLA_USE_BF16, 7605632)) \
        and (G['__import_transformers_dot_modeling_utils'].XLA_USE_BF16 == '0') \
        and (___check_type_id(G['__import_transformers_dot_modeling_utils'].XLA_DOWNCAST_BF16, 7605632)) \
        and (G['__import_transformers_dot_modeling_utils'].XLA_DOWNCAST_BF16 == '0') \
        and (___check_type_id(G['__import_transformers_dot_modeling_utils'].ENV_VARS_TRUE_VALUES, 7622752)) \
        and (G['__import_transformers_dot_modeling_utils'].ENV_VARS_TRUE_VALUES == {'YES', 'ON', 'TRUE', '1'}) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_transformers_dot_models_dot_t5_dot_modeling_t5'].torch.long, 140091352233728)) \
        and (G['__import_transformers_dot_models_dot_t5_dot_modeling_t5'].torch.long == torch.int64) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_transformers_dot_models_dot_t5_dot_modeling_t5'].torch.float16, 140091352233728)) \
        and (G['__import_transformers_dot_models_dot_t5_dot_modeling_t5'].torch.float16 == torch.float16) \
        and (___check_type_id(G['__import_transformers_dot_models_dot_t5_dot_modeling_t5'].torch.float32, 140091352233728)) \
        and (G['__import_transformers_dot_models_dot_t5_dot_modeling_t5'].torch.float32 == torch.float32) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_transformers_dot_models_dot_t5_dot_modeling_t5'].torch.bfloat16, 140091352233728)) \
        and (G['__import_transformers_dot_models_dot_t5_dot_modeling_t5'].torch.bfloat16 == torch.bfloat16) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(G['__import_transformers_dot_modeling_utils'].ModuleUtilsMixin.create_extended_attention_mask_for_decoder.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].model.forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].model.forward.__defaults__[8], 7628576)) \
        and (___check_obj_id(L['mod'].model.forward.__defaults__[9], 7628576)) \
        and (___check_obj_id(L['mod'].model.forward.__defaults__[10], 7628576)) \
        and (___check_obj_id(L['mod'].model.forward.__defaults__[11], 7628576)) \
        and (___check_obj_id(L['mod'].model.forward.__defaults__[12], 7628576)) \
        and (___check_obj_id(L['mod'].model.forward.__defaults__[13], 7628576)) \
        and (___check_obj_id(L['mod'].model.forward.__defaults__[14], 7628576)) \
        and (___check_obj_id(L['mod'].model.forward.__defaults__[15], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.forward.__defaults__[8], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.forward.__defaults__[9], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.forward.__defaults__[8], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.forward.__defaults__[9], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.forward.__defaults__[10], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.forward.__defaults__[11], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.forward.__defaults__[10], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.forward.__defaults__[11], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.get_head_mask.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.get_head_mask.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].forward.__defaults__[8], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].forward.__defaults__[9], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].forward.__defaults__[8], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].forward.__defaults__[9], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].forward.__defaults__[8], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].forward.__defaults__[9], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].forward.__defaults__[8], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].forward.__defaults__[9], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].forward.__defaults__[8], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].forward.__defaults__[9], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].forward.__defaults__[8], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].forward.__defaults__[9], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[0].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[0].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[0].forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[0].forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[0].forward.__defaults__[8], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[0].forward.__defaults__[9], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[1].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[1].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[1].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[1].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[1].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[1].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[1].forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[1].forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[1].forward.__defaults__[8], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[1].forward.__defaults__[9], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[2].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[2].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[2].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[2].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[2].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[2].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[2].forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[2].forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[2].forward.__defaults__[8], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[2].forward.__defaults__[9], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[3].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[3].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[3].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[3].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[3].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[3].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[3].forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[3].forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[3].forward.__defaults__[8], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[3].forward.__defaults__[9], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[4].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[4].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[4].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[4].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[4].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[4].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[4].forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[4].forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[4].forward.__defaults__[8], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[4].forward.__defaults__[9], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[5].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[5].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[5].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[5].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[5].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[5].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[5].forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[5].forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[5].forward.__defaults__[8], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[5].forward.__defaults__[9], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].model.encoder.block[0].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].model.encoder.block[1].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].model.encoder.block[2].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].model.encoder.block[3].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].model.encoder.block[4].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].model.encoder.block[5].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[0].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[1].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[1].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[1].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[1].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[1].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[1].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[1].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].layer[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].layer[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].layer[0].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].layer[1].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].layer[1].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].layer[1].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].layer[1].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].layer[1].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].layer[1].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].layer[1].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].layer[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].layer[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].layer[0].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].layer[1].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].layer[1].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].layer[1].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].layer[1].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].layer[1].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].layer[1].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].layer[1].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].layer[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].layer[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].layer[0].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].layer[1].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].layer[1].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].layer[1].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].layer[1].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].layer[1].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].layer[1].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].layer[1].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].layer[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].layer[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].layer[0].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].layer[1].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].layer[1].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].layer[1].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].layer[1].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].layer[1].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].layer[1].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].layer[1].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].layer[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].layer[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].layer[0].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].layer[1].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].layer[1].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].layer[1].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].layer[1].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].layer[1].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].layer[1].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].layer[1].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[0].layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[0].layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[0].layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[0].layer[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[0].layer[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[0].layer[0].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[1].layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[1].layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[1].layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[1].layer[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[1].layer[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[1].layer[0].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[2].layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[2].layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[2].layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[2].layer[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[2].layer[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[2].layer[0].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[3].layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[3].layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[3].layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[3].layer[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[3].layer[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[3].layer[0].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[4].layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[4].layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[4].layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[4].layer[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[4].layer[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[4].layer[0].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[5].layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[5].layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[5].layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[5].layer[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[5].layer[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[5].layer[0].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.get_extended_attention_mask.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.get_extended_attention_mask.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.get_extended_attention_mask.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.get_extended_attention_mask.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[0].SelfAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[0].SelfAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[0].SelfAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[0].SelfAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[0].SelfAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[0].SelfAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[0].SelfAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[0].SelfAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].layer[0].SelfAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].layer[0].SelfAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].layer[0].SelfAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].layer[0].SelfAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].layer[0].SelfAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].layer[0].SelfAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].layer[0].SelfAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].layer[0].SelfAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].layer[0].SelfAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].layer[0].SelfAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].layer[0].SelfAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].layer[0].SelfAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].layer[0].SelfAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].layer[0].SelfAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].layer[0].SelfAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].layer[0].SelfAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].layer[0].SelfAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].layer[0].SelfAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].layer[0].SelfAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].layer[0].SelfAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].layer[0].SelfAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].layer[0].SelfAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].layer[0].SelfAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].layer[0].SelfAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].layer[0].SelfAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].layer[0].SelfAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].layer[0].SelfAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].layer[0].SelfAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].layer[0].SelfAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].layer[0].SelfAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].layer[0].SelfAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].layer[0].SelfAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].layer[0].SelfAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].layer[0].SelfAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].layer[0].SelfAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].layer[0].SelfAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].layer[0].SelfAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].layer[0].SelfAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].layer[0].SelfAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].layer[0].SelfAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[0].layer[0].SelfAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[0].layer[0].SelfAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[0].layer[0].SelfAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[0].layer[0].SelfAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[0].layer[0].SelfAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[0].layer[0].SelfAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[0].layer[0].SelfAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[0].layer[0].SelfAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[1].layer[0].SelfAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[1].layer[0].SelfAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[1].layer[0].SelfAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[1].layer[0].SelfAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[1].layer[0].SelfAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[1].layer[0].SelfAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[1].layer[0].SelfAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[1].layer[0].SelfAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[2].layer[0].SelfAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[2].layer[0].SelfAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[2].layer[0].SelfAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[2].layer[0].SelfAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[2].layer[0].SelfAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[2].layer[0].SelfAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[2].layer[0].SelfAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[2].layer[0].SelfAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[3].layer[0].SelfAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[3].layer[0].SelfAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[3].layer[0].SelfAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[3].layer[0].SelfAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[3].layer[0].SelfAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[3].layer[0].SelfAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[3].layer[0].SelfAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[3].layer[0].SelfAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[4].layer[0].SelfAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[4].layer[0].SelfAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[4].layer[0].SelfAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[4].layer[0].SelfAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[4].layer[0].SelfAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[4].layer[0].SelfAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[4].layer[0].SelfAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[4].layer[0].SelfAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[5].layer[0].SelfAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[5].layer[0].SelfAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[5].layer[0].SelfAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[5].layer[0].SelfAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[5].layer[0].SelfAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[5].layer[0].SelfAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[5].layer[0].SelfAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.block[5].layer[0].SelfAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[1].EncDecAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[1].EncDecAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[1].EncDecAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[1].EncDecAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[1].EncDecAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[1].EncDecAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[1].EncDecAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[1].EncDecAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].layer[1].EncDecAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].layer[1].EncDecAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].layer[1].EncDecAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].layer[1].EncDecAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].layer[1].EncDecAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].layer[1].EncDecAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].layer[1].EncDecAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[1].layer[1].EncDecAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].layer[1].EncDecAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].layer[1].EncDecAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].layer[1].EncDecAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].layer[1].EncDecAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].layer[1].EncDecAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].layer[1].EncDecAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].layer[1].EncDecAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[2].layer[1].EncDecAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].layer[1].EncDecAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].layer[1].EncDecAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].layer[1].EncDecAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].layer[1].EncDecAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].layer[1].EncDecAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].layer[1].EncDecAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].layer[1].EncDecAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[3].layer[1].EncDecAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].layer[1].EncDecAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].layer[1].EncDecAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].layer[1].EncDecAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].layer[1].EncDecAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].layer[1].EncDecAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].layer[1].EncDecAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].layer[1].EncDecAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[4].layer[1].EncDecAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].layer[1].EncDecAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].layer[1].EncDecAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].layer[1].EncDecAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].layer[1].EncDecAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].layer[1].EncDecAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].layer[1].EncDecAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].layer[1].EncDecAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[5].layer[1].EncDecAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[0].SelfAttention.compute_bias.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.block[0].layer[0].SelfAttention.compute_bias.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.block[0].layer[0].SelfAttention._relative_position_bucket.__defaults__[0], 7677664)) \
        and (___check_type_id(L['mod'].model.decoder.block[0].layer[0].SelfAttention._relative_position_bucket.__defaults__[1], 7640416)) \
        and (L['mod'].model.decoder.block[0].layer[0].SelfAttention._relative_position_bucket.__defaults__[1] == 32) \
        and (___check_type_id(L['mod'].model.decoder.block[0].layer[0].SelfAttention._relative_position_bucket.__defaults__[2], 7640416)) \
        and (L['mod'].model.decoder.block[0].layer[0].SelfAttention._relative_position_bucket.__defaults__[2] == 128) \
        and (___check_obj_id(L['mod'].model.encoder.block[0].layer[0].SelfAttention._relative_position_bucket.__defaults__[0], 7677664)) \
        and (___check_type_id(L['mod'].model.encoder.block[0].layer[0].SelfAttention._relative_position_bucket.__defaults__[1], 7640416)) \
        and (L['mod'].model.encoder.block[0].layer[0].SelfAttention._relative_position_bucket.__defaults__[1] == 32) \
        and (___check_type_id(L['mod'].model.encoder.block[0].layer[0].SelfAttention._relative_position_bucket.__defaults__[2], 7640416)) \
        and (L['mod'].model.encoder.block[0].layer[0].SelfAttention._relative_position_bucket.__defaults__[2] == 128) \
        and (___check_tensors(L['cloned_inputs'][0], L['cloned_inputs'][1], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_3*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_3(*args, **kwargs):
    pass

def __transformed_code_1_for_resume_in_forward_and_backward_pass(___stack0, self, mod, collect_outputs, cloned_inputs):
    inputs = None; loss = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_3(cloned_inputs[0], cloned_inputs[1])
    tmp_36 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_37 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_38 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_39 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_40 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_41 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_42 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_43 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_44 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_45 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_46 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_47 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_48 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_49 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_50 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_51 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_52 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_53 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_54 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_55 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_56 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_57 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_58 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_59 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_60 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_61 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_62 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_63 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_64 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_65 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_66 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_67 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_68 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_69 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_70 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_71 = __import_torch_dot__dynamo_dot_utils.make_cell()
    import importlib
    __temp_45 = importlib.import_module('transformers.modeling_outputs'
        ).Seq2SeqLMOutput(loss=None, logits=graph_out_0[0], past_key_values=((
        graph_out_0[1], graph_out_0[2], graph_out_0[3], graph_out_0[4]), (
        graph_out_0[5], graph_out_0[6], graph_out_0[7], graph_out_0[8]), (
        graph_out_0[9], graph_out_0[10], graph_out_0[11], graph_out_0[12]), (
        graph_out_0[13], graph_out_0[14], graph_out_0[15], graph_out_0[16]), (
        graph_out_0[17], graph_out_0[18], graph_out_0[19], graph_out_0[20]), (
        graph_out_0[21], graph_out_0[22], graph_out_0[23], graph_out_0[24])),
        decoder_hidden_states=None, decoder_attentions=None, cross_attentions=
        None, encoder_last_hidden_state=graph_out_0[25], encoder_hidden_states=
        None, encoder_attentions=None)
    tmp_71.cell_contents = mod.model.decoder.block[5].layer[1].EncDecAttention
    tmp_70.cell_contents = 4
    tmp_69.cell_contents = mod.model.decoder.block[5].layer[0].SelfAttention
    tmp_68.cell_contents = 4
    tmp_67.cell_contents = mod.model.decoder.block[4].layer[1].EncDecAttention
    tmp_66.cell_contents = 4
    tmp_65.cell_contents = mod.model.decoder.block[4].layer[0].SelfAttention
    tmp_64.cell_contents = 4
    tmp_63.cell_contents = mod.model.decoder.block[3].layer[1].EncDecAttention
    tmp_62.cell_contents = 4
    tmp_61.cell_contents = mod.model.decoder.block[3].layer[0].SelfAttention
    tmp_60.cell_contents = 4
    tmp_59.cell_contents = mod.model.decoder.block[2].layer[1].EncDecAttention
    tmp_58.cell_contents = 4
    tmp_57.cell_contents = mod.model.decoder.block[2].layer[0].SelfAttention
    tmp_56.cell_contents = 4
    tmp_55.cell_contents = mod.model.decoder.block[1].layer[1].EncDecAttention
    tmp_54.cell_contents = 4
    tmp_53.cell_contents = mod.model.decoder.block[1].layer[0].SelfAttention
    tmp_52.cell_contents = 4
    tmp_51.cell_contents = mod.model.decoder.block[0].layer[1].EncDecAttention
    tmp_50.cell_contents = 4
    tmp_49.cell_contents = mod.model.decoder.block[0].layer[0].SelfAttention
    tmp_48.cell_contents = 4
    tmp_47.cell_contents = mod.model.encoder.block[5].layer[0].SelfAttention
    tmp_46.cell_contents = 4
    tmp_45.cell_contents = mod.model.encoder.block[4].layer[0].SelfAttention
    tmp_44.cell_contents = 4
    tmp_43.cell_contents = mod.model.encoder.block[3].layer[0].SelfAttention
    tmp_42.cell_contents = 4
    tmp_41.cell_contents = mod.model.encoder.block[2].layer[0].SelfAttention
    tmp_40.cell_contents = 4
    tmp_39.cell_contents = mod.model.encoder.block[1].layer[0].SelfAttention
    tmp_38.cell_contents = 4
    tmp_37.cell_contents = mod.model.encoder.block[0].layer[0].SelfAttention
    tmp_36.cell_contents = 4
    pred = __temp_45
    ___context_manager_0_4 = __import_contextlib.nullcontext()
    ___context_manager_0_4.__enter__()
    try:
        __temp_48 = self.compute_loss(__temp_45)
    finally:
        ___context_manager_0_4.__exit__(None, None, None)
    return __resume_at_48_5(__import_contextlib.nullcontext, __temp_48, self,
        mod, collect_outputs, cloned_inputs, pred)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_20_1(___stack0, self, mod, collect_outputs, cloned_inputs):
    with self.autocast() as __temp_68:
        pred = mod(*cloned_inputs)
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
        and (___check_obj_id(L['mod'], 140089032502352)) \
        and (L['mod'].training == False) \
        and (___check_type_id(L['self'], 116913984)) \
        and (___check_type_id(L['___stack0'], 7642176)) \
        and (len(L['___stack0']) == 2) \
        and (hasattr(L['___stack0'][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack0'][1], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140086189056224))) \
        and (___compile_config_hash() == 'f2426a96de4e7a6f9de1fdbfe83f26e9') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['___stack0'][0], L['___stack0'][1], tensor_check_names=tensor_check_names))

def __transformed_code_0_for_resume_in_forward_and_backward_pass(___stack0, self, mod, collect_outputs):
    inputs = None; loss = None; pred = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    cloned_inputs = ___stack0
    return __resume_at_20_1(self.optimizer_zero_grad(mod), self, mod,
        collect_outputs, cloned_inputs)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_6_0(___stack0, self, mod, collect_outputs):
    cloned_inputs = ___stack0
    self.optimizer_zero_grad(mod)
    with self.autocast() as __temp_78:
        pred = mod(*cloned_inputs)
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
        and (___check_type_id(L['inputs'], 7642176)) \
        and (len(L['inputs']) == 2) \
        and (hasattr(L['inputs'][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['inputs'][1], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140086189056224))) \
        and (___compile_config_hash() == 'f2426a96de4e7a6f9de1fdbfe83f26e9') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['inputs'][0], L['inputs'][1], tensor_check_names=tensor_check_names))

def __transformed_code_0_for_forward_and_backward_pass(self, mod, inputs, collect_outputs):
    cloned_inputs = None; loss = None; pred = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    return __resume_at_6_0(clone_inputs(inputs), self, mod, collect_outputs)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward_and_backward_pass(self, mod, inputs, collect_outputs):
    cloned_inputs = clone_inputs(inputs)
    self.optimizer_zero_grad(mod)
    with self.autocast() as __temp_89:
        pred = mod(*cloned_inputs)
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
