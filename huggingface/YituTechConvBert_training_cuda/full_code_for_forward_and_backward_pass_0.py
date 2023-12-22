
# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_108_5(___stack0, mod, collect_outputs, cloned_inputs, pred, loss):
    'Failed to decompile.'

def transformed___resume_at_108_5(___stack0, mod, collect_outputs, cloned_inputs, pred, loss):
    L = {"___stack0": ___stack0, "mod": mod, "collect_outputs": collect_outputs, "cloned_inputs": cloned_inputs, "pred": pred, "loss": loss}
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_108_5(___stack0, mod, collect_outputs, cloned_inputs, pred, loss)

#============ end of __resume_at_108_5 ============#

def __guard_2_for_resume_in_forward_and_backward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_type_id(L['self'], 172495312)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139969208917520))) \
        and (___compile_config_hash() == 'd93f7af15a40318aa3e65b15e0690f99') \
        and (not ___needs_nopython())

def __transformed_code_2_for_resume_in_forward_and_backward_pass(___stack0, self, mod, collect_outputs, cloned_inputs, pred, loss):
    inputs = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    return __resume_at_108_5(self.optimizer_step(), mod, collect_outputs,
        cloned_inputs, pred, loss)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_100_4(___stack0, self, mod, collect_outputs, cloned_inputs, pred, loss):
    'Failed to decompile.'

def transformed___resume_at_100_4(___stack0, self, mod, collect_outputs, cloned_inputs, pred, loss):
    L = {"___stack0": ___stack0, "self": self, "mod": mod, "collect_outputs": collect_outputs, "cloned_inputs": cloned_inputs, "pred": pred, "loss": loss}
    if __guard_2_for_resume_in_forward_and_backward_pass(L):
        return __transformed_code_2_for_resume_in_forward_and_backward_pass(___stack0, self, mod, collect_outputs, cloned_inputs, pred, loss)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_100_4(___stack0, self, mod, collect_outputs, cloned_inputs, pred, loss)

#============ end of __resume_at_100_4 ============#

def __guard_1_for_resume_in_forward_and_backward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['mod'], 139972480399248)) \
        and (L['mod'].training == True) \
        and (___check_type_id(L['self'], 172495312)) \
        and (___check_type_id(L['cloned_inputs'], 7638432)) \
        and (set(L['cloned_inputs'].keys()) == {'input_ids', 'labels'}) \
        and (___check_obj_id(L['self'].autocast, 41204608)) \
        and (___check_type_id(L['self'].grad_scaler, 158206816)) \
        and (hasattr(L['cloned_inputs']['labels'], '_dynamo_dynamic_indices') == False) \
        and (___check_type_id(L['cloned_inputs']['input_ids'], 103065440)) \
        and (hasattr(L['cloned_inputs']['input_ids'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['mod'].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].forward.__defaults__[8], 7628576)) \
        and (___check_obj_id(L['mod'].forward.__defaults__[9], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139969208917520))) \
        and (___compile_config_hash() == 'd93f7af15a40318aa3e65b15e0690f99') \
        and (not ___needs_nopython()) \
        and (___check_type_id(G['__import_transformers_dot_activations'].ACT2FN, 158951696)) \
        and (str(G['__import_transformers_dot_activations'].ACT2FN.keys()) == "odict_keys(['gelu', 'gelu_10', 'gelu_fast', 'gelu_new', 'gelu_python', 'gelu_pytorch_tanh', 'gelu_accurate', 'laplace', 'linear', 'mish', 'quick_gelu', 'relu', 'relu2', 'relu6', 'sigmoid', 'silu', 'swish', 'tanh'])") \
        and (___check_type_id(G['__import_transformers_dot_modeling_utils'].XLA_USE_BF16, 7605632)) \
        and (G['__import_transformers_dot_modeling_utils'].XLA_USE_BF16 == '0') \
        and (___check_type_id(G['__import_transformers_dot_modeling_utils'].XLA_DOWNCAST_BF16, 7605632)) \
        and (G['__import_transformers_dot_modeling_utils'].XLA_DOWNCAST_BF16 == '0') \
        and (___check_type_id(G['__import_transformers_dot_modeling_utils'].ENV_VARS_TRUE_VALUES, 7622752)) \
        and (G['__import_transformers_dot_modeling_utils'].ENV_VARS_TRUE_VALUES == {'ON', '1', 'YES', 'TRUE'}) \
        and (___check_obj_id(G['__import_transformers_dot_utils_dot_import_utils']._torch_available, 7677664)) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_obj_id(G['__import_transformers_dot_utils_dot_import_utils']._torch_fx_available, 7677664)) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['mod'].convbert.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.forward.__defaults__[8], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.get_head_mask.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.forward.__defaults__[6], 7677664)) \
        and (___check_obj_id(L['mod'].convbert.embeddings.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.embeddings.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.embeddings.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.embeddings.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[1].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[1].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[1].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[1].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[1].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[2].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[2].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[2].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[2].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[2].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[3].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[3].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[3].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[3].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[3].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[4].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[4].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[4].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[4].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[4].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[5].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[5].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[5].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[5].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[5].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[6].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[6].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[6].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[6].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[6].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[7].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[7].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[7].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[7].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[7].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[8].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[8].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[8].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[8].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[8].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[9].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[9].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[9].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[9].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[9].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[10].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[10].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[10].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[10].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[10].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[11].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[11].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[11].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[11].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[11].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.get_extended_attention_mask.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.get_extended_attention_mask.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[0].attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[0].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[0].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[0].attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[1].attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[1].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[1].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[1].attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[2].attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[2].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[2].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[2].attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[3].attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[3].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[3].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[3].attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[4].attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[4].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[4].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[4].attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[5].attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[5].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[5].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[5].attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[6].attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[6].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[6].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[6].attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[7].attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[7].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[7].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[7].attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[8].attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[8].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[8].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[8].attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[9].attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[9].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[9].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[9].attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[10].attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[10].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[10].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[10].attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[11].attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[11].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[11].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[11].attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[0].attention.self.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[0].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[0].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[0].attention.self.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[1].attention.self.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[1].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[1].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[1].attention.self.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[2].attention.self.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[2].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[2].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[2].attention.self.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[3].attention.self.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[3].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[3].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[3].attention.self.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[4].attention.self.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[4].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[4].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[4].attention.self.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[5].attention.self.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[5].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[5].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[5].attention.self.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[6].attention.self.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[6].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[6].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[6].attention.self.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[7].attention.self.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[7].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[7].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[7].attention.self.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[8].attention.self.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[8].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[8].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[8].attention.self.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[9].attention.self.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[9].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[9].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[9].attention.self.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[10].attention.self.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[10].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[10].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[10].attention.self.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[11].attention.self.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[11].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[11].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].convbert.encoder.layer[11].attention.self.forward.__defaults__[3], 7677632)) \
        and (___check_tensors(L['cloned_inputs']['labels'], L['cloned_inputs']['input_ids'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_3*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_3(*args, **kwargs):
    pass

def __transformed_code_1_for_resume_in_forward_and_backward_pass(___stack0, self, mod, collect_outputs, cloned_inputs):
    inputs = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_3(cloned_inputs['input_ids'], cloned_inputs[
        'labels'])
    import importlib
    loss = graph_out_0[0]
    pred = importlib.import_module('transformers.modeling_outputs').MaskedLMOutput(
        loss=graph_out_0[0], logits=graph_out_0[1], hidden_states=None,
        attentions=None)
    return __resume_at_100_4(graph_out_0[0].backward(), self, mod,
        collect_outputs, cloned_inputs, pred, loss)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_20_1(___stack0, self, mod, collect_outputs, cloned_inputs):
    with self.autocast() as __temp_20:
        __temp_22 = {}
        __temp_22.update(cloned_inputs)
        pred = mod(*(), **__temp_22)
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
        and (___check_obj_id(L['mod'], 139972480399248)) \
        and (L['mod'].training == True) \
        and (___check_type_id(L['self'], 172495312)) \
        and (___check_type_id(L['___stack0'], 7638432)) \
        and (set(L['___stack0'].keys()) == {'input_ids', 'labels'}) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139969208917520))) \
        and (___compile_config_hash() == 'd93f7af15a40318aa3e65b15e0690f99') \
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
    with self.autocast() as __temp_32:
        __temp_34 = {}
        __temp_34.update(cloned_inputs)
        pred = mod(*(), **__temp_34)
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
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139969208917520))) \
        and (___compile_config_hash() == 'd93f7af15a40318aa3e65b15e0690f99') \
        and (not ___needs_nopython())

def __transformed_code_0_for_forward_and_backward_pass(self, mod, inputs, collect_outputs):
    cloned_inputs = None; loss = None; pred = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    return __resume_at_6_0(clone_inputs(inputs), self, mod, collect_outputs)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward_and_backward_pass(self, mod, inputs, collect_outputs):
    cloned_inputs = clone_inputs(inputs)
    self.optimizer_zero_grad(mod)
    with self.autocast() as __temp_45:
        __temp_47 = {}
        __temp_47.update(cloned_inputs)
        pred = mod(*(), **__temp_47)
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
