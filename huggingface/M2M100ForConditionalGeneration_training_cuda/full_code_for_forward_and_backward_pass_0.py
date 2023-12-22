
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
        and (___check_type_id(L['self'], 152711152)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140296258690576))) \
        and (___compile_config_hash() == 'd79dab715bf14823c422ba384d769dfc') \
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
        and (___check_obj_id(L['mod'], 140299537366032)) \
        and (L['mod'].training == False) \
        and (___check_type_id(L['self'], 152711152)) \
        and (___check_type_id(L['cloned_inputs'], 7638432)) \
        and (set(L['cloned_inputs'].keys()) == {'input_ids', 'decoder_input_ids', 'labels'}) \
        and (___check_obj_id(L['self'].autocast, 21425152)) \
        and (___check_type_id(L['self'].grad_scaler, 138455584)) \
        and (hasattr(L['cloned_inputs']['labels'], '_dynamo_dynamic_indices') == False) \
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
        and (___check_obj_id(L['mod'].forward.__defaults__[10], 7628576)) \
        and (___check_obj_id(L['mod'].forward.__defaults__[11], 7628576)) \
        and (___check_obj_id(L['mod'].forward.__defaults__[12], 7628576)) \
        and (___check_obj_id(L['mod'].forward.__defaults__[13], 7628576)) \
        and (___check_obj_id(L['mod'].forward.__defaults__[14], 7628576)) \
        and (___check_obj_id(L['mod'].forward.__defaults__[15], 7628576)) \
        and (hasattr(L['cloned_inputs']['decoder_input_ids'], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140296258690576))) \
        and (___compile_config_hash() == 'd79dab715bf14823c422ba384d769dfc') \
        and (not ___needs_nopython()) \
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
        and (___check_type_id(G['__import_transformers_dot_models_dot_m2m_100_dot_modeling_m2m_100'].torch.float16, 140301520504576)) \
        and (G['__import_transformers_dot_models_dot_m2m_100_dot_modeling_m2m_100'].torch.float16 == torch.float16) \
        and (___check_obj_id(G['__import_transformers_dot_integrations_dot_deepspeed']._hf_deepspeed_config_weak_ref, 7628576)) \
        and (___check_type_id(G['__import_transformers_dot_models_dot_m2m_100_dot_modeling_m2m_100']._make_causal_mask.__defaults__[0], 7640416)) \
        and (G['__import_transformers_dot_models_dot_m2m_100_dot_modeling_m2m_100']._make_causal_mask.__defaults__[0] == 0) \
        and (___check_type_id(G['__import_transformers_dot_models_dot_m2m_100_dot_modeling_m2m_100'].create_position_ids_from_input_ids.__defaults__[0], 7640416)) \
        and (G['__import_transformers_dot_models_dot_m2m_100_dot_modeling_m2m_100'].create_position_ids_from_input_ids.__defaults__[0] == 0) \
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
        and (___check_obj_id(L['mod'].model.decoder.forward.__defaults__[10], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.forward.__defaults__[11], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[0].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[0].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[0].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[0].forward.__defaults__[7], 7677664)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[1].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[1].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[1].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[1].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[1].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[1].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[1].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[1].forward.__defaults__[7], 7677664)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[2].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[2].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[2].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[2].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[2].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[2].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[2].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[2].forward.__defaults__[7], 7677664)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[3].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[3].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[3].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[3].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[3].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[3].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[3].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[3].forward.__defaults__[7], 7677664)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[4].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[4].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[4].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[4].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[4].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[4].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[4].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[4].forward.__defaults__[7], 7677664)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[5].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[5].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[5].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[5].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[5].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[5].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[5].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[5].forward.__defaults__[7], 7677664)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[6].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[6].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[6].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[6].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[6].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[6].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[6].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[6].forward.__defaults__[7], 7677664)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[7].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[7].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[7].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[7].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[7].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[7].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[7].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[7].forward.__defaults__[7], 7677664)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[8].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[8].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[8].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[8].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[8].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[8].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[8].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[8].forward.__defaults__[7], 7677664)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[9].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[9].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[9].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[9].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[9].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[9].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[9].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[9].forward.__defaults__[7], 7677664)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[0].forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[1].forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[2].forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[3].forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[4].forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[5].forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[6].forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[7].forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[8].forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[9].forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[10].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[10].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[10].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[10].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[10].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[10].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[10].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[10].forward.__defaults__[7], 7677664)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[11].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[11].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[11].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[11].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[11].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[11].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[11].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[11].forward.__defaults__[7], 7677664)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[10].forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[11].forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[0].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[0].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[0].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[0].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[0].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[1].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[1].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[1].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[1].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[1].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[2].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[2].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[2].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[2].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[2].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[3].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[3].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[3].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[3].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[3].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[4].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[4].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[4].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[4].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[4].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[5].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[5].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[5].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[5].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[5].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[6].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[6].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[6].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[6].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[6].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[7].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[7].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[7].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[7].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[7].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[8].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[8].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[8].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[8].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[8].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[9].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[9].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[9].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[9].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[9].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[0].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[0].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[0].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[0].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[0].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[1].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[1].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[1].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[1].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[1].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[2].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[2].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[2].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[2].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[2].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[3].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[3].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[3].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[3].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[3].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[4].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[4].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[4].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[4].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[4].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[5].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[5].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[5].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[5].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[5].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[6].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[6].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[6].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[6].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[6].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[7].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[7].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[7].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[7].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[7].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[8].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[8].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[8].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[8].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[8].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[9].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[9].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[9].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[9].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[9].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[10].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[10].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[10].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[10].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[10].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[11].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[11].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[11].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[11].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[11].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[10].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[10].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[10].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[10].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[10].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[11].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[11].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[11].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[11].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.layers[11].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[0].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[0].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[0].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[0].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[0].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[1].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[1].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[1].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[1].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[1].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[2].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[2].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[2].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[2].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[2].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[3].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[3].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[3].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[3].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[3].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[4].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[4].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[4].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[4].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[4].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[5].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[5].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[5].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[5].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[5].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[6].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[6].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[6].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[6].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[6].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[7].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[7].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[7].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[7].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[7].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[8].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[8].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[8].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[8].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[8].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[9].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[9].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[9].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[9].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[9].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[10].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[10].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[10].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[10].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[10].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[11].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[11].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[11].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[11].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.layers[11].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.decoder.embed_positions.forward.__closure__[1].cell_contents.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.decoder.embed_positions.forward.__closure__[1].cell_contents.__defaults__[1], 7628576)) \
        and (___check_type_id(L['mod'].model.decoder.embed_positions.forward.__closure__[1].cell_contents.__defaults__[2], 7640416)) \
        and (L['mod'].model.decoder.embed_positions.forward.__closure__[1].cell_contents.__defaults__[2] == 0) \
        and (___check_obj_id(L['mod'].model.encoder.embed_positions.forward.__closure__[1].cell_contents.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.encoder.embed_positions.forward.__closure__[1].cell_contents.__defaults__[1], 7628576)) \
        and (___check_type_id(L['mod'].model.encoder.embed_positions.forward.__closure__[1].cell_contents.__defaults__[2], 7640416)) \
        and (L['mod'].model.encoder.embed_positions.forward.__closure__[1].cell_contents.__defaults__[2] == 0) \
        and (___check_tensors(L['cloned_inputs']['labels'], L['cloned_inputs']['input_ids'], L['cloned_inputs']['decoder_input_ids'], tensor_check_names=tensor_check_names))

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
    graph_out_0 = __compiled_fn_3(cloned_inputs['labels'], cloned_inputs[
        'decoder_input_ids'], cloned_inputs['input_ids'])
    import importlib
    loss = graph_out_0[0]
    pred = importlib.import_module('transformers.modeling_outputs'
        ).Seq2SeqLMOutput(loss=graph_out_0[0], logits=graph_out_0[1],
        past_key_values=((graph_out_0[2], graph_out_0[3], graph_out_0[4],
        graph_out_0[5]), (graph_out_0[6], graph_out_0[7], graph_out_0[8],
        graph_out_0[9]), (graph_out_0[10], graph_out_0[11], graph_out_0[12],
        graph_out_0[13]), (graph_out_0[14], graph_out_0[15], graph_out_0[16],
        graph_out_0[17]), (graph_out_0[18], graph_out_0[19], graph_out_0[20],
        graph_out_0[21]), (graph_out_0[22], graph_out_0[23], graph_out_0[24],
        graph_out_0[25]), (graph_out_0[26], graph_out_0[27], graph_out_0[28],
        graph_out_0[29]), (graph_out_0[30], graph_out_0[31], graph_out_0[32],
        graph_out_0[33]), (graph_out_0[34], graph_out_0[35], graph_out_0[36],
        graph_out_0[37]), (graph_out_0[38], graph_out_0[39], graph_out_0[40],
        graph_out_0[41]), (graph_out_0[42], graph_out_0[43], graph_out_0[44],
        graph_out_0[45]), (graph_out_0[46], graph_out_0[47], graph_out_0[48],
        graph_out_0[49])), decoder_hidden_states=None, decoder_attentions=None,
        cross_attentions=None, encoder_last_hidden_state=graph_out_0[50],
        encoder_hidden_states=None, encoder_attentions=None)
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
        and (___check_obj_id(L['mod'], 140299537366032)) \
        and (L['mod'].training == False) \
        and (___check_type_id(L['self'], 152711152)) \
        and (___check_type_id(L['___stack0'], 7638432)) \
        and (set(L['___stack0'].keys()) == {'input_ids', 'decoder_input_ids', 'labels'}) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140296258690576))) \
        and (___compile_config_hash() == 'd79dab715bf14823c422ba384d769dfc') \
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
        and (set(L['inputs'].keys()) == {'input_ids', 'decoder_input_ids', 'labels'}) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140296258690576))) \
        and (___compile_config_hash() == 'd79dab715bf14823c422ba384d769dfc') \
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
