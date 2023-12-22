
# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_138_6(___stack0, self, mod, collect_outputs, cloned_inputs, pred, loss):
    'Failed to decompile.'

def transformed___resume_at_138_6(___stack0, self, mod, collect_outputs, cloned_inputs, pred, loss):
    L = {"___stack0": ___stack0, "self": self, "mod": mod, "collect_outputs": collect_outputs, "cloned_inputs": cloned_inputs, "pred": pred, "loss": loss}
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_138_6(___stack0, self, mod, collect_outputs, cloned_inputs, pred, loss)

#============ end of __resume_at_138_6 ============#

def __guard_2_for_resume_in_forward_and_backward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_type_id(L['self'], 113960752)) \
        and (___check_obj_id(L['___stack0'], 31202432)) \
        and (hasattr(L['___stack1'], '_dynamo_dynamic_indices') == False) \
        and (___check_type_id(L['self'].grad_scaler, 147878992)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140373239226592))) \
        and (___compile_config_hash() == '7d01ec1fdd371704fb566d1209fb8f88') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['___stack1'], tensor_check_names=tensor_check_names))

def __transformed_code_2_for_resume_in_forward_and_backward_pass(___stack0, ___stack1, self, mod, collect_outputs, cloned_inputs, pred):
    inputs = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    loss = ___stack1
    return __resume_at_138_6(___stack1.backward(), self, mod, collect_outputs,
        cloned_inputs, pred, loss)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_48_5(___stack0, ___stack1, self, mod, collect_outputs, cloned_inputs, pred):
    with ___stack0() as __temp_21:
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
        and (___check_obj_id(L['mod'], 140373238309376)) \
        and (L['mod'].training == False) \
        and (___check_type_id(L['self'], 113960752)) \
        and (___check_type_id(L['cloned_inputs'], 7642176)) \
        and (len(L['cloned_inputs']) == 2) \
        and (___check_obj_id(L['self'].autocast, 31202432)) \
        and (___check_type_id(L['cloned_inputs'][0], 92743584)) \
        and (hasattr(L['cloned_inputs'][0], '_dynamo_dynamic_indices') == False) \
        and (___check_type_id(L['cloned_inputs'][1], 92743584)) \
        and (hasattr(L['cloned_inputs'][1], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140373239226592))) \
        and (___compile_config_hash() == '7d01ec1fdd371704fb566d1209fb8f88') \
        and (not ___needs_nopython()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_transformers_dot_models_dot_bart_dot_modeling_bart'].torch.long, 140378402268928)) \
        and (G['__import_transformers_dot_models_dot_bart_dot_modeling_bart'].torch.long == torch.int64) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_transformers_dot_models_dot_bart_dot_modeling_bart'].torch.float16, 140378402268928)) \
        and (G['__import_transformers_dot_models_dot_bart_dot_modeling_bart'].torch.float16 == torch.float16) \
        and (___check_type_id(G['__import_transformers_dot_models_dot_bart_dot_modeling_bart']._make_causal_mask.__defaults__[0], 7640416)) \
        and (G['__import_transformers_dot_models_dot_bart_dot_modeling_bart']._make_causal_mask.__defaults__[0] == 0) \
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
        and (___check_obj_id(L['mod'].model.model.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.forward.__defaults__[8], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.forward.__defaults__[9], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.forward.__defaults__[10], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.forward.__defaults__[11], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.forward.__defaults__[12], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.forward.__defaults__[13], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.forward.__defaults__[14], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.forward.__defaults__[8], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.forward.__defaults__[9], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.forward.__defaults__[10], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.forward.__defaults__[11], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[0].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[0].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[0].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[0].forward.__defaults__[7], 7677664)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[1].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[1].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[1].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[1].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[1].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[1].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[1].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[1].forward.__defaults__[7], 7677664)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[2].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[2].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[2].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[2].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[2].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[2].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[2].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[2].forward.__defaults__[7], 7677664)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[3].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[3].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[3].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[3].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[3].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[3].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[3].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[3].forward.__defaults__[7], 7677664)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[4].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[4].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[4].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[4].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[4].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[4].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[4].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[4].forward.__defaults__[7], 7677664)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[5].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[5].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[5].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[5].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[5].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[5].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[5].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[5].forward.__defaults__[7], 7677664)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[0].forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[1].forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[2].forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[3].forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[4].forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[5].forward.__defaults__[0], 7677632)) \
        and (___check_type_id(L['mod'].model.model.decoder.embed_positions.forward.__defaults__[0], 7640416)) \
        and (L['mod'].model.model.decoder.embed_positions.forward.__defaults__[0] == 0) \
        and (___check_type_id(L['mod'].model.model.encoder.embed_positions.forward.__defaults__[0], 7640416)) \
        and (L['mod'].model.model.encoder.embed_positions.forward.__defaults__[0] == 0) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[0].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[0].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[0].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[0].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[0].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[1].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[1].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[1].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[1].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[1].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[2].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[2].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[2].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[2].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[2].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[3].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[3].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[3].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[3].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[3].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[4].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[4].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[4].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[4].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[4].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[5].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[5].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[5].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[5].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[5].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[0].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[0].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[0].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[0].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[0].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[1].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[1].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[1].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[1].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[1].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[2].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[2].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[2].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[2].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[2].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[3].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[3].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[3].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[3].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[3].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[4].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[4].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[4].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[4].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[4].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[5].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[5].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[5].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[5].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.encoder.layers[5].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[0].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[0].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[0].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[0].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[0].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[1].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[1].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[1].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[1].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[1].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[2].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[2].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[2].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[2].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[2].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[3].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[3].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[3].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[3].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[3].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[4].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[4].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[4].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[4].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[4].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[5].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[5].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[5].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[5].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].model.model.decoder.layers[5].encoder_attn.forward.__defaults__[4], 7677632)) \
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
    import importlib
    __temp_9 = importlib.import_module('transformers.modeling_outputs'
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
    pred = __temp_9
    ___context_manager_0_4 = __import_contextlib.nullcontext()
    ___context_manager_0_4.__enter__()
    try:
        __temp_12 = self.compute_loss(__temp_9)
    finally:
        ___context_manager_0_4.__exit__(None, None, None)
    return __resume_at_48_5(__import_contextlib.nullcontext, __temp_12, self,
        mod, collect_outputs, cloned_inputs, pred)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_20_1(___stack0, self, mod, collect_outputs, cloned_inputs):
    with self.autocast() as __temp_28:
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
        and (___check_obj_id(L['mod'], 140373238309376)) \
        and (L['mod'].training == False) \
        and (___check_type_id(L['self'], 113960752)) \
        and (___check_type_id(L['___stack0'], 7642176)) \
        and (len(L['___stack0']) == 2) \
        and (hasattr(L['___stack0'][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack0'][1], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140373239226592))) \
        and (___compile_config_hash() == '7d01ec1fdd371704fb566d1209fb8f88') \
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
    with self.autocast() as __temp_38:
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
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140373239226592))) \
        and (___compile_config_hash() == '7d01ec1fdd371704fb566d1209fb8f88') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['inputs'][0], L['inputs'][1], tensor_check_names=tensor_check_names))

def __transformed_code_0_for_forward_and_backward_pass(self, mod, inputs, collect_outputs):
    cloned_inputs = None; loss = None; pred = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    return __resume_at_6_0(clone_inputs(inputs), self, mod, collect_outputs)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward_and_backward_pass(self, mod, inputs, collect_outputs):
    cloned_inputs = clone_inputs(inputs)
    self.optimizer_zero_grad(mod)
    with self.autocast() as __temp_49:
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
