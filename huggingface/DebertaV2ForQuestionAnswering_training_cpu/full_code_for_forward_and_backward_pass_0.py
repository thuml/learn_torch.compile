
# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_100_4(___stack0, self, mod, collect_outputs, cloned_inputs, pred, loss):
    'Failed to decompile.'

def transformed___resume_at_100_4(___stack0, self, mod, collect_outputs, cloned_inputs, pred, loss):
    L = {"___stack0": ___stack0, "self": self, "mod": mod, "collect_outputs": collect_outputs, "cloned_inputs": cloned_inputs, "pred": pred, "loss": loss}
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_100_4(___stack0, self, mod, collect_outputs, cloned_inputs, pred, loss)

#============ end of __resume_at_100_4 ============#

def __guard_1_for_resume_in_forward_and_backward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['mod'], 140100625492448)) \
        and (L['mod'].training == True) \
        and (___check_type_id(L['self'], 157357120)) \
        and (___check_type_id(L['cloned_inputs'], 7638432)) \
        and (set(L['cloned_inputs'].keys()) == {'input_ids', 'end_positions', 'start_positions'}) \
        and (___check_obj_id(L['self'].autocast, 26070464)) \
        and (___check_type_id(L['self'].grad_scaler, 143096592)) \
        and (___check_type_id(L['cloned_inputs']['input_ids'], 87930384)) \
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
        and (hasattr(L['cloned_inputs']['end_positions'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['cloned_inputs']['start_positions'], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140097472618000))) \
        and (___compile_config_hash() == '145837fb36154ee059bf4595826fea03') \
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
        and (___check_type_id(G['__import_transformers_dot_models_dot_deberta_v2_dot_modeling_deberta_v2'].torch.long, 140102734735104)) \
        and (G['__import_transformers_dot_models_dot_deberta_v2_dot_modeling_deberta_v2'].torch.long == torch.int64) \
        and (___check_type_id(G['__import_transformers_dot_models_dot_deberta_v2_dot_modeling_deberta_v2'].torch.float, 140102734735104)) \
        and (G['__import_transformers_dot_models_dot_deberta_v2_dot_modeling_deberta_v2'].torch.float == torch.float32) \
        and (___check_obj_id(L['mod'].deberta.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.forward.__defaults__[0], 7677664)) \
        and (___check_obj_id(L['mod'].deberta.encoder.forward.__defaults__[1], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.forward.__defaults__[4], 7677664)) \
        and (___check_obj_id(L['mod'].deberta.embeddings.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.embeddings.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.embeddings.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.embeddings.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.embeddings.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.get_rel_pos.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.get_rel_pos.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[0].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[1].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[1].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[1].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[1].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[2].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[2].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[2].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[2].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[3].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[3].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[3].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[3].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[4].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[4].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[4].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[4].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[5].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[5].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[5].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[5].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[6].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[6].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[6].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[6].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[7].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[7].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[7].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[7].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[8].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[8].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[8].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[8].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[9].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[9].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[9].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[9].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[10].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[10].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[10].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[10].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[11].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[11].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[11].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[11].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[12].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[12].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[12].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[12].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[13].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[13].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[13].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[13].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[14].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[14].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[14].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[14].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[15].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[15].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[15].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[15].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[16].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[16].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[16].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[16].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[17].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[17].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[17].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[17].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[18].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[18].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[18].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[18].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[19].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[19].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[19].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[19].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[20].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[20].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[20].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[20].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[21].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[21].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[21].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[21].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[22].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[22].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[22].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[22].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[23].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[23].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[23].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[23].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[0].attention.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[0].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[0].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[0].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[1].attention.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[1].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[1].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[1].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[2].attention.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[2].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[2].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[2].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[3].attention.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[3].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[3].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[3].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[4].attention.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[4].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[4].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[4].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[5].attention.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[5].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[5].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[5].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[6].attention.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[6].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[6].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[6].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[7].attention.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[7].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[7].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[7].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[8].attention.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[8].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[8].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[8].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[9].attention.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[9].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[9].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[9].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[10].attention.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[10].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[10].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[10].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[11].attention.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[11].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[11].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[11].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[12].attention.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[12].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[12].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[12].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[13].attention.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[13].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[13].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[13].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[14].attention.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[14].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[14].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[14].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[15].attention.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[15].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[15].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[15].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[16].attention.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[16].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[16].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[16].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[17].attention.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[17].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[17].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[17].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[18].attention.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[18].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[18].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[18].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[19].attention.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[19].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[19].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[19].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[20].attention.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[20].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[20].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[20].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[21].attention.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[21].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[21].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[21].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[22].attention.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[22].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[22].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[22].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[23].attention.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[23].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[23].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[23].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[0].attention.self.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[0].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[0].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[0].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[1].attention.self.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[1].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[1].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[1].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[2].attention.self.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[2].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[2].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[2].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[3].attention.self.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[3].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[3].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[3].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[4].attention.self.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[4].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[4].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[4].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[5].attention.self.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[5].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[5].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[5].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[6].attention.self.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[6].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[6].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[6].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[7].attention.self.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[7].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[7].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[7].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[8].attention.self.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[8].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[8].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[8].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[9].attention.self.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[9].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[9].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[9].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[10].attention.self.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[10].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[10].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[10].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[11].attention.self.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[11].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[11].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[11].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[12].attention.self.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[12].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[12].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[12].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[13].attention.self.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[13].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[13].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[13].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[14].attention.self.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[14].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[14].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[14].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[15].attention.self.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[15].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[15].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[15].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[16].attention.self.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[16].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[16].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[16].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[17].attention.self.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[17].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[17].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[17].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[18].attention.self.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[18].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[18].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[18].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[19].attention.self.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[19].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[19].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[19].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[20].attention.self.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[20].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[20].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[20].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[21].attention.self.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[21].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[21].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[21].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[22].attention.self.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[22].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[22].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[22].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[23].attention.self.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[23].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[23].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].deberta.encoder.layer[23].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_tensors(L['cloned_inputs']['input_ids'], L['cloned_inputs']['end_positions'], L['cloned_inputs']['start_positions'], tensor_check_names=tensor_check_names))

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
        'start_positions'], cloned_inputs['end_positions'])
    import importlib
    loss = graph_out_0[0]
    pred = importlib.import_module('transformers.modeling_outputs'
        ).QuestionAnsweringModelOutput(loss=graph_out_0[0], start_logits=
        graph_out_0[1], end_logits=graph_out_0[2], hidden_states=None,
        attentions=None)
    return __resume_at_100_4(graph_out_0[0].backward(), self, mod,
        collect_outputs, cloned_inputs, pred, loss)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_20_1(___stack0, self, mod, collect_outputs, cloned_inputs):
    with self.autocast() as __temp_16:
        __temp_18 = {}
        __temp_18.update(cloned_inputs)
        pred = mod(*(), **__temp_18)
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
        and (___check_obj_id(L['mod'], 140100625492448)) \
        and (L['mod'].training == True) \
        and (___check_type_id(L['self'], 157357120)) \
        and (___check_type_id(L['___stack0'], 7638432)) \
        and (set(L['___stack0'].keys()) == {'input_ids', 'end_positions', 'start_positions'}) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140097472618000))) \
        and (___compile_config_hash() == '145837fb36154ee059bf4595826fea03') \
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
    with self.autocast() as __temp_28:
        __temp_30 = {}
        __temp_30.update(cloned_inputs)
        pred = mod(*(), **__temp_30)
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
        and (set(L['inputs'].keys()) == {'input_ids', 'end_positions', 'start_positions'}) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140097472618000))) \
        and (___compile_config_hash() == '145837fb36154ee059bf4595826fea03') \
        and (not ___needs_nopython())

def __transformed_code_0_for_forward_and_backward_pass(self, mod, inputs, collect_outputs):
    cloned_inputs = None; loss = None; pred = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    return __resume_at_6_0(clone_inputs(inputs), self, mod, collect_outputs)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward_and_backward_pass(self, mod, inputs, collect_outputs):
    cloned_inputs = clone_inputs(inputs)
    self.optimizer_zero_grad(mod)
    with self.autocast() as __temp_41:
        __temp_43 = {}
        __temp_43.update(cloned_inputs)
        pred = mod(*(), **__temp_43)
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
