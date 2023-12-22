
def __guard_0_for_forward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['mod'], 139820795969392)) \
        and (L['mod'].training == False) \
        and (___check_type_id(L['self'], 146273168)) \
        and (___check_type_id(L['inputs'], 7638432)) \
        and (set(L['inputs'].keys()) == {'decoder_input_ids', 'input_ids', 'labels'}) \
        and (___check_obj_id(L['self'].autocast, 14994496)) \
        and (hasattr(L['inputs']['labels'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['inputs']['input_ids'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['inputs']['decoder_input_ids'], '_dynamo_dynamic_indices') == False) \
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
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139817688112656))) \
        and (___compile_config_hash() == '57fc53215d971ec12cb77612ae63a007') \
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
        and (___check_type_id(G['__import_transformers_dot_models_dot_pegasus_dot_modeling_pegasus'].torch.long, 139822949725952)) \
        and (G['__import_transformers_dot_models_dot_pegasus_dot_modeling_pegasus'].torch.long == torch.int64) \
        and (___check_type_id(G['__import_transformers_dot_models_dot_pegasus_dot_modeling_pegasus'].torch.float16, 139822949725952)) \
        and (G['__import_transformers_dot_models_dot_pegasus_dot_modeling_pegasus'].torch.float16 == torch.float16) \
        and (___check_type_id(G['__import_transformers_dot_models_dot_pegasus_dot_modeling_pegasus']._make_causal_mask.__defaults__[0], 7640416)) \
        and (G['__import_transformers_dot_models_dot_pegasus_dot_modeling_pegasus']._make_causal_mask.__defaults__[0] == 0) \
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
        and (___check_type_id(L['mod'].model.decoder.embed_positions.forward.__closure__[1].cell_contents.__defaults__[0], 7640416)) \
        and (L['mod'].model.decoder.embed_positions.forward.__closure__[1].cell_contents.__defaults__[0] == 0) \
        and (___check_type_id(L['mod'].model.encoder.embed_positions.forward.__closure__[1].cell_contents.__defaults__[0], 7640416)) \
        and (L['mod'].model.encoder.embed_positions.forward.__closure__[1].cell_contents.__defaults__[0] == 0) \
        and (___check_tensors(L['inputs']['labels'], L['inputs']['input_ids'], L['inputs']['decoder_input_ids'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_0*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_0(*args, **kwargs):
    pass

def __transformed_code_0_for_forward_pass(self, mod, inputs, collect_outputs):
    graph_out_0 = __compiled_fn_0(inputs['labels'], inputs['decoder_input_ids'],
        inputs['input_ids'])
    import importlib
    return importlib.import_module('transformers.modeling_outputs'
        ).Seq2SeqLMOutput(loss=graph_out_0[0], logits=graph_out_0[1],
        past_key_values=None, decoder_hidden_states=None, decoder_attentions=
        None, cross_attentions=None, encoder_last_hidden_state=graph_out_0[2],
        encoder_hidden_states=None, encoder_attentions=None)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward_pass(self, mod, inputs, collect_outputs):
    with self.autocast() as __temp_8:
        __temp_10 = {}
        __temp_10.update(inputs)
        return mod(*(), **__temp_10)
    return None

def transformed_forward_pass(self, mod, inputs, collect_outputs):
    L = {"self": self, "mod": mod, "inputs": inputs, "collect_outputs": collect_outputs}
    if __guard_0_for_forward_pass(L):
        return __transformed_code_0_for_forward_pass(self, mod, inputs, collect_outputs)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward_pass(self, mod, inputs, collect_outputs)

#============ end of forward_pass ============#
