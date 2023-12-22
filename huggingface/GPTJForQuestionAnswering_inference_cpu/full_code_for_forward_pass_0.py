
def __guard_0_for_forward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['mod'], 139906578003712)) \
        and (L['mod'].training == False) \
        and (___check_type_id(L['self'], 148486576)) \
        and (___check_type_id(L['inputs'], 7638432)) \
        and (set(L['inputs'].keys()) == {'end_positions', 'start_positions', 'input_ids'}) \
        and (___check_obj_id(L['self'].autocast, 17202768)) \
        and (hasattr(L['inputs']['input_ids'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['inputs']['end_positions'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['inputs']['start_positions'], '_dynamo_dynamic_indices') == False) \
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
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139903300083216))) \
        and (___compile_config_hash() == '2ccf6e38d1d882186643b44a7260a9ea') \
        and (___check_type_id(G['__import_transformers_dot_activations'].math.pi, 7644160)) \
        and (G['__import_transformers_dot_activations'].math.pi == 3.141592653589793) \
        and (___check_obj_id(G['__import_transformers_dot_utils_dot_import_utils']._torch_available, 7677664)) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_obj_id(G['__import_transformers_dot_utils_dot_import_utils']._torch_fx_available, 7677664)) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_transformers_dot_models_dot_gptj_dot_modeling_gptj'].torch.long, 139908561983232)) \
        and (G['__import_transformers_dot_models_dot_gptj_dot_modeling_gptj'].torch.long == torch.int64) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_transformers_dot_models_dot_gptj_dot_modeling_gptj'].torch.float32, 139908561983232)) \
        and (G['__import_transformers_dot_models_dot_gptj_dot_modeling_gptj'].torch.float32 == torch.float32) \
        and (___check_obj_id(L['mod'].transformer.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.forward.__defaults__[8], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.forward.__defaults__[9], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.forward.__defaults__[10], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[0].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[1].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[1].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[1].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[1].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[1].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[1].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[2].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[2].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[2].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[2].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[2].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[2].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[3].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[3].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[3].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[3].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[3].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[3].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[4].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[4].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[4].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[4].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[4].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[4].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[5].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[5].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[5].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[5].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[5].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[5].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[6].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[6].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[6].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[6].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[6].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[6].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[7].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[7].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[7].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[7].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[7].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[7].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[8].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[8].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[8].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[8].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[8].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[8].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[9].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[9].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[9].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[9].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[9].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[9].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.get_head_mask.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[10].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[10].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[10].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[10].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[10].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[10].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[11].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[11].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[11].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[11].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[11].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[11].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[12].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[12].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[12].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[12].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[12].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[12].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[13].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[13].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[13].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[13].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[13].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[13].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[14].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[14].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[14].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[14].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[14].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[14].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[15].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[15].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[15].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[15].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[15].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[15].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[16].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[16].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[16].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[16].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[16].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[16].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[17].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[17].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[17].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[17].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[17].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[17].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[18].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[18].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[18].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[18].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[18].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[18].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[19].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[19].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[19].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[19].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[19].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[19].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[20].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[20].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[20].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[20].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[20].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[20].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[21].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[21].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[21].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[21].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[21].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[21].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[22].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[22].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[22].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[22].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[22].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[22].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[23].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[23].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[23].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[23].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[23].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[23].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[24].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[24].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[24].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[24].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[24].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[24].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[25].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[25].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[25].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[25].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[25].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[25].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[26].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[26].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[26].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[26].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[26].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[26].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[27].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[27].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[27].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[27].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[27].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[27].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[0].attn._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[0].attn._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[1].attn._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[1].attn._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[2].attn._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[2].attn._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[3].attn._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[3].attn._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[4].attn._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[4].attn._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[5].attn._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[5].attn._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[6].attn._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[6].attn._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[7].attn._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[7].attn._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[8].attn._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[8].attn._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[9].attn._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[9].attn._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[10].attn._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[10].attn._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[11].attn._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[11].attn._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[12].attn._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[12].attn._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[13].attn._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[13].attn._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[14].attn._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[14].attn._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[15].attn._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[15].attn._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[16].attn._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[16].attn._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[17].attn._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[17].attn._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[18].attn._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[18].attn._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[19].attn._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[19].attn._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[20].attn._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[20].attn._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[21].attn._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[21].attn._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[22].attn._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[22].attn._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[23].attn._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[23].attn._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[24].attn._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[24].attn._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[25].attn._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[25].attn._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[26].attn._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[26].attn._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[27].attn._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[27].attn._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[0].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[0].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[0].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[0].attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[0].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[0].attn.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[1].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[1].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[1].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[1].attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[1].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[1].attn.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[2].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[2].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[2].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[2].attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[2].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[2].attn.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[3].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[3].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[3].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[3].attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[3].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[3].attn.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[4].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[4].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[4].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[4].attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[4].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[4].attn.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[5].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[5].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[5].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[5].attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[5].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[5].attn.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[6].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[6].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[6].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[6].attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[6].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[6].attn.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[7].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[7].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[7].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[7].attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[7].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[7].attn.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[8].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[8].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[8].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[8].attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[8].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[8].attn.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[9].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[9].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[9].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[9].attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[9].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[9].attn.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[10].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[10].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[10].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[10].attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[10].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[10].attn.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[11].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[11].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[11].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[11].attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[11].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[11].attn.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[12].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[12].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[12].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[12].attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[12].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[12].attn.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[13].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[13].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[13].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[13].attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[13].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[13].attn.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[14].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[14].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[14].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[14].attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[14].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[14].attn.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[15].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[15].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[15].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[15].attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[15].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[15].attn.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[16].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[16].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[16].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[16].attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[16].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[16].attn.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[17].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[17].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[17].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[17].attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[17].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[17].attn.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[18].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[18].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[18].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[18].attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[18].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[18].attn.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[19].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[19].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[19].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[19].attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[19].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[19].attn.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[20].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[20].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[20].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[20].attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[20].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[20].attn.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[21].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[21].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[21].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[21].attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[21].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[21].attn.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[22].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[22].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[22].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[22].attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[22].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[22].attn.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[23].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[23].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[23].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[23].attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[23].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[23].attn.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[24].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[24].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[24].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[24].attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[24].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[24].attn.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[25].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[25].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[25].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[25].attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[25].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[25].attn.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[26].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[26].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[26].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[26].attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[26].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[26].attn.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[27].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[27].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[27].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[27].attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[27].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[27].attn.forward.__defaults__[5], 7677632)) \
        and (___check_tensors(L['inputs']['input_ids'], L['inputs']['end_positions'], L['inputs']['start_positions'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_0*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_0(*args, **kwargs):
    pass

def __transformed_code_0_for_forward_pass(self, mod, inputs, collect_outputs):
    graph_out_0 = __compiled_fn_0(inputs['input_ids'], inputs['start_positions'
        ], inputs['end_positions'])
    import importlib
    return importlib.import_module('transformers.modeling_outputs'
        ).QuestionAnsweringModelOutput(loss=graph_out_0[0], start_logits=
        graph_out_0[1], end_logits=graph_out_0[2], hidden_states=None,
        attentions=None)


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
