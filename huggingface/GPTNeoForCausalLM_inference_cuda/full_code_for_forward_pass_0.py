
def __guard_0_for_forward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['mod'], 140681978034592)) \
        and (L['mod'].training == False) \
        and (___check_type_id(L['self'], 170791520)) \
        and (___check_type_id(L['inputs'], 7638432)) \
        and (set(L['inputs'].keys()) == {'input_ids', 'labels'}) \
        and (___check_obj_id(L['self'].autocast, 39513040)) \
        and (hasattr(L['inputs']['labels'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['inputs']['input_ids'], '_dynamo_dynamic_indices') == False) \
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
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140678705307152))) \
        and (___compile_config_hash() == 'a5ff280ed748f318431b89340b5092f6') \
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
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_transformers_dot_models_dot_gpt_neo_dot_modeling_gpt_neo'].torch.long, 140683966854912)) \
        and (G['__import_transformers_dot_models_dot_gpt_neo_dot_modeling_gpt_neo'].torch.long == torch.int64) \
        and (___check_type_id(G['__import_transformers_dot_models_dot_gpt_neo_dot_modeling_gpt_neo'].torch.float32, 140683966854912)) \
        and (G['__import_transformers_dot_models_dot_gpt_neo_dot_modeling_gpt_neo'].torch.float32 == torch.float32) \
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
        and (___check_obj_id(L['mod'].transformer.h[0].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[1].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[1].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[1].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[1].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[1].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[2].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[2].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[2].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[2].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[2].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[3].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[3].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[3].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[3].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[3].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[4].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[4].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[4].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[4].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[4].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[5].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[5].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[5].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[5].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[5].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[6].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[6].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[6].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[6].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[6].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[7].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[7].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[7].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[7].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[7].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[8].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[8].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[8].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[8].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[8].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[9].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[9].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[9].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[9].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[9].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.get_head_mask.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[10].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[10].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[10].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[10].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[10].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[11].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[11].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[11].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[11].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[11].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[12].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[12].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[12].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[12].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[12].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[13].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[13].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[13].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[13].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[13].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[14].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[14].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[14].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[14].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[14].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[15].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[15].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[15].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[15].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[15].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[16].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[16].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[16].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[16].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[16].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[17].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[17].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[17].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[17].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[17].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[18].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[18].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[18].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[18].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[18].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[19].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[19].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[19].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[19].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[19].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[20].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[20].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[20].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[20].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[20].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[21].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[21].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[21].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[21].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[21].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[22].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[22].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[22].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[22].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[22].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[23].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[23].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[23].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[23].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[23].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[0].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[0].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[0].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[0].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[0].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[1].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[1].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[1].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[1].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[1].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[2].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[2].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[2].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[2].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[2].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[3].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[3].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[3].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[3].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[3].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[4].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[4].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[4].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[4].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[4].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[5].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[5].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[5].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[5].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[5].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[6].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[6].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[6].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[6].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[6].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[7].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[7].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[7].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[7].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[7].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[8].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[8].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[8].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[8].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[8].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[9].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[9].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[9].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[9].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[9].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[10].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[10].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[10].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[10].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[10].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[11].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[11].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[11].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[11].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[11].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[12].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[12].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[12].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[12].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[12].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[13].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[13].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[13].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[13].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[13].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[14].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[14].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[14].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[14].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[14].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[15].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[15].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[15].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[15].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[15].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[16].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[16].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[16].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[16].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[16].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[17].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[17].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[17].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[17].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[17].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[18].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[18].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[18].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[18].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[18].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[19].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[19].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[19].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[19].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[19].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[20].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[20].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[20].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[20].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[20].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[21].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[21].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[21].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[21].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[21].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[22].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[22].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[22].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[22].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[22].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[23].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[23].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[23].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[23].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[23].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[0].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[0].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[1].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[1].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[2].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[2].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[3].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[3].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[4].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[4].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[5].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[5].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[6].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[6].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[7].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[7].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[8].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[8].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[9].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[9].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[10].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[10].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[11].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[11].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[12].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[12].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[13].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[13].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[14].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[14].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[15].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[15].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[16].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[16].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[17].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[17].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[18].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[18].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[19].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[19].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[20].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[20].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[21].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[21].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[22].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[22].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[23].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[23].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[0].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[0].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[0].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[0].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[0].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[1].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[1].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[1].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[1].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[1].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[2].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[2].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[2].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[2].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[2].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[3].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[3].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[3].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[3].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[3].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[4].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[4].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[4].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[4].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[4].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[5].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[5].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[5].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[5].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[5].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[6].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[6].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[6].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[6].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[6].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[7].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[7].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[7].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[7].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[7].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[8].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[8].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[8].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[8].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[8].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[9].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[9].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[9].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[9].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[9].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[10].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[10].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[10].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[10].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[10].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[11].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[11].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[11].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[11].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[11].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[12].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[12].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[12].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[12].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[12].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[13].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[13].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[13].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[13].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[13].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[14].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[14].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[14].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[14].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[14].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[15].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[15].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[15].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[15].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[15].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[16].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[16].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[16].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[16].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[16].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[17].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[17].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[17].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[17].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[17].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[18].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[18].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[18].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[18].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[18].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[19].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[19].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[19].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[19].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[19].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[20].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[20].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[20].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[20].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[20].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[21].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[21].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[21].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[21].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[21].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[22].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[22].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[22].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[22].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[22].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[23].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[23].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[23].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.h[23].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.h[23].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['inputs']['labels'], L['inputs']['input_ids'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_0*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_0(*args, **kwargs):
    pass

def __transformed_code_0_for_forward_pass(self, mod, inputs, collect_outputs):
    graph_out_0 = __compiled_fn_0(inputs['input_ids'], inputs['labels'])
    import importlib
    return importlib.import_module('transformers.modeling_outputs'
        ).CausalLMOutputWithPast(loss=graph_out_0[0], logits=graph_out_0[1],
        past_key_values=((graph_out_0[2], graph_out_0[3]), (graph_out_0[4],
        graph_out_0[5]), (graph_out_0[6], graph_out_0[7]), (graph_out_0[8],
        graph_out_0[9]), (graph_out_0[10], graph_out_0[11]), (graph_out_0[12],
        graph_out_0[13]), (graph_out_0[14], graph_out_0[15]), (graph_out_0[16],
        graph_out_0[17]), (graph_out_0[18], graph_out_0[19]), (graph_out_0[20],
        graph_out_0[21]), (graph_out_0[22], graph_out_0[23]), (graph_out_0[24],
        graph_out_0[25]), (graph_out_0[26], graph_out_0[27]), (graph_out_0[28],
        graph_out_0[29]), (graph_out_0[30], graph_out_0[31]), (graph_out_0[32],
        graph_out_0[33]), (graph_out_0[34], graph_out_0[35]), (graph_out_0[36],
        graph_out_0[37]), (graph_out_0[38], graph_out_0[39]), (graph_out_0[40],
        graph_out_0[41]), (graph_out_0[42], graph_out_0[43]), (graph_out_0[44],
        graph_out_0[45]), (graph_out_0[46], graph_out_0[47]), (graph_out_0[48],
        graph_out_0[49])), hidden_states=None, attentions=None)


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
