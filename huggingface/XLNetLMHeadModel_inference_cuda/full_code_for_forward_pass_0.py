
def __guard_0_for_forward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['mod'], 139764666236784)) \
        and (L['mod'].training == False) \
        and (___check_type_id(L['self'], 163732640)) \
        and (___check_type_id(L['inputs'], 7638432)) \
        and (set(L['inputs'].keys()) == {'labels', 'input_ids'}) \
        and (___check_obj_id(L['self'].autocast, 32443344)) \
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
        and (___check_obj_id(L['mod'].forward.__defaults__[12], 7628576)) \
        and (___check_obj_id(L['mod'].forward.__defaults__[13], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139761395949072))) \
        and (___compile_config_hash() == '16951632189bbf1fa7c3849605c0586d') \
        and (___check_type_id(G['__import_transformers_dot_modeling_utils'].XLA_USE_BF16, 7605632)) \
        and (G['__import_transformers_dot_modeling_utils'].XLA_USE_BF16 == '0') \
        and (___check_type_id(G['__import_transformers_dot_modeling_utils'].XLA_DOWNCAST_BF16, 7605632)) \
        and (G['__import_transformers_dot_modeling_utils'].XLA_DOWNCAST_BF16 == '0') \
        and (___check_type_id(G['__import_transformers_dot_modeling_utils'].ENV_VARS_TRUE_VALUES, 7622752)) \
        and (G['__import_transformers_dot_modeling_utils'].ENV_VARS_TRUE_VALUES == {'1', 'YES', 'ON', 'TRUE'}) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_transformers_dot_models_dot_xlnet_dot_modeling_xlnet'].torch.long, 139766657484544)) \
        and (G['__import_transformers_dot_models_dot_xlnet_dot_modeling_xlnet'].torch.long == torch.int64) \
        and (___check_type_id(G['__import_transformers_dot_models_dot_xlnet_dot_modeling_xlnet'].torch.float, 139766657484544)) \
        and (G['__import_transformers_dot_models_dot_xlnet_dot_modeling_xlnet'].torch.float == torch.float32) \
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
        and (___check_obj_id(L['mod'].transformer.forward.__defaults__[11], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.forward.__defaults__[12], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[0].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[1].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[1].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[1].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[1].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[2].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[2].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[2].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[2].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[3].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[3].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[3].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[3].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[4].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[4].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[4].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[4].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[5].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[5].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[5].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[5].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[6].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[6].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[6].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[6].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[7].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[7].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[7].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[7].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[8].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[8].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[8].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[8].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[9].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[9].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[9].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[9].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[10].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[10].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[10].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[10].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[11].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[11].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[11].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[11].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[12].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[12].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[12].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[12].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[13].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[13].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[13].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[13].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[14].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[14].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[14].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[14].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[15].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[15].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[15].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[15].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[16].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[16].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[16].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[16].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[17].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[17].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[17].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[17].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[18].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[18].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[18].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[18].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[19].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[19].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[19].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[19].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[20].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[20].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[20].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[20].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[21].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[21].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[21].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[21].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[22].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[22].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[22].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[22].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[23].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[23].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[23].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[23].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.positional_embedding.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[0].rel_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[0].rel_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[0].rel_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[0].rel_attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[1].rel_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[1].rel_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[1].rel_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[1].rel_attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[2].rel_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[2].rel_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[2].rel_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[2].rel_attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[3].rel_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[3].rel_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[3].rel_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[3].rel_attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[4].rel_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[4].rel_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[4].rel_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[4].rel_attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[5].rel_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[5].rel_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[5].rel_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[5].rel_attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[6].rel_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[6].rel_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[6].rel_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[6].rel_attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[7].rel_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[7].rel_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[7].rel_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[7].rel_attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[8].rel_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[8].rel_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[8].rel_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[8].rel_attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[9].rel_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[9].rel_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[9].rel_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[9].rel_attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[10].rel_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[10].rel_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[10].rel_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[10].rel_attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[11].rel_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[11].rel_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[11].rel_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[11].rel_attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[12].rel_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[12].rel_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[12].rel_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[12].rel_attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[13].rel_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[13].rel_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[13].rel_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[13].rel_attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[14].rel_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[14].rel_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[14].rel_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[14].rel_attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[15].rel_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[15].rel_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[15].rel_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[15].rel_attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[16].rel_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[16].rel_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[16].rel_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[16].rel_attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[17].rel_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[17].rel_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[17].rel_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[17].rel_attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[18].rel_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[18].rel_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[18].rel_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[18].rel_attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[19].rel_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[19].rel_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[19].rel_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[19].rel_attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[20].rel_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[20].rel_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[20].rel_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[20].rel_attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[21].rel_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[21].rel_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[21].rel_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[21].rel_attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[22].rel_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[22].rel_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[22].rel_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[22].rel_attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[23].rel_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[23].rel_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[23].rel_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[23].rel_attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.relative_positional_encoding.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[0].rel_attn.rel_attn_core.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[0].rel_attn.rel_attn_core.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[0].rel_attn.rel_attn_core.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[0].rel_attn.rel_attn_core.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[1].rel_attn.rel_attn_core.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[1].rel_attn.rel_attn_core.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[1].rel_attn.rel_attn_core.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[1].rel_attn.rel_attn_core.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[2].rel_attn.rel_attn_core.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[2].rel_attn.rel_attn_core.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[2].rel_attn.rel_attn_core.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[2].rel_attn.rel_attn_core.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[3].rel_attn.rel_attn_core.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[3].rel_attn.rel_attn_core.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[3].rel_attn.rel_attn_core.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[3].rel_attn.rel_attn_core.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[4].rel_attn.rel_attn_core.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[4].rel_attn.rel_attn_core.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[4].rel_attn.rel_attn_core.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[4].rel_attn.rel_attn_core.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[5].rel_attn.rel_attn_core.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[5].rel_attn.rel_attn_core.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[5].rel_attn.rel_attn_core.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[5].rel_attn.rel_attn_core.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[6].rel_attn.rel_attn_core.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[6].rel_attn.rel_attn_core.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[6].rel_attn.rel_attn_core.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[6].rel_attn.rel_attn_core.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[7].rel_attn.rel_attn_core.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[7].rel_attn.rel_attn_core.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[7].rel_attn.rel_attn_core.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[7].rel_attn.rel_attn_core.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[8].rel_attn.rel_attn_core.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[8].rel_attn.rel_attn_core.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[8].rel_attn.rel_attn_core.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[8].rel_attn.rel_attn_core.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[9].rel_attn.rel_attn_core.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[9].rel_attn.rel_attn_core.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[9].rel_attn.rel_attn_core.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[9].rel_attn.rel_attn_core.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[0].rel_attn.post_attention.__defaults__[0], 7677664)) \
        and (___check_type_id(L['mod'].transformer.layer[0].rel_attn.rel_shift_bnij.__defaults__[0], 7640416)) \
        and (L['mod'].transformer.layer[0].rel_attn.rel_shift_bnij.__defaults__[0] == -1) \
        and (___check_obj_id(L['mod'].transformer.layer[10].rel_attn.rel_attn_core.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[10].rel_attn.rel_attn_core.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[10].rel_attn.rel_attn_core.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[10].rel_attn.rel_attn_core.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[11].rel_attn.rel_attn_core.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[11].rel_attn.rel_attn_core.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[11].rel_attn.rel_attn_core.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[11].rel_attn.rel_attn_core.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[12].rel_attn.rel_attn_core.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[12].rel_attn.rel_attn_core.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[12].rel_attn.rel_attn_core.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[12].rel_attn.rel_attn_core.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[13].rel_attn.rel_attn_core.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[13].rel_attn.rel_attn_core.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[13].rel_attn.rel_attn_core.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[13].rel_attn.rel_attn_core.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[14].rel_attn.rel_attn_core.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[14].rel_attn.rel_attn_core.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[14].rel_attn.rel_attn_core.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[14].rel_attn.rel_attn_core.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[15].rel_attn.rel_attn_core.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[15].rel_attn.rel_attn_core.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[15].rel_attn.rel_attn_core.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[15].rel_attn.rel_attn_core.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[16].rel_attn.rel_attn_core.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[16].rel_attn.rel_attn_core.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[16].rel_attn.rel_attn_core.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[16].rel_attn.rel_attn_core.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[17].rel_attn.rel_attn_core.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[17].rel_attn.rel_attn_core.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[17].rel_attn.rel_attn_core.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[17].rel_attn.rel_attn_core.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[18].rel_attn.rel_attn_core.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[18].rel_attn.rel_attn_core.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[18].rel_attn.rel_attn_core.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[18].rel_attn.rel_attn_core.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[19].rel_attn.rel_attn_core.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[19].rel_attn.rel_attn_core.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[19].rel_attn.rel_attn_core.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[19].rel_attn.rel_attn_core.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[1].rel_attn.post_attention.__defaults__[0], 7677664)) \
        and (___check_type_id(L['mod'].transformer.layer[1].rel_attn.rel_shift_bnij.__defaults__[0], 7640416)) \
        and (L['mod'].transformer.layer[1].rel_attn.rel_shift_bnij.__defaults__[0] == -1) \
        and (___check_obj_id(L['mod'].transformer.layer[20].rel_attn.rel_attn_core.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[20].rel_attn.rel_attn_core.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[20].rel_attn.rel_attn_core.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[20].rel_attn.rel_attn_core.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[21].rel_attn.rel_attn_core.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[21].rel_attn.rel_attn_core.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[21].rel_attn.rel_attn_core.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[21].rel_attn.rel_attn_core.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[22].rel_attn.rel_attn_core.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[22].rel_attn.rel_attn_core.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[22].rel_attn.rel_attn_core.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[22].rel_attn.rel_attn_core.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[23].rel_attn.rel_attn_core.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[23].rel_attn.rel_attn_core.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[23].rel_attn.rel_attn_core.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].transformer.layer[23].rel_attn.rel_attn_core.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].transformer.layer[2].rel_attn.post_attention.__defaults__[0], 7677664)) \
        and (___check_type_id(L['mod'].transformer.layer[2].rel_attn.rel_shift_bnij.__defaults__[0], 7640416)) \
        and (L['mod'].transformer.layer[2].rel_attn.rel_shift_bnij.__defaults__[0] == -1) \
        and (___check_obj_id(L['mod'].transformer.layer[3].rel_attn.post_attention.__defaults__[0], 7677664)) \
        and (___check_type_id(L['mod'].transformer.layer[3].rel_attn.rel_shift_bnij.__defaults__[0], 7640416)) \
        and (L['mod'].transformer.layer[3].rel_attn.rel_shift_bnij.__defaults__[0] == -1) \
        and (___check_obj_id(L['mod'].transformer.layer[4].rel_attn.post_attention.__defaults__[0], 7677664)) \
        and (___check_type_id(L['mod'].transformer.layer[4].rel_attn.rel_shift_bnij.__defaults__[0], 7640416)) \
        and (L['mod'].transformer.layer[4].rel_attn.rel_shift_bnij.__defaults__[0] == -1) \
        and (___check_obj_id(L['mod'].transformer.layer[5].rel_attn.post_attention.__defaults__[0], 7677664)) \
        and (___check_type_id(L['mod'].transformer.layer[5].rel_attn.rel_shift_bnij.__defaults__[0], 7640416)) \
        and (L['mod'].transformer.layer[5].rel_attn.rel_shift_bnij.__defaults__[0] == -1) \
        and (___check_obj_id(L['mod'].transformer.layer[6].rel_attn.post_attention.__defaults__[0], 7677664)) \
        and (___check_type_id(L['mod'].transformer.layer[6].rel_attn.rel_shift_bnij.__defaults__[0], 7640416)) \
        and (L['mod'].transformer.layer[6].rel_attn.rel_shift_bnij.__defaults__[0] == -1) \
        and (___check_obj_id(L['mod'].transformer.layer[7].rel_attn.post_attention.__defaults__[0], 7677664)) \
        and (___check_type_id(L['mod'].transformer.layer[7].rel_attn.rel_shift_bnij.__defaults__[0], 7640416)) \
        and (L['mod'].transformer.layer[7].rel_attn.rel_shift_bnij.__defaults__[0] == -1) \
        and (___check_obj_id(L['mod'].transformer.layer[8].rel_attn.post_attention.__defaults__[0], 7677664)) \
        and (___check_type_id(L['mod'].transformer.layer[8].rel_attn.rel_shift_bnij.__defaults__[0], 7640416)) \
        and (L['mod'].transformer.layer[8].rel_attn.rel_shift_bnij.__defaults__[0] == -1) \
        and (___check_obj_id(L['mod'].transformer.layer[9].rel_attn.post_attention.__defaults__[0], 7677664)) \
        and (___check_type_id(L['mod'].transformer.layer[9].rel_attn.rel_shift_bnij.__defaults__[0], 7640416)) \
        and (L['mod'].transformer.layer[9].rel_attn.rel_shift_bnij.__defaults__[0] == -1) \
        and (___check_obj_id(L['mod'].transformer.layer[10].rel_attn.post_attention.__defaults__[0], 7677664)) \
        and (___check_type_id(L['mod'].transformer.layer[10].rel_attn.rel_shift_bnij.__defaults__[0], 7640416)) \
        and (L['mod'].transformer.layer[10].rel_attn.rel_shift_bnij.__defaults__[0] == -1) \
        and (___check_obj_id(L['mod'].transformer.layer[11].rel_attn.post_attention.__defaults__[0], 7677664)) \
        and (___check_type_id(L['mod'].transformer.layer[11].rel_attn.rel_shift_bnij.__defaults__[0], 7640416)) \
        and (L['mod'].transformer.layer[11].rel_attn.rel_shift_bnij.__defaults__[0] == -1) \
        and (___check_obj_id(L['mod'].transformer.layer[12].rel_attn.post_attention.__defaults__[0], 7677664)) \
        and (___check_type_id(L['mod'].transformer.layer[12].rel_attn.rel_shift_bnij.__defaults__[0], 7640416)) \
        and (L['mod'].transformer.layer[12].rel_attn.rel_shift_bnij.__defaults__[0] == -1) \
        and (___check_obj_id(L['mod'].transformer.layer[13].rel_attn.post_attention.__defaults__[0], 7677664)) \
        and (___check_type_id(L['mod'].transformer.layer[13].rel_attn.rel_shift_bnij.__defaults__[0], 7640416)) \
        and (L['mod'].transformer.layer[13].rel_attn.rel_shift_bnij.__defaults__[0] == -1) \
        and (___check_obj_id(L['mod'].transformer.layer[14].rel_attn.post_attention.__defaults__[0], 7677664)) \
        and (___check_type_id(L['mod'].transformer.layer[14].rel_attn.rel_shift_bnij.__defaults__[0], 7640416)) \
        and (L['mod'].transformer.layer[14].rel_attn.rel_shift_bnij.__defaults__[0] == -1) \
        and (___check_obj_id(L['mod'].transformer.layer[15].rel_attn.post_attention.__defaults__[0], 7677664)) \
        and (___check_type_id(L['mod'].transformer.layer[15].rel_attn.rel_shift_bnij.__defaults__[0], 7640416)) \
        and (L['mod'].transformer.layer[15].rel_attn.rel_shift_bnij.__defaults__[0] == -1) \
        and (___check_obj_id(L['mod'].transformer.layer[16].rel_attn.post_attention.__defaults__[0], 7677664)) \
        and (___check_type_id(L['mod'].transformer.layer[16].rel_attn.rel_shift_bnij.__defaults__[0], 7640416)) \
        and (L['mod'].transformer.layer[16].rel_attn.rel_shift_bnij.__defaults__[0] == -1) \
        and (___check_obj_id(L['mod'].transformer.layer[17].rel_attn.post_attention.__defaults__[0], 7677664)) \
        and (___check_type_id(L['mod'].transformer.layer[17].rel_attn.rel_shift_bnij.__defaults__[0], 7640416)) \
        and (L['mod'].transformer.layer[17].rel_attn.rel_shift_bnij.__defaults__[0] == -1) \
        and (___check_obj_id(L['mod'].transformer.layer[18].rel_attn.post_attention.__defaults__[0], 7677664)) \
        and (___check_type_id(L['mod'].transformer.layer[18].rel_attn.rel_shift_bnij.__defaults__[0], 7640416)) \
        and (L['mod'].transformer.layer[18].rel_attn.rel_shift_bnij.__defaults__[0] == -1) \
        and (___check_obj_id(L['mod'].transformer.layer[19].rel_attn.post_attention.__defaults__[0], 7677664)) \
        and (___check_type_id(L['mod'].transformer.layer[19].rel_attn.rel_shift_bnij.__defaults__[0], 7640416)) \
        and (L['mod'].transformer.layer[19].rel_attn.rel_shift_bnij.__defaults__[0] == -1) \
        and (___check_obj_id(L['mod'].transformer.layer[20].rel_attn.post_attention.__defaults__[0], 7677664)) \
        and (___check_type_id(L['mod'].transformer.layer[20].rel_attn.rel_shift_bnij.__defaults__[0], 7640416)) \
        and (L['mod'].transformer.layer[20].rel_attn.rel_shift_bnij.__defaults__[0] == -1) \
        and (___check_obj_id(L['mod'].transformer.layer[21].rel_attn.post_attention.__defaults__[0], 7677664)) \
        and (___check_type_id(L['mod'].transformer.layer[21].rel_attn.rel_shift_bnij.__defaults__[0], 7640416)) \
        and (L['mod'].transformer.layer[21].rel_attn.rel_shift_bnij.__defaults__[0] == -1) \
        and (___check_obj_id(L['mod'].transformer.layer[22].rel_attn.post_attention.__defaults__[0], 7677664)) \
        and (___check_type_id(L['mod'].transformer.layer[22].rel_attn.rel_shift_bnij.__defaults__[0], 7640416)) \
        and (L['mod'].transformer.layer[22].rel_attn.rel_shift_bnij.__defaults__[0] == -1) \
        and (___check_obj_id(L['mod'].transformer.layer[23].rel_attn.post_attention.__defaults__[0], 7677664)) \
        and (___check_type_id(L['mod'].transformer.layer[23].rel_attn.rel_shift_bnij.__defaults__[0], 7640416)) \
        and (L['mod'].transformer.layer[23].rel_attn.rel_shift_bnij.__defaults__[0] == -1) \
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
    return importlib.import_module('transformers.models.xlnet.modeling_xlnet'
        ).XLNetLMHeadModelOutput(loss=graph_out_0[0], logits=graph_out_0[1],
        mems=(graph_out_0[2], graph_out_0[3], graph_out_0[4], graph_out_0[5],
        graph_out_0[6], graph_out_0[7], graph_out_0[8], graph_out_0[9],
        graph_out_0[10], graph_out_0[11], graph_out_0[12], graph_out_0[13],
        graph_out_0[14], graph_out_0[15], graph_out_0[16], graph_out_0[17],
        graph_out_0[18], graph_out_0[19], graph_out_0[20], graph_out_0[21],
        graph_out_0[22], graph_out_0[23], graph_out_0[24], graph_out_0[25]),
        hidden_states=None, attentions=None)


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
