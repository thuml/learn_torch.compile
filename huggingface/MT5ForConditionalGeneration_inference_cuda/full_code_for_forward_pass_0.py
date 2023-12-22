
def __guard_0_for_forward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['mod'], 140341152781152)) \
        and (L['mod'].training == False) \
        and (___check_type_id(L['self'], 154849904)) \
        and (___check_type_id(L['inputs'], 7638432)) \
        and (set(L['inputs'].keys()) == {'input_ids', 'decoder_input_ids', 'labels'}) \
        and (___check_obj_id(L['self'].autocast, 23563344)) \
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
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140337887813136))) \
        and (___compile_config_hash() == 'b9562eb4655b14d8d986587dc540ea4c') \
        and (___check_type_id(G['__import_transformers_dot_activations'].math.pi, 7644160)) \
        and (G['__import_transformers_dot_activations'].math.pi == 3.141592653589793) \
        and (___check_type_id(G['__import_transformers_dot_modeling_utils'].XLA_USE_BF16, 7605632)) \
        and (G['__import_transformers_dot_modeling_utils'].XLA_USE_BF16 == '0') \
        and (___check_type_id(G['__import_transformers_dot_modeling_utils'].XLA_DOWNCAST_BF16, 7605632)) \
        and (G['__import_transformers_dot_modeling_utils'].XLA_DOWNCAST_BF16 == '0') \
        and (___check_type_id(G['__import_transformers_dot_modeling_utils'].ENV_VARS_TRUE_VALUES, 7622752)) \
        and (G['__import_transformers_dot_modeling_utils'].ENV_VARS_TRUE_VALUES == {'TRUE', '1', 'ON', 'YES'}) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_transformers_dot_models_dot_mt5_dot_modeling_mt5'].torch.long, 140343149618944)) \
        and (G['__import_transformers_dot_models_dot_mt5_dot_modeling_mt5'].torch.long == torch.int64) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_transformers_dot_models_dot_mt5_dot_modeling_mt5'].torch.float16, 140343149618944)) \
        and (G['__import_transformers_dot_models_dot_mt5_dot_modeling_mt5'].torch.float16 == torch.float16) \
        and (___check_type_id(G['__import_transformers_dot_models_dot_mt5_dot_modeling_mt5'].torch.float32, 140343149618944)) \
        and (G['__import_transformers_dot_models_dot_mt5_dot_modeling_mt5'].torch.float32 == torch.float32) \
        and (___check_type_id(G['__import_transformers_dot_models_dot_mt5_dot_modeling_mt5'].torch.bfloat16, 140343149618944)) \
        and (G['__import_transformers_dot_models_dot_mt5_dot_modeling_mt5'].torch.bfloat16 == torch.bfloat16) \
        and (___check_obj_id(G['__import_transformers_dot_modeling_utils'].ModuleUtilsMixin.create_extended_attention_mask_for_decoder.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.forward.__defaults__[8], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.forward.__defaults__[9], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.forward.__defaults__[8], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.forward.__defaults__[9], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.forward.__defaults__[10], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.forward.__defaults__[11], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.forward.__defaults__[10], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.forward.__defaults__[11], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.get_head_mask.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.get_head_mask.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].forward.__defaults__[8], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[0].forward.__defaults__[9], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[1].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[1].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[1].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[1].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[1].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[1].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[1].forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[1].forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[1].forward.__defaults__[8], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[1].forward.__defaults__[9], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[2].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[2].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[2].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[2].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[2].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[2].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[2].forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[2].forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[2].forward.__defaults__[8], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[2].forward.__defaults__[9], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[3].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[3].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[3].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[3].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[3].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[3].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[3].forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[3].forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[3].forward.__defaults__[8], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[3].forward.__defaults__[9], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[4].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[4].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[4].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[4].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[4].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[4].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[4].forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[4].forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[4].forward.__defaults__[8], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[4].forward.__defaults__[9], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[5].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[5].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[5].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[5].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[5].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[5].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[5].forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[5].forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[5].forward.__defaults__[8], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[5].forward.__defaults__[9], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[6].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[6].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[6].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[6].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[6].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[6].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[6].forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[6].forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[6].forward.__defaults__[8], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[6].forward.__defaults__[9], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[7].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[7].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[7].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[7].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[7].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[7].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[7].forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[7].forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[7].forward.__defaults__[8], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[7].forward.__defaults__[9], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[0].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[0].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[0].forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[0].forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[0].forward.__defaults__[8], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[0].forward.__defaults__[9], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[1].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[1].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[1].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[1].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[1].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[1].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[1].forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[1].forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[1].forward.__defaults__[8], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[1].forward.__defaults__[9], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[2].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[2].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[2].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[2].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[2].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[2].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[2].forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[2].forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[2].forward.__defaults__[8], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[2].forward.__defaults__[9], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[3].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[3].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[3].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[3].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[3].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[3].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[3].forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[3].forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[3].forward.__defaults__[8], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[3].forward.__defaults__[9], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[4].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[4].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[4].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[4].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[4].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[4].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[4].forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[4].forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[4].forward.__defaults__[8], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[4].forward.__defaults__[9], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[5].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[5].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[5].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[5].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[5].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[5].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[5].forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[5].forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[5].forward.__defaults__[8], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[5].forward.__defaults__[9], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[6].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[6].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[6].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[6].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[6].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[6].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[6].forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[6].forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[6].forward.__defaults__[8], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[6].forward.__defaults__[9], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[7].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[7].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[7].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[7].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[7].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[7].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[7].forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[7].forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[7].forward.__defaults__[8], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[7].forward.__defaults__[9], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[0].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].decoder.block[1].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].decoder.block[2].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].decoder.block[3].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].decoder.block[4].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].decoder.block[5].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].decoder.block[6].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].decoder.block[7].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].encoder.block[0].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].encoder.block[1].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].encoder.block[2].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].encoder.block[3].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].encoder.block[4].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].encoder.block[5].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].encoder.block[6].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].encoder.block[7].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[0].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[1].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[1].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[1].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[1].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[1].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[1].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[1].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[1].layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[1].layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[1].layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[1].layer[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[1].layer[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[1].layer[0].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[1].layer[1].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[1].layer[1].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[1].layer[1].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[1].layer[1].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[1].layer[1].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[1].layer[1].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[1].layer[1].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[2].layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[2].layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[2].layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[2].layer[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[2].layer[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[2].layer[0].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[2].layer[1].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[2].layer[1].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[2].layer[1].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[2].layer[1].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[2].layer[1].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[2].layer[1].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[2].layer[1].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[3].layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[3].layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[3].layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[3].layer[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[3].layer[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[3].layer[0].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[3].layer[1].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[3].layer[1].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[3].layer[1].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[3].layer[1].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[3].layer[1].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[3].layer[1].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[3].layer[1].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[4].layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[4].layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[4].layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[4].layer[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[4].layer[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[4].layer[0].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[4].layer[1].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[4].layer[1].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[4].layer[1].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[4].layer[1].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[4].layer[1].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[4].layer[1].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[4].layer[1].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[5].layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[5].layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[5].layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[5].layer[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[5].layer[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[5].layer[0].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[5].layer[1].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[5].layer[1].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[5].layer[1].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[5].layer[1].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[5].layer[1].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[5].layer[1].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[5].layer[1].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[6].layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[6].layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[6].layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[6].layer[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[6].layer[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[6].layer[0].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[6].layer[1].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[6].layer[1].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[6].layer[1].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[6].layer[1].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[6].layer[1].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[6].layer[1].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[6].layer[1].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[7].layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[7].layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[7].layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[7].layer[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[7].layer[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[7].layer[0].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[7].layer[1].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[7].layer[1].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[7].layer[1].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[7].layer[1].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[7].layer[1].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[7].layer[1].forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[7].layer[1].forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[0].layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[0].layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[0].layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[0].layer[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[0].layer[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[0].layer[0].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[1].layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[1].layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[1].layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[1].layer[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[1].layer[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[1].layer[0].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[2].layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[2].layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[2].layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[2].layer[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[2].layer[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[2].layer[0].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[3].layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[3].layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[3].layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[3].layer[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[3].layer[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[3].layer[0].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[4].layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[4].layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[4].layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[4].layer[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[4].layer[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[4].layer[0].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[5].layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[5].layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[5].layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[5].layer[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[5].layer[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[5].layer[0].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[6].layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[6].layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[6].layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[6].layer[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[6].layer[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[6].layer[0].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[7].layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[7].layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[7].layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[7].layer[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[7].layer[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[7].layer[0].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.get_extended_attention_mask.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.get_extended_attention_mask.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.get_extended_attention_mask.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.get_extended_attention_mask.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[0].SelfAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[0].SelfAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[0].SelfAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[0].SelfAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[0].SelfAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[0].SelfAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[0].SelfAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[0].SelfAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[1].layer[0].SelfAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[1].layer[0].SelfAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[1].layer[0].SelfAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[1].layer[0].SelfAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[1].layer[0].SelfAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[1].layer[0].SelfAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[1].layer[0].SelfAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[1].layer[0].SelfAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[2].layer[0].SelfAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[2].layer[0].SelfAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[2].layer[0].SelfAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[2].layer[0].SelfAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[2].layer[0].SelfAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[2].layer[0].SelfAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[2].layer[0].SelfAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[2].layer[0].SelfAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[3].layer[0].SelfAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[3].layer[0].SelfAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[3].layer[0].SelfAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[3].layer[0].SelfAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[3].layer[0].SelfAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[3].layer[0].SelfAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[3].layer[0].SelfAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[3].layer[0].SelfAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[4].layer[0].SelfAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[4].layer[0].SelfAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[4].layer[0].SelfAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[4].layer[0].SelfAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[4].layer[0].SelfAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[4].layer[0].SelfAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[4].layer[0].SelfAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[4].layer[0].SelfAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[5].layer[0].SelfAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[5].layer[0].SelfAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[5].layer[0].SelfAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[5].layer[0].SelfAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[5].layer[0].SelfAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[5].layer[0].SelfAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[5].layer[0].SelfAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[5].layer[0].SelfAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[6].layer[0].SelfAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[6].layer[0].SelfAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[6].layer[0].SelfAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[6].layer[0].SelfAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[6].layer[0].SelfAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[6].layer[0].SelfAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[6].layer[0].SelfAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[6].layer[0].SelfAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[7].layer[0].SelfAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[7].layer[0].SelfAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[7].layer[0].SelfAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[7].layer[0].SelfAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[7].layer[0].SelfAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[7].layer[0].SelfAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[7].layer[0].SelfAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[7].layer[0].SelfAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[0].layer[0].SelfAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[0].layer[0].SelfAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[0].layer[0].SelfAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[0].layer[0].SelfAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[0].layer[0].SelfAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[0].layer[0].SelfAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[0].layer[0].SelfAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[0].layer[0].SelfAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[1].layer[0].SelfAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[1].layer[0].SelfAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[1].layer[0].SelfAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[1].layer[0].SelfAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[1].layer[0].SelfAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[1].layer[0].SelfAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[1].layer[0].SelfAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[1].layer[0].SelfAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[2].layer[0].SelfAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[2].layer[0].SelfAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[2].layer[0].SelfAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[2].layer[0].SelfAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[2].layer[0].SelfAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[2].layer[0].SelfAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[2].layer[0].SelfAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[2].layer[0].SelfAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[3].layer[0].SelfAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[3].layer[0].SelfAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[3].layer[0].SelfAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[3].layer[0].SelfAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[3].layer[0].SelfAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[3].layer[0].SelfAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[3].layer[0].SelfAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[3].layer[0].SelfAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[4].layer[0].SelfAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[4].layer[0].SelfAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[4].layer[0].SelfAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[4].layer[0].SelfAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[4].layer[0].SelfAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[4].layer[0].SelfAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[4].layer[0].SelfAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[4].layer[0].SelfAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[5].layer[0].SelfAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[5].layer[0].SelfAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[5].layer[0].SelfAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[5].layer[0].SelfAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[5].layer[0].SelfAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[5].layer[0].SelfAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[5].layer[0].SelfAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[5].layer[0].SelfAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[6].layer[0].SelfAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[6].layer[0].SelfAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[6].layer[0].SelfAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[6].layer[0].SelfAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[6].layer[0].SelfAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[6].layer[0].SelfAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[6].layer[0].SelfAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[6].layer[0].SelfAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[7].layer[0].SelfAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[7].layer[0].SelfAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[7].layer[0].SelfAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[7].layer[0].SelfAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[7].layer[0].SelfAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[7].layer[0].SelfAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[7].layer[0].SelfAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].encoder.block[7].layer[0].SelfAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[1].EncDecAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[1].EncDecAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[1].EncDecAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[1].EncDecAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[1].EncDecAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[1].EncDecAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[1].EncDecAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[1].EncDecAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[1].layer[1].EncDecAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[1].layer[1].EncDecAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[1].layer[1].EncDecAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[1].layer[1].EncDecAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[1].layer[1].EncDecAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[1].layer[1].EncDecAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[1].layer[1].EncDecAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[1].layer[1].EncDecAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[2].layer[1].EncDecAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[2].layer[1].EncDecAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[2].layer[1].EncDecAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[2].layer[1].EncDecAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[2].layer[1].EncDecAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[2].layer[1].EncDecAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[2].layer[1].EncDecAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[2].layer[1].EncDecAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[3].layer[1].EncDecAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[3].layer[1].EncDecAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[3].layer[1].EncDecAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[3].layer[1].EncDecAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[3].layer[1].EncDecAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[3].layer[1].EncDecAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[3].layer[1].EncDecAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[3].layer[1].EncDecAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[4].layer[1].EncDecAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[4].layer[1].EncDecAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[4].layer[1].EncDecAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[4].layer[1].EncDecAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[4].layer[1].EncDecAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[4].layer[1].EncDecAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[4].layer[1].EncDecAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[4].layer[1].EncDecAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[5].layer[1].EncDecAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[5].layer[1].EncDecAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[5].layer[1].EncDecAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[5].layer[1].EncDecAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[5].layer[1].EncDecAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[5].layer[1].EncDecAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[5].layer[1].EncDecAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[5].layer[1].EncDecAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[6].layer[1].EncDecAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[6].layer[1].EncDecAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[6].layer[1].EncDecAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[6].layer[1].EncDecAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[6].layer[1].EncDecAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[6].layer[1].EncDecAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[6].layer[1].EncDecAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[6].layer[1].EncDecAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[7].layer[1].EncDecAttention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[7].layer[1].EncDecAttention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[7].layer[1].EncDecAttention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[7].layer[1].EncDecAttention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[7].layer[1].EncDecAttention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[7].layer[1].EncDecAttention.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[7].layer[1].EncDecAttention.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[7].layer[1].EncDecAttention.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[0].SelfAttention.compute_bias.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].encoder.block[0].layer[0].SelfAttention.compute_bias.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].decoder.block[0].layer[0].SelfAttention._relative_position_bucket.__defaults__[0], 7677664)) \
        and (___check_type_id(L['mod'].decoder.block[0].layer[0].SelfAttention._relative_position_bucket.__defaults__[1], 7640416)) \
        and (L['mod'].decoder.block[0].layer[0].SelfAttention._relative_position_bucket.__defaults__[1] == 32) \
        and (___check_type_id(L['mod'].decoder.block[0].layer[0].SelfAttention._relative_position_bucket.__defaults__[2], 7640416)) \
        and (L['mod'].decoder.block[0].layer[0].SelfAttention._relative_position_bucket.__defaults__[2] == 128) \
        and (___check_obj_id(L['mod'].encoder.block[0].layer[0].SelfAttention._relative_position_bucket.__defaults__[0], 7677664)) \
        and (___check_type_id(L['mod'].encoder.block[0].layer[0].SelfAttention._relative_position_bucket.__defaults__[1], 7640416)) \
        and (L['mod'].encoder.block[0].layer[0].SelfAttention._relative_position_bucket.__defaults__[1] == 32) \
        and (___check_type_id(L['mod'].encoder.block[0].layer[0].SelfAttention._relative_position_bucket.__defaults__[2], 7640416)) \
        and (L['mod'].encoder.block[0].layer[0].SelfAttention._relative_position_bucket.__defaults__[2] == 128) \
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
    graph_out_0 = __compiled_fn_0(inputs['input_ids'], inputs['labels'], inputs
        ['decoder_input_ids'])
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
    tmp_72 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_73 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_74 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_75 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_76 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_77 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_78 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_79 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_80 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_81 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_82 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_83 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_84 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_85 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_86 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_87 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_88 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_89 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_90 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_91 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_92 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_93 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_94 = __import_torch_dot__dynamo_dot_utils.make_cell()
    tmp_95 = __import_torch_dot__dynamo_dot_utils.make_cell()
    import importlib
    tmp_95.cell_contents = mod.decoder.block[7].layer[1].EncDecAttention
    tmp_94.cell_contents = 1
    tmp_93.cell_contents = mod.decoder.block[7].layer[0].SelfAttention
    tmp_92.cell_contents = 1
    tmp_91.cell_contents = mod.decoder.block[6].layer[1].EncDecAttention
    tmp_90.cell_contents = 1
    tmp_89.cell_contents = mod.decoder.block[6].layer[0].SelfAttention
    tmp_88.cell_contents = 1
    tmp_87.cell_contents = mod.decoder.block[5].layer[1].EncDecAttention
    tmp_86.cell_contents = 1
    tmp_85.cell_contents = mod.decoder.block[5].layer[0].SelfAttention
    tmp_84.cell_contents = 1
    tmp_83.cell_contents = mod.decoder.block[4].layer[1].EncDecAttention
    tmp_82.cell_contents = 1
    tmp_81.cell_contents = mod.decoder.block[4].layer[0].SelfAttention
    tmp_80.cell_contents = 1
    tmp_79.cell_contents = mod.decoder.block[3].layer[1].EncDecAttention
    tmp_78.cell_contents = 1
    tmp_77.cell_contents = mod.decoder.block[3].layer[0].SelfAttention
    tmp_76.cell_contents = 1
    tmp_75.cell_contents = mod.decoder.block[2].layer[1].EncDecAttention
    tmp_74.cell_contents = 1
    tmp_73.cell_contents = mod.decoder.block[2].layer[0].SelfAttention
    tmp_72.cell_contents = 1
    tmp_71.cell_contents = mod.decoder.block[1].layer[1].EncDecAttention
    tmp_70.cell_contents = 1
    tmp_69.cell_contents = mod.decoder.block[1].layer[0].SelfAttention
    tmp_68.cell_contents = 1
    tmp_67.cell_contents = mod.decoder.block[0].layer[1].EncDecAttention
    tmp_66.cell_contents = 1
    tmp_65.cell_contents = mod.decoder.block[0].layer[0].SelfAttention
    tmp_64.cell_contents = 1
    tmp_63.cell_contents = mod.encoder.block[7].layer[0].SelfAttention
    tmp_62.cell_contents = 1
    tmp_61.cell_contents = mod.encoder.block[6].layer[0].SelfAttention
    tmp_60.cell_contents = 1
    tmp_59.cell_contents = mod.encoder.block[5].layer[0].SelfAttention
    tmp_58.cell_contents = 1
    tmp_57.cell_contents = mod.encoder.block[4].layer[0].SelfAttention
    tmp_56.cell_contents = 1
    tmp_55.cell_contents = mod.encoder.block[3].layer[0].SelfAttention
    tmp_54.cell_contents = 1
    tmp_53.cell_contents = mod.encoder.block[2].layer[0].SelfAttention
    tmp_52.cell_contents = 1
    tmp_51.cell_contents = mod.encoder.block[1].layer[0].SelfAttention
    tmp_50.cell_contents = 1
    tmp_49.cell_contents = mod.encoder.block[0].layer[0].SelfAttention
    tmp_48.cell_contents = 1
    return importlib.import_module('transformers.modeling_outputs'
        ).Seq2SeqLMOutput(loss=graph_out_0[0], logits=graph_out_0[1],
        past_key_values=((graph_out_0[2], graph_out_0[3], graph_out_0[4],
        graph_out_0[5]), (graph_out_0[6], graph_out_0[7], graph_out_0[8],
        graph_out_0[9]), (graph_out_0[10], graph_out_0[11], graph_out_0[12],
        graph_out_0[13]), (graph_out_0[14], graph_out_0[15], graph_out_0[16],
        graph_out_0[17]), (graph_out_0[18], graph_out_0[19], graph_out_0[20],
        graph_out_0[21]), (graph_out_0[22], graph_out_0[23], graph_out_0[24],
        graph_out_0[25]), (graph_out_0[26], graph_out_0[27], graph_out_0[28],
        graph_out_0[29]), (graph_out_0[30], graph_out_0[31], graph_out_0[32],
        graph_out_0[33])), decoder_hidden_states=None, decoder_attentions=None,
        cross_attentions=None, encoder_last_hidden_state=graph_out_0[34],
        encoder_hidden_states=None, encoder_attentions=None)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward_pass(self, mod, inputs, collect_outputs):
    with self.autocast() as __temp_56:
        __temp_58 = {}
        __temp_58.update(inputs)
        return mod(*(), **__temp_58)
    return None

def transformed_forward_pass(self, mod, inputs, collect_outputs):
    L = {"self": self, "mod": mod, "inputs": inputs, "collect_outputs": collect_outputs}
    if __guard_0_for_forward_pass(L):
        return __transformed_code_0_for_forward_pass(self, mod, inputs, collect_outputs)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward_pass(self, mod, inputs, collect_outputs)

#============ end of forward_pass ============#
