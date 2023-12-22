
def __guard_0_for_forward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['mod'], 139675997861584)) \
        and (L['mod'].training == False) \
        and (___check_type_id(L['self'], 143005600)) \
        and (___check_type_id(L['inputs'], 7638432)) \
        and (set(L['inputs'].keys()) == {'labels', 'input_ids', 'decoder_input_ids'}) \
        and (___check_obj_id(L['self'].autocast, 11717840)) \
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
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139672733031952))) \
        and (___compile_config_hash() == 'b627201523f543c774e6027ed2ac1ee9') \
        and (___check_type_id(G['__import_transformers_dot_modeling_utils'].XLA_USE_BF16, 7605632)) \
        and (G['__import_transformers_dot_modeling_utils'].XLA_USE_BF16 == '0') \
        and (___check_type_id(G['__import_transformers_dot_modeling_utils'].XLA_DOWNCAST_BF16, 7605632)) \
        and (G['__import_transformers_dot_modeling_utils'].XLA_DOWNCAST_BF16 == '0') \
        and (___check_type_id(G['__import_transformers_dot_modeling_utils'].ENV_VARS_TRUE_VALUES, 7622752)) \
        and (G['__import_transformers_dot_modeling_utils'].ENV_VARS_TRUE_VALUES == {'1', 'TRUE', 'YES', 'ON'}) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_transformers_dot_models_dot_t5_dot_modeling_t5'].torch.long, 139677994841856)) \
        and (G['__import_transformers_dot_models_dot_t5_dot_modeling_t5'].torch.long == torch.int64) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_transformers_dot_models_dot_t5_dot_modeling_t5'].torch.float16, 139677994841856)) \
        and (G['__import_transformers_dot_models_dot_t5_dot_modeling_t5'].torch.float16 == torch.float16) \
        and (___check_type_id(G['__import_transformers_dot_models_dot_t5_dot_modeling_t5'].torch.float32, 139677994841856)) \
        and (G['__import_transformers_dot_models_dot_t5_dot_modeling_t5'].torch.float32 == torch.float32) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_transformers_dot_models_dot_t5_dot_modeling_t5'].torch.bfloat16, 139677994841856)) \
        and (G['__import_transformers_dot_models_dot_t5_dot_modeling_t5'].torch.bfloat16 == torch.bfloat16) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
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
        and (___check_obj_id(L['mod'].decoder.block[0].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].decoder.block[1].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].decoder.block[2].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].decoder.block[3].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].decoder.block[4].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].decoder.block[5].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].encoder.block[0].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].encoder.block[1].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].encoder.block[2].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].encoder.block[3].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].encoder.block[4].forward.__defaults__[10], 7677664)) \
        and (___check_obj_id(L['mod'].encoder.block[5].forward.__defaults__[10], 7677664)) \
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
    tmp_71.cell_contents = mod.decoder.block[5].layer[1].EncDecAttention
    tmp_70.cell_contents = 1
    tmp_69.cell_contents = mod.decoder.block[5].layer[0].SelfAttention
    tmp_68.cell_contents = 1
    tmp_67.cell_contents = mod.decoder.block[4].layer[1].EncDecAttention
    tmp_66.cell_contents = 1
    tmp_65.cell_contents = mod.decoder.block[4].layer[0].SelfAttention
    tmp_64.cell_contents = 1
    tmp_63.cell_contents = mod.decoder.block[3].layer[1].EncDecAttention
    tmp_62.cell_contents = 1
    tmp_61.cell_contents = mod.decoder.block[3].layer[0].SelfAttention
    tmp_60.cell_contents = 1
    tmp_59.cell_contents = mod.decoder.block[2].layer[1].EncDecAttention
    tmp_58.cell_contents = 1
    tmp_57.cell_contents = mod.decoder.block[2].layer[0].SelfAttention
    tmp_56.cell_contents = 1
    tmp_55.cell_contents = mod.decoder.block[1].layer[1].EncDecAttention
    tmp_54.cell_contents = 1
    tmp_53.cell_contents = mod.decoder.block[1].layer[0].SelfAttention
    tmp_52.cell_contents = 1
    tmp_51.cell_contents = mod.decoder.block[0].layer[1].EncDecAttention
    tmp_50.cell_contents = 1
    tmp_49.cell_contents = mod.decoder.block[0].layer[0].SelfAttention
    tmp_48.cell_contents = 1
    tmp_47.cell_contents = mod.encoder.block[5].layer[0].SelfAttention
    tmp_46.cell_contents = 1
    tmp_45.cell_contents = mod.encoder.block[4].layer[0].SelfAttention
    tmp_44.cell_contents = 1
    tmp_43.cell_contents = mod.encoder.block[3].layer[0].SelfAttention
    tmp_42.cell_contents = 1
    tmp_41.cell_contents = mod.encoder.block[2].layer[0].SelfAttention
    tmp_40.cell_contents = 1
    tmp_39.cell_contents = mod.encoder.block[1].layer[0].SelfAttention
    tmp_38.cell_contents = 1
    tmp_37.cell_contents = mod.encoder.block[0].layer[0].SelfAttention
    tmp_36.cell_contents = 1
    return importlib.import_module('transformers.modeling_outputs'
        ).Seq2SeqLMOutput(loss=graph_out_0[0], logits=graph_out_0[1],
        past_key_values=((graph_out_0[2], graph_out_0[3], graph_out_0[4],
        graph_out_0[5]), (graph_out_0[6], graph_out_0[7], graph_out_0[8],
        graph_out_0[9]), (graph_out_0[10], graph_out_0[11], graph_out_0[12],
        graph_out_0[13]), (graph_out_0[14], graph_out_0[15], graph_out_0[16],
        graph_out_0[17]), (graph_out_0[18], graph_out_0[19], graph_out_0[20],
        graph_out_0[21]), (graph_out_0[22], graph_out_0[23], graph_out_0[24],
        graph_out_0[25])), decoder_hidden_states=None, decoder_attentions=None,
        cross_attentions=None, encoder_last_hidden_state=graph_out_0[26],
        encoder_hidden_states=None, encoder_attentions=None)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward_pass(self, mod, inputs, collect_outputs):
    with self.autocast() as __temp_44:
        __temp_46 = {}
        __temp_46.update(inputs)
        return mod(*(), **__temp_46)
    return None

def transformed_forward_pass(self, mod, inputs, collect_outputs):
    L = {"self": self, "mod": mod, "inputs": inputs, "collect_outputs": collect_outputs}
    if __guard_0_for_forward_pass(L):
        return __transformed_code_0_for_forward_pass(self, mod, inputs, collect_outputs)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward_pass(self, mod, inputs, collect_outputs)

#============ end of forward_pass ============#
