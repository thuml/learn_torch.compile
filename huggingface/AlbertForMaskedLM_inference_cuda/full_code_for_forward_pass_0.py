
def __guard_0_for_forward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['mod'], 139661541570256)) \
        and (L['mod'].training == False) \
        and (___check_type_id(L['self'], 150543616)) \
        and (___check_type_id(L['inputs'], 7638432)) \
        and (set(L['inputs'].keys()) == {'labels', 'input_ids'}) \
        and (___check_obj_id(L['self'].autocast, 19254224)) \
        and (hasattr(L['inputs']['labels'], '_dynamo_dynamic_indices') == False) \
        and (___check_type_id(L['inputs']['input_ids'], 81115056)) \
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
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139661542366736))) \
        and (___compile_config_hash() == '990cd94988e887433cfc002a149e62de') \
        and (___check_type_id(G['__import_transformers_dot_activations'].math.pi, 7644160)) \
        and (G['__import_transformers_dot_activations'].math.pi == 3.141592653589793) \
        and (___check_type_id(G['__import_transformers_dot_modeling_utils'].XLA_USE_BF16, 7605632)) \
        and (G['__import_transformers_dot_modeling_utils'].XLA_USE_BF16 == '0') \
        and (___check_type_id(G['__import_transformers_dot_modeling_utils'].XLA_DOWNCAST_BF16, 7605632)) \
        and (G['__import_transformers_dot_modeling_utils'].XLA_DOWNCAST_BF16 == '0') \
        and (___check_type_id(G['__import_transformers_dot_modeling_utils'].ENV_VARS_TRUE_VALUES, 7622752)) \
        and (G['__import_transformers_dot_modeling_utils'].ENV_VARS_TRUE_VALUES == {'YES', '1', 'ON', 'TRUE'}) \
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
        and (___check_obj_id(L['mod'].albert.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].albert.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].albert.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].albert.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['mod'].albert.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['mod'].albert.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['mod'].albert.forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['mod'].albert.forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['mod'].albert.forward.__defaults__[8], 7628576)) \
        and (___check_obj_id(L['mod'].albert.get_head_mask.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].albert.encoder.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].albert.encoder.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].albert.encoder.forward.__defaults__[2], 7677632)) \
        and (___check_obj_id(L['mod'].albert.encoder.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].albert.encoder.forward.__defaults__[4], 7677664)) \
        and (___check_obj_id(L['mod'].albert.embeddings.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].albert.embeddings.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].albert.embeddings.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['mod'].albert.embeddings.forward.__defaults__[3], 7628576)) \
        and (___check_type_id(L['mod'].albert.embeddings.forward.__defaults__[4], 7640416)) \
        and (L['mod'].albert.embeddings.forward.__defaults__[4] == 0) \
        and (___check_obj_id(L['mod'].albert.encoder.albert_layer_groups[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].albert.encoder.albert_layer_groups[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].albert.encoder.albert_layer_groups[0].forward.__defaults__[2], 7677632)) \
        and (___check_obj_id(L['mod'].albert.encoder.albert_layer_groups[0].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].albert.encoder.albert_layer_groups[0].albert_layers[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].albert.encoder.albert_layer_groups[0].albert_layers[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].albert.encoder.albert_layer_groups[0].albert_layers[0].forward.__defaults__[2], 7677632)) \
        and (___check_obj_id(L['mod'].albert.encoder.albert_layer_groups[0].albert_layers[0].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['mod'].albert.encoder.albert_layer_groups[0].albert_layers[0].attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].albert.encoder.albert_layer_groups[0].albert_layers[0].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['mod'].albert.encoder.albert_layer_groups[0].albert_layers[0].attention.forward.__defaults__[2], 7677632)) \
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
    return importlib.import_module('transformers.modeling_outputs').MaskedLMOutput(
        loss=graph_out_0[0], logits=graph_out_0[1], hidden_states=None,
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
