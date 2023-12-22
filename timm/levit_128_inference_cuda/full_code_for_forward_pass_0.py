
def __guard_0_for_forward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['mod'], 140145458337712)) \
        and (L['mod'].training == False) \
        and (___check_type_id(L['self'], 32001232)) \
        and (___check_type_id(L['inputs'], 7642176)) \
        and (len(L['inputs']) == 1) \
        and (hasattr(L['inputs'][0], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['self'].autocast, 31074720)) \
        and (___check_obj_id(L['mod'].forward_head.__defaults__[0], 7677632)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140144480549568))) \
        and (___compile_config_hash() == 'e7188dc6131643d03ab9cf0d0e17b22a') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_tensors(L['inputs'][0], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_0*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_0(*args, **kwargs):
    pass

def __transformed_code_0_for_forward_pass(self, mod, inputs, collect_outputs):
    graph_out_0 = __compiled_fn_0(inputs[0])
    getattr(getattr(mod.stages, '2').blocks, '3').attn.attention_bias_cache.clear()
    getattr(getattr(mod.stages, '2').blocks, '3').attn.attention_bias_cache.update(
        {'cuda:0': graph_out_0[14]})
    getattr(getattr(mod.stages, '2').blocks, '2').attn.attention_bias_cache.clear()
    getattr(getattr(mod.stages, '2').blocks, '2').attn.attention_bias_cache.update(
        {'cuda:0': graph_out_0[13]})
    getattr(getattr(mod.stages, '2').blocks, '1').attn.attention_bias_cache.clear()
    getattr(getattr(mod.stages, '2').blocks, '1').attn.attention_bias_cache.update(
        {'cuda:0': graph_out_0[12]})
    getattr(getattr(mod.stages, '2').blocks, '0').attn.attention_bias_cache.clear()
    getattr(getattr(mod.stages, '2').blocks, '0').attn.attention_bias_cache.update(
        {'cuda:0': graph_out_0[11]})
    getattr(mod.stages, '2').downsample.attn_downsample.attention_bias_cache.clear(
        )
    getattr(mod.stages, '2'
        ).downsample.attn_downsample.attention_bias_cache.update({'cuda:0':
        graph_out_0[10]})
    getattr(getattr(mod.stages, '1').blocks, '3').attn.attention_bias_cache.clear()
    getattr(getattr(mod.stages, '1').blocks, '3').attn.attention_bias_cache.update(
        {'cuda:0': graph_out_0[9]})
    getattr(getattr(mod.stages, '1').blocks, '2').attn.attention_bias_cache.clear()
    getattr(getattr(mod.stages, '1').blocks, '2').attn.attention_bias_cache.update(
        {'cuda:0': graph_out_0[8]})
    getattr(getattr(mod.stages, '1').blocks, '1').attn.attention_bias_cache.clear()
    getattr(getattr(mod.stages, '1').blocks, '1').attn.attention_bias_cache.update(
        {'cuda:0': graph_out_0[7]})
    getattr(getattr(mod.stages, '1').blocks, '0').attn.attention_bias_cache.clear()
    getattr(getattr(mod.stages, '1').blocks, '0').attn.attention_bias_cache.update(
        {'cuda:0': graph_out_0[6]})
    getattr(mod.stages, '1').downsample.attn_downsample.attention_bias_cache.clear(
        )
    getattr(mod.stages, '1'
        ).downsample.attn_downsample.attention_bias_cache.update({'cuda:0':
        graph_out_0[5]})
    getattr(getattr(mod.stages, '0').blocks, '3').attn.attention_bias_cache.clear()
    getattr(getattr(mod.stages, '0').blocks, '3').attn.attention_bias_cache.update(
        {'cuda:0': graph_out_0[4]})
    getattr(getattr(mod.stages, '0').blocks, '2').attn.attention_bias_cache.clear()
    getattr(getattr(mod.stages, '0').blocks, '2').attn.attention_bias_cache.update(
        {'cuda:0': graph_out_0[3]})
    getattr(getattr(mod.stages, '0').blocks, '1').attn.attention_bias_cache.clear()
    getattr(getattr(mod.stages, '0').blocks, '1').attn.attention_bias_cache.update(
        {'cuda:0': graph_out_0[2]})
    getattr(getattr(mod.stages, '0').blocks, '0').attn.attention_bias_cache.clear()
    getattr(getattr(mod.stages, '0').blocks, '0').attn.attention_bias_cache.update(
        {'cuda:0': graph_out_0[1]})
    return graph_out_0[0]


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward_pass(self, mod, inputs, collect_outputs):
    with self.autocast() as __temp_48:
        return mod(*inputs)
    return None

def transformed_forward_pass(self, mod, inputs, collect_outputs):
    L = {"self": self, "mod": mod, "inputs": inputs, "collect_outputs": collect_outputs}
    if __guard_0_for_forward_pass(L):
        return __transformed_code_0_for_forward_pass(self, mod, inputs, collect_outputs)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward_pass(self, mod, inputs, collect_outputs)

#============ end of forward_pass ============#
