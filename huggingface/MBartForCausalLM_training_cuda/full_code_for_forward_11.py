
def __guard_13_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 139707509242896)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677664)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (___check_obj_id(L['encoder_hidden_states'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139704230551056))) \
        and (___compile_config_hash() == 'd496b5c37a955fd1e53ea51d6c55fc22') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_19*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_19(*args, **kwargs):
    pass

def __transformed_code_13_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_19(hidden_states, attention_mask)
    return graph_out_0[0], (graph_out_0[1], graph_out_0[2])


def __guard_12_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 139707509234016)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677664)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (___check_obj_id(L['encoder_hidden_states'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139704230551056))) \
        and (___compile_config_hash() == 'd496b5c37a955fd1e53ea51d6c55fc22') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_18*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_18(*args, **kwargs):
    pass

def __transformed_code_12_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_18(hidden_states, attention_mask)
    return graph_out_0[0], (graph_out_0[1], graph_out_0[2])


def __guard_11_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 139707509239488)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677664)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (___check_obj_id(L['encoder_hidden_states'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139704230551056))) \
        and (___compile_config_hash() == 'd496b5c37a955fd1e53ea51d6c55fc22') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_17*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_17(*args, **kwargs):
    pass

def __transformed_code_11_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_17(hidden_states, attention_mask)
    return graph_out_0[0], (graph_out_0[1], graph_out_0[2])


def __guard_10_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 139707507562368)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677664)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (___check_obj_id(L['encoder_hidden_states'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139704230551056))) \
        and (___compile_config_hash() == 'd496b5c37a955fd1e53ea51d6c55fc22') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_16*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_16(*args, **kwargs):
    pass

def __transformed_code_10_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_16(hidden_states, attention_mask)
    return graph_out_0[0], (graph_out_0[1], graph_out_0[2])


def __guard_9_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 139707507556512)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677664)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (___check_obj_id(L['encoder_hidden_states'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139704230551056))) \
        and (___compile_config_hash() == 'd496b5c37a955fd1e53ea51d6c55fc22') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_15*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_15(*args, **kwargs):
    pass

def __transformed_code_9_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_15(hidden_states, attention_mask)
    return graph_out_0[0], (graph_out_0[1], graph_out_0[2])


def __guard_8_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 139707507559344)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677664)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (___check_obj_id(L['encoder_hidden_states'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139704230551056))) \
        and (___compile_config_hash() == 'd496b5c37a955fd1e53ea51d6c55fc22') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_14*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_14(*args, **kwargs):
    pass

def __transformed_code_8_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_14(hidden_states, attention_mask)
    return graph_out_0[0], (graph_out_0[1], graph_out_0[2])


def __guard_7_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 139707507563232)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677664)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (___check_obj_id(L['encoder_hidden_states'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139704230551056))) \
        and (___compile_config_hash() == 'd496b5c37a955fd1e53ea51d6c55fc22') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_13*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_13(*args, **kwargs):
    pass

def __transformed_code_7_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_13(hidden_states, attention_mask)
    return graph_out_0[0], (graph_out_0[1], graph_out_0[2])


def __guard_6_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 139707507569328)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677664)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (___check_obj_id(L['encoder_hidden_states'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139704230551056))) \
        and (___compile_config_hash() == 'd496b5c37a955fd1e53ea51d6c55fc22') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_12*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_12(*args, **kwargs):
    pass

def __transformed_code_6_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_12(hidden_states, attention_mask)
    return graph_out_0[0], (graph_out_0[1], graph_out_0[2])


def __guard_5_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 139707507570672)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677664)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (___check_obj_id(L['encoder_hidden_states'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139704230551056))) \
        and (___compile_config_hash() == 'd496b5c37a955fd1e53ea51d6c55fc22') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_11*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_11(*args, **kwargs):
    pass

def __transformed_code_5_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_11(hidden_states, attention_mask)
    return graph_out_0[0], (graph_out_0[1], graph_out_0[2])


def __guard_4_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 139707507572064)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677664)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (___check_obj_id(L['encoder_hidden_states'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139704230551056))) \
        and (___compile_config_hash() == 'd496b5c37a955fd1e53ea51d6c55fc22') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_10*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_10(*args, **kwargs):
    pass

def __transformed_code_4_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_10(hidden_states, attention_mask)
    return graph_out_0[0], (graph_out_0[1], graph_out_0[2])


def __guard_3_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 139707507570960)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677664)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (___check_obj_id(L['encoder_hidden_states'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139704230551056))) \
        and (___compile_config_hash() == 'd496b5c37a955fd1e53ea51d6c55fc22') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_9*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_9(*args, **kwargs):
    pass

def __transformed_code_3_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_9(hidden_states, attention_mask)
    return graph_out_0[0], (graph_out_0[1], graph_out_0[2])


def __guard_2_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 139707507571536)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677664)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (___check_obj_id(L['encoder_hidden_states'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139704230551056))) \
        and (___compile_config_hash() == 'd496b5c37a955fd1e53ea51d6c55fc22') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_8*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_8(*args, **kwargs):
    pass

def __transformed_code_2_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_8(hidden_states, attention_mask)
    return graph_out_0[0], (graph_out_0[1], graph_out_0[2])


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    residual = hidden_states
    hidden_states = self.self_attn_layer_norm(hidden_states)
    if past_key_value is not None:
        self_attn_past_key_value = past_key_value[slice(None, 2)]
    else:
        self_attn_past_key_value = None
    __temp_608 = self.self_attn(hidden_states=hidden_states, past_key_value=
        self_attn_past_key_value, attention_mask=attention_mask,
        layer_head_mask=layer_head_mask, output_attentions=output_attentions)
    hidden_states = __temp_608[0]
    self_attn_weights = __temp_608[1]
    present_key_value = __temp_608[2]
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout,
        training=self.training)
    hidden_states = residual + hidden_states
    cross_attn_present_key_value = None
    cross_attn_weights = None
    if encoder_hidden_states is not None:
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        if past_key_value is not None:
            cross_attn_past_key_value = past_key_value[slice(-2, None)]
        else:
            cross_attn_past_key_value = None
        __temp_611 = self.encoder_attn(hidden_states=hidden_states,
            key_value_states=encoder_hidden_states, attention_mask=
            encoder_attention_mask, layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=cross_attn_past_key_value, output_attentions=
            output_attentions)
        hidden_states = __temp_611[0]
        cross_attn_weights = __temp_611[1]
        cross_attn_present_key_value = __temp_611[2]
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout,
            training=self.training)
        hidden_states = residual + hidden_states
        present_key_value = present_key_value + cross_attn_present_key_value
    residual = hidden_states
    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.activation_fn(self.fc1(hidden_states))
    hidden_states = nn.functional.dropout(hidden_states, p=self.
        activation_dropout, training=self.training)
    hidden_states = self.fc2(hidden_states)
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout,
        training=self.training)
    hidden_states = residual + hidden_states
    outputs = hidden_states,
    if output_attentions:
        outputs += self_attn_weights, cross_attn_weights
    if use_cache:
        outputs += present_key_value,
    return outputs

def transformed_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    L = {"self": self, "hidden_states": hidden_states, "attention_mask": attention_mask, "encoder_hidden_states": encoder_hidden_states, "encoder_attention_mask": encoder_attention_mask, "layer_head_mask": layer_head_mask, "cross_attn_layer_head_mask": cross_attn_layer_head_mask, "past_key_value": past_key_value, "output_attentions": output_attentions, "use_cache": use_cache}
    if __guard_13_for_forward(L):
        return __transformed_code_13_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_12_for_forward(L):
        return __transformed_code_12_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_11_for_forward(L):
        return __transformed_code_11_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_10_for_forward(L):
        return __transformed_code_10_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_9_for_forward(L):
        return __transformed_code_9_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_8_for_forward(L):
        return __transformed_code_8_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_7_for_forward(L):
        return __transformed_code_7_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_6_for_forward(L):
        return __transformed_code_6_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_5_for_forward(L):
        return __transformed_code_5_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_4_for_forward(L):
        return __transformed_code_4_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_3_for_forward(L):
        return __transformed_code_3_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_2_for_forward(L):
        return __transformed_code_2_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)

#============ end of forward ============#
